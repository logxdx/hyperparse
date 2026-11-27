#pragma once
// PP-FormulaNet-S formula recognition on TensorRT, as three graphs: encoder /
// cross-KV prep / 3-token decode step (models/formula). The decode is greedy,
// 3 tokens per step at pos = 3*it, with a static 1056-slot self-attention KV
// on TRT enqueueV3. Preprocess is OpenCV-free (ink trim -> PIL-exact bicubic ->
// gray -> centre pad).
//
// Batching. The step graph always runs at batch kBatch: its TensorRT kernels differ by batch
// size (FP16 near-ties then flip a token now and then), while a row's tokens never depend on
// the other rows of the batch. So any crop decodes the same whatever it shares a batch with,
// and rows can be moved between batches as long as they are at the same position:
//   - crops are taken longest-expected first (ink pixels) in cohorts of kGroups * kBatch,
//     which start together and step together, kGroups step batches per step;
//   - when rows finish, the rows left are packed into fewer batches (their self- and
//     cross-attention KV and next tokens are copied over), and empty batches stop;
//   - EOS is seen kEosLag steps late without stalling the GPU (the flags go to pinned memory);
//   - the next cohort's crops are preprocessed, encoded (one crop at a time: a batched conv
//     drifts near-tie tokens) and prepped on a second stream during this cohort's decode.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/common/cuda/cuda_ptr.h"
#include "core/engine/trt/trt_engine.h"
#include "core/formula/ppformulanet/formula_tokenizer.h"
#include "cuda_raii.h"

namespace hp {

inline constexpr int kFormulaS = 384;

// BGR8 crop (row stride `step` bytes) -> [384*384] float encoder input.
void formula_preprocess(const uint8_t *bgr, size_t step, int w, int h, float *out);

struct FormulaCrop {
  const uint8_t *bgr;  // top-left of the crop in host memory
  size_t step;         // bytes per source row
  int w, h;
};

class FormulaRecognizer {
 public:
  static constexpr int kBatch = 8;   // rows per step graph launch (the plan's opt/max batch)
  static constexpr int kGroups = 4;  // step batches per cohort
  static constexpr int kCohort = kBatch * kGroups;
  // encoder.plan, prep.plan, step_fp16.plan, tokenizer.json
  bool load(const std::string &dir);
  // LaTeX per crop ("" when nothing decoded).
  std::vector<std::string> run(const std::vector<FormulaCrop> &crops, cudaStream_t stream);
  int steps_run = 0;  // stats: step graph launches (kBatch rows each)
  std::string dump_dir;  // if set: x_<n>.f32 (encoder input), c_<n>.bgr (int32 w, h, BGR rows), tok_<n>.txt per crop
  int dump_n = 0;

 private:
  static constexpr int kEosLag = 2;  // steps queued ahead of the EOS check
  // Encoding of one cohort (crops order[c0 .. c0 + n)) into cross-KV set `set`, a crop at a
  // time, so that it can interleave with a decode.
  struct EncodeJob {
    int c0 = 0, n = 0, set = 0, next = 0;
    [[nodiscard]] bool done() const { return next >= n; }
  };
  void encode_step(const std::vector<FormulaCrop> &crops, const std::vector<int> &order, EncodeJob &job);
  // Decodes the n crops of cross-KV set `set` into toks[0 .. n), advancing `next` meanwhile.
  void decode_cohort(int n, int set, std::vector<std::vector<int64_t>> &toks, cudaStream_t s,
                     const std::vector<FormulaCrop> &crops, const std::vector<int> &order, EncodeJob &next);

  std::unique_ptr<hp::core::engine::TrtEngine> enc_, prep_, step_;
  std::optional<hp::core::formula::FormulaTokenizer> tok_;
  std::string enc_out_;
  CudaStream enc_stream_;
  CudaEvent x_free_[2], set_ready_[2], set_free_[2], step_done_[kEosLag + 1];
  int x_half_ = 0;                    // pinned input half the next encoder batch fills
  hp::core::CudaHostPtr<float> h_x_;  // 2 x kBatch encoder inputs
  hp::core::CudaHostPtr<int> h_eos_;  // per cohort row: step + 1 of its first EOS, 0 = none yet (GPU-written)
  hp::core::CudaPtr<float> d_x_, d_mem_, d_ckv_, d_kv_, d_logits_;
  hp::core::CudaPtr<int64_t> d_tok_, d_pos_, d_hist_;
};

}  // namespace hp
