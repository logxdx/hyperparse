#pragma once
// PP-FormulaNet-S formula recognition on TensorRT, as three graphs: encoder /
// cross-KV prep / 3-token decode step (models/formula). The decode loop is greedy,
// 3 tokens per step at pos = 3*it, with a static 1056-slot self-attention KV
// ping-pong and an EOS scan every 16 steps, on TRT enqueueV3. Preprocess is
// OpenCV-free (ink trim -> PIL-exact bicubic -> gray -> centre pad).

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/common/cuda/cuda_ptr.h"
#include "core/engine/trt/trt_engine.h"
#include "core/formula/ppformulanet/formula_tokenizer.h"

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
  static constexpr int kMaxBatch = 8;
  // encoder.plan, prep.plan, step_fp16.plan, tokenizer.json
  bool load(const std::string &dir);
  // LaTeX per crop ("" when nothing decoded).
  std::vector<std::string> run(const std::vector<FormulaCrop> &crops, cudaStream_t stream);
  int steps_run = 0;  // stats: decode steps executed
  std::string dump_dir;  // if set: x_<n>.f32 (encoder input) + tok_<n>.txt per crop, for reference checks
  int dump_n = 0;

 private:
  void decode_chunk(int B, std::vector<std::vector<int64_t>> &out, cudaStream_t s);
  std::unique_ptr<hp::core::engine::TrtEngine> enc_, prep_, step_;
  std::optional<hp::core::formula::FormulaTokenizer> tok_;
  std::string enc_out_;
  hp::core::CudaHostPtr<float> h_x_;
  hp::core::CudaPtr<float> d_x_, d_mem_, d_ck_, d_cv_, d_kA_, d_kB_, d_vA_, d_vB_, d_logits_;
  hp::core::CudaPtr<int64_t> d_tok_, d_pos_, d_all_;
  hp::core::CudaPtr<unsigned char> d_done_;
};

}  // namespace hp
