#pragma once
// PP-OCR CTC recognition with cache-aware batching.
//
// Measured on the RTX 4050 (rec_small_fp16, profile 0): per-column cost is flat
// (~0.75 us/column) while a batch holds <= ~10k input columns (batch x width) and
// roughly doubles beyond it (activations spill out of L2; e.g. 32x1600 costs 2.26
// ms/line vs 1.22 at 4x1600). The width-bucket PaddleRec batches up to 32 lines per
// bucket regardless of width, so wide-line batches land in the slow regime. Here
// lines are sorted by exact width and packed under a column budget; every batch's
// crops are cut in ONE CropBatch launch (bit-identical to batch_roi_warp), argmax
// for all batches lands in one buffer and comes back in one D2H.
//
// The plan outputs FP16 logits (PlanSpec::logits_output): the engine's softmax over
// 18,385 classes and its FP32 output were 20% of the engine time plus an FP32 argmax
// pass; cuda_argmax_logits takes the argmax and the softmax maximum in one FP16 pass.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "crop_batch.h"
#include "cuda_raii.h"
#include "text_recognizer.h"
#include "core/common/cuda/cuda_ptr.h"
#include "core/common/geometry/box.h"
#include "core/decode/gpu_image.h"
#include "core/engine/trt/trt_engine.h"

namespace hp {

class RecBatcher final : public TextRecognizer {
 public:
  static constexpr int kH = 48, kMaxBatch = 32, kColumnBudget = 8192, kMaxW = 4000;

  bool load(const std::string &plan, const std::string &dict);
  [[nodiscard]] Texts run(const hp::core::GpuImage &img, const std::vector<hp::core::Box> &boxes, cudaStream_t stream,
                          bool no_cjk) override;
  // Crops, every batch, argmax and the D2H of the class ids are queued; the CTC
  // decode waits for collect().
  void enqueue(const hp::core::GpuImage &img, std::span<const RecGroup> groups, cudaStream_t stream) override;
  [[nodiscard]] Texts collect() override;
  int batches_run = 0, lines_run = 0;

 private:
  std::unique_ptr<hp::core::engine::TrtEngine> engine_;
  std::vector<std::string> labels_;
  hp::core::CudaPtr<uint8_t> d_non_cjk_;  // class mask for no_cjk (script.h)
  int classes_ = 0;
  CropBatch crops_;
  hp::core::CudaPtr<float> d_in_, d_score_;
  hp::core::CudaPtr<uint16_t> d_out_;  // FP16 logits of one batch (logits plan)
  hp::core::CudaPtr<int> d_idx_;
  hp::core::CudaHostPtr<float> h_score_;
  hp::core::CudaHostPtr<int> h_idx_;
  size_t in_cap_ = 0, rows_cap_ = 0;

  // The enqueued work, for collect(). Batch lines [beg, end) index slot_out_.
  struct Batch {
    int beg, end, W;
    size_t in_off, row_off;
    int T;
    bool no_cjk;
  };
  std::vector<Batch> bs_;
  std::vector<int> slot_out_;  // packed line -> output index
  int n_out_ = 0;
  CudaEvent done_;
};

}  // namespace hp
