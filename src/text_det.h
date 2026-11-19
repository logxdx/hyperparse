#pragma once
// PP-OCR DB text detector, all on GPU: TRT engine -> threshold -> CCL -> JFA
// per-component unclip -> oriented boxes. Same kernels and constants as
// PaddleDet's GPU_CCL=2 path (core/kernels), without
// OpenCV and without its own resize: the input is written by multi_resize.

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "multi_resize.h"
#include "core/common/cuda/cuda_ptr.h"
#include "core/common/geometry/box.h"
#include "core/detection/det_config.h"
#include "core/engine/trt/trt_engine.h"
#include "core/kernels/kernels.h"

namespace hp {

class TextDet {
 public:
  bool load(const std::string &plan_path, int max_side = 1280,
            const hp::core::detection::DbParams &db = hp::core::detection::kDbDefaults);
  [[nodiscard]] int max_side() const { return max_side_; }

  ResizeOut resize_spec(int page_w, int page_h);
  // Packed-region mode: the caller writes a [3, h, w] canvas (CropBatch::add_det_region)
  // into the returned buffer; run() then returns boxes in canvas pixels.
  float *canvas_input(int w, int h);
  // Full-page det scale for a page (what resize_spec would use), for region packing.
  [[nodiscard]] float page_scale(int page_w, int page_h) const;

  // Execute + postprocess on `stream`; syncs the stream once (box count/coords D2H).
  [[nodiscard]] std::vector<hp::core::Box> run(cudaStream_t stream);

 private:
  int max_side_ = 1280;
  int page_w_ = 0, page_h_ = 0, in_w_ = 0, in_h_ = 0;
  hp::core::detection::DbParams db_{};
  std::unique_ptr<hp::core::engine::TrtEngine> engine_;

  hp::core::CudaPtr<float> d_in_, d_pred_;
  hp::core::CudaPtr<uint8_t> d_bitmap_;
  hp::core::CudaPtr<int> d_labels_, d_compact_ids_, d_id_counter_, d_num_boxes_, d_perim_;
  hp::core::CudaPtr<hp::core::kernels::GpuDetBox> d_bboxes_;
  hp::core::CudaPtr<uint32_t> d_jfa_labels_, d_jfa_seeds_, d_jfa_seeds_alt_;
  hp::core::CudaPtr<float> d_expand_, d_orient_;
  hp::core::CudaPtr<unsigned long long> d_moments_;
  hp::core::CudaHostPtr<hp::core::kernels::GpuDetBox> h_boxes_, h_exp_boxes_;
};

}  // namespace hp
