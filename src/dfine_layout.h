#pragma once
// D-FINE (OXR stage2_b2, 12 classes) on TensorRT. The engine carries the decode
// (sigmoid + top-300): scores [B,300], labels [B,300] int32, boxes [B,300,4]
// xyxy in letterbox-canvas pixels. Input is written in place by multi_resize.

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "multi_resize.h"
#include "core/common/cuda/cuda_ptr.h"
#include "core/engine/trt/trt_engine.h"

namespace hp {

inline constexpr std::array<std::string_view, 12> kDfineLabels = {
    "Caption", "Footnote", "Formula", "Text", "Page-footer", "Page-header",
    "Picture", "Title", "Table", "Code Block", "Stamp", "Chemical Structure"};

// Nearest PP-DocLayoutV3 label (the class vocabulary of core/layout), per D-FINE class.
inline constexpr std::array<std::string_view, 12> kDfineToPP = {
    "figure_title", "footnote", "display_formula", "text", "footer", "header",
    "image", "paragraph_title", "table", "algorithm", "seal", "image"};

struct LayoutBox {
  int cls = 0;  // D-FINE class id
  float score = 0.f;
  float x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // page pixels
};

class DfineLayout {
 public:
  static constexpr int kTopK = 300;

  bool load(const std::string &plan_path, int size);
  [[nodiscard]] int size() const { return size_; }

  // Spec for multi_resize (writes straight into the engine input) + remembers the
  // letterbox so collect() can map boxes back to page pixels.
  ResizeOut resize_spec(int page_w, int page_h);

  // Execute and queue the (small) D2H copies. Call collect() after the stream syncs.
  void enqueue(cudaStream_t stream);
  [[nodiscard]] std::vector<LayoutBox> collect(float score_thresh = 0.5f) const;

 private:
  int size_ = 960;
  int page_w_ = 0, page_h_ = 0;
  Letterbox lb_;
  std::unique_ptr<hp::core::engine::TrtEngine> engine_;
  hp::core::CudaPtr<float> d_in_, d_scores_, d_boxes_;
  hp::core::CudaPtr<int> d_labels_;
  hp::core::CudaHostPtr<float> h_scores_, h_boxes_;
  hp::core::CudaHostPtr<int> h_labels_;
};

}  // namespace hp
