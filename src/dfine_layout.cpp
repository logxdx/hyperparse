#include "dfine_layout.h"

#include <algorithm>
#include <stdexcept>

namespace hp {

bool DfineLayout::load(const std::string &plan_path, int size) {
  size_ = size;
  engine_ = std::make_unique<hp::core::engine::TrtEngine>(plan_path);
  if (!engine_->load()) return false;
  d_in_ = hp::core::CudaPtr<float>(static_cast<size_t>(3) * size * size);
  d_scores_ = hp::core::CudaPtr<float>(kTopK);
  d_boxes_ = hp::core::CudaPtr<float>(kTopK * 4);
  d_labels_ = hp::core::CudaPtr<int>(kTopK);
  h_scores_ = hp::core::CudaHostPtr<float>(kTopK);
  h_boxes_ = hp::core::CudaHostPtr<float>(kTopK * 4);
  h_labels_ = hp::core::CudaHostPtr<int>(kTopK);

  nvinfer1::Dims d{};
  d.nbDims = 4;
  d.d[0] = 1;
  d.d[1] = 3;
  d.d[2] = d.d[3] = size;
  if (!engine_->set_input_shape("images", d)) return false;
  engine_->set_tensor_address("images", d_in_.get());
  engine_->set_tensor_address("scores", d_scores_.get());
  engine_->set_tensor_address("labels", d_labels_.get());
  engine_->set_tensor_address("boxes", d_boxes_.get());
  return true;
}

ResizeOut DfineLayout::resize_spec(int page_w, int page_h) {
  page_w_ = page_w;
  page_h_ = page_h;
  return dfine_spec(page_w, page_h, size_, d_in_.get(), lb_);
}

void DfineLayout::enqueue(cudaStream_t stream) {
  if (!engine_->execute(stream)) throw std::runtime_error("D-FINE execute failed");
  CUDA_CHECK(cudaMemcpyAsync(h_scores_.get(), d_scores_.get(), kTopK * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_labels_.get(), d_labels_.get(), kTopK * sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_boxes_.get(), d_boxes_.get(), kTopK * 4 * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
}

std::vector<LayoutBox> DfineLayout::collect(float score_thresh) const {
  std::vector<LayoutBox> out;
  const float inv = 1.f / lb_.scale;
  auto cx = [&](float v) { return std::clamp((v - lb_.x0) * inv, 0.f, static_cast<float>(page_w_)); };
  auto cy = [&](float v) { return std::clamp((v - lb_.y0) * inv, 0.f, static_cast<float>(page_h_)); };
  for (int i = 0; i < kTopK; ++i) {
    const float s = h_scores_.get()[i];
    const int c = h_labels_.get()[i];
    if (s < score_thresh || c < 0 || c >= static_cast<int>(kDfineLabels.size())) continue;
    const float *b = h_boxes_.get() + 4 * i;
    LayoutBox lb{c, s, cx(b[0]), cy(b[1]), cx(b[2]), cy(b[3])};
    if (lb.x1 - lb.x0 < 1.f || lb.y1 - lb.y0 < 1.f) continue;
    out.push_back(lb);
  }
  std::ranges::sort(out, {}, [](const LayoutBox &b) { return -b.score; });
  return out;
}

}  // namespace hp
