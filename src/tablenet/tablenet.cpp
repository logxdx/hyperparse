#include "tablenet/tablenet.h"

#include <stdexcept>

#include "core/common/cuda/cuda_check.h"

namespace hp::tn {
namespace {

nvinfer1::Dims dims(std::initializer_list<int64_t> v) {
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int32_t>(v.size());
  int i = 0;
  for (int64_t x : v) d.d[i++] = x;
  return d;
}

}  // namespace

bool TableNet::load(const std::string &plan_path) {
  engine_ = std::make_unique<hp::core::engine::TrtEngine>(plan_path);
  if (!engine_->load()) return false;
  d_image_ = hp::core::CudaPtr<float>(static_cast<size_t>(kImgSide) * kImgSide);
  d_valid_ = hp::core::CudaPtr<float>(2);
  engine_->set_tensor_address("image", d_image_.get());
  engine_->set_tensor_address("valid", d_valid_.get());
  if (!engine_->set_input_shape("image", dims({1, 1, kImgSide, kImgSide})) ||
      !engine_->set_input_shape("valid", dims({1, 2})))
    return false;
  reserve(256);
  return true;
}

void TableNet::reserve(int n) {
  if (n <= cap_) return;
  int cap = std::max(cap_, 64);
  while (cap < n) cap *= 2;
  cap = std::min<int>(std::max(cap, n), static_cast<int>(kMaxUnits));
  const auto c = static_cast<size_t>(cap);
  d_feats_.reset(c * kFeatDim);
  d_mask_.reset(c);
  d_pairf_.reset(c * c * 2);
  d_boxes_.reset(c * 4);
  d_node_.reset(c * 2);
  d_pair_.reset(static_cast<size_t>(n_pair_) * c * c);
  d_cls_.reset(c * n_cls_);
  // every mask entry stays true (the reference always passes an all-true mask)
  CUDA_CHECK(cudaMemset(d_mask_.get(), 1, c));
  engine_->set_tensor_address("feats", d_feats_.get());
  engine_->set_tensor_address("mask", d_mask_.get());
  engine_->set_tensor_address("pairf", d_pairf_.get());
  engine_->set_tensor_address("boxes", d_boxes_.get());
  engine_->set_tensor_address("node", d_node_.get());
  engine_->set_tensor_address("pair", d_pair_.get());
  engine_->set_tensor_address("cls", d_cls_.get());
  cap_ = cap;
}

Logits TableNet::forward(const std::vector<float> &feats, const std::vector<float> &pairf, int n, const ImgInput &img,
                         cudaStream_t stream) {
  if (n < 2 || n > static_cast<int>(kMaxUnits)) throw std::invalid_argument("LayoutNet: unit count out of range");
  std::lock_guard lock(mu_);  // one context and one set of buffers
  reserve(n);
  const auto N = static_cast<int64_t>(n);
  if (!engine_->set_input_shape("feats", dims({1, N, kFeatDim})) || !engine_->set_input_shape("mask", dims({1, N})) ||
      !engine_->set_input_shape("pairf", dims({1, N, N, 2})) || !engine_->set_input_shape("boxes", dims({1, N, 4})))
    throw std::runtime_error("LayoutNet: cannot set input shapes");
  const auto nn = static_cast<size_t>(n);
  CUDA_CHECK(cudaMemcpyAsync(d_feats_.get(), feats.data(), nn * kFeatDim * sizeof(float), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_pairf_.get(), pairf.data(), nn * nn * 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_image_.get(), img.canvas.data(), img.canvas.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_boxes_.get(), img.boxes.data(), nn * 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_valid_.get(), img.valid, 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
  if (!engine_->execute(stream)) throw std::runtime_error("LayoutNet: execute failed");
  const nvinfer1::Dims ps = engine_->tensor_shape("pair"), cs = engine_->tensor_shape("cls");
  Logits l;
  l.n = n;
  l.n_pair = ps.nbDims == 4 ? static_cast<int>(ps.d[1]) : n_pair_;
  l.n_cls = cs.nbDims == 3 ? static_cast<int>(cs.d[2]) : n_cls_;
  l.node.resize(nn * 2);
  l.pair.resize(static_cast<size_t>(l.n_pair) * nn * nn);
  l.cls.resize(nn * l.n_cls);
  CUDA_CHECK(cudaMemcpyAsync(l.node.data(), d_node_.get(), l.node.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(l.pair.data(), d_pair_.get(), l.pair.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(l.cls.data(), d_cls_.get(), l.cls.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return l;
}

}  // namespace hp::tn
