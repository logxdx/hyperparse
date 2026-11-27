#include "ordernet/net.h"

#include <algorithm>
#include <stdexcept>

#include "core/common/cuda/cuda_check.h"

namespace hp::on {
namespace {

nvinfer1::Dims dims(std::initializer_list<int64_t> v) {
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int32_t>(v.size());
  int i = 0;
  for (int64_t x : v) d.d[i++] = x;
  return d;
}

}  // namespace

bool OrderNet::load(const std::string &plan_path) {
  engine_ = std::make_unique<hp::core::engine::TrtEngine>(plan_path);
  if (!engine_->load()) return false;
  reserve(256);
  return true;
}

void OrderNet::reserve(int n) {
  if (n <= cap_) return;
  int cap = std::max(cap_, 64);
  while (cap < n) cap *= 2;
  cap = std::min(std::max(cap, n), kMaxLines);
  const auto c = static_cast<size_t>(cap);
  d_feats_.reset(c * kFeats);
  d_mask_.reset(c);
  d_a_.reset(c * c);
  d_cls_.reset(c * kClasses);
  d_same_.reset(c * c);
  d_tab_.reset(3 * c * c);
  CUDA_CHECK(cudaMemset(d_mask_.get(), 1, c));  // every line is real
  engine_->set_tensor_address("feats", d_feats_.get());
  engine_->set_tensor_address("mask", d_mask_.get());
  engine_->set_tensor_address("a", d_a_.get());
  engine_->set_tensor_address("cls", d_cls_.get());
  engine_->set_tensor_address("same", d_same_.get());
  engine_->set_tensor_address("tab", d_tab_.get());
  cap_ = cap;
}

Heads OrderNet::forward(const std::vector<float> &feats, int n, cudaStream_t stream) {
  if (n < 2 || n > kMaxLines) throw std::invalid_argument("OrderNet: line count out of range");
  std::lock_guard lock(mu_);  // one context and one set of buffers
  reserve(n);
  const auto N = static_cast<int64_t>(n);
  if (!engine_->set_input_shape("feats", dims({1, N, kFeats})) || !engine_->set_input_shape("mask", dims({1, N})))
    throw std::runtime_error("OrderNet: cannot set input shapes");
  const auto nn = static_cast<size_t>(n);
  CUDA_CHECK(cudaMemcpyAsync(d_feats_.get(), feats.data(), nn * kFeats * sizeof(float), cudaMemcpyHostToDevice, stream));
  if (!engine_->execute(stream)) throw std::runtime_error("OrderNet: execute failed");
  Heads h;
  h.n = n;
  h.a.resize(nn * nn);
  h.cls.resize(nn * kClasses);
  h.same.resize(nn * nn);
  CUDA_CHECK(cudaMemcpyAsync(h.a.data(), d_a_.get(), h.a.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h.cls.data(), d_cls_.get(), h.cls.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h.same.data(), d_same_.get(), h.same.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h;
}

}  // namespace hp::on
