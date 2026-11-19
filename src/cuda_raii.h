#pragma once
// Move-only owners for CUDA streams and events, so every GPU object in a
// ModelSet is released by its destructor (unload == destroy the ModelSet).

#include <utility>

#include <cuda_runtime.h>

#include "core/common/cuda/cuda_check.h"

namespace hp {

class CudaStream {
 public:
  CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
  ~CudaStream() {
    if (s_) cudaStreamDestroy(s_);
  }
  CudaStream(CudaStream &&o) noexcept : s_(std::exchange(o.s_, nullptr)) {}
  CudaStream &operator=(CudaStream &&o) noexcept {
    std::swap(s_, o.s_);
    return *this;
  }
  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  operator cudaStream_t() const { return s_; }  // NOLINT: drop-in for the raw handle
  void sync() const { CUDA_CHECK(cudaStreamSynchronize(s_)); }

 private:
  cudaStream_t s_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() { CUDA_CHECK(cudaEventCreateWithFlags(&e_, cudaEventDisableTiming)); }
  ~CudaEvent() {
    if (e_) cudaEventDestroy(e_);
  }
  CudaEvent(CudaEvent &&o) noexcept : e_(std::exchange(o.e_, nullptr)) {}
  CudaEvent &operator=(CudaEvent &&o) noexcept {
    std::swap(e_, o.e_);
    return *this;
  }
  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;

  operator cudaEvent_t() const { return e_; }  // NOLINT: drop-in for the raw handle

 private:
  cudaEvent_t e_ = nullptr;
};

}  // namespace hp
