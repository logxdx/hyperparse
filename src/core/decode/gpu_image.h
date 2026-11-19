#pragma once

#include <cstddef>

namespace hp::core::decode {

// Lightweight non-owning GPU image descriptor (replaces cv::cuda::GpuMat).
// Supports designated initializers: GpuImage{.data = ptr, .step = s, .rows = h, .cols = w}
struct GpuImage {
  void *data = nullptr;
  std::size_t step = 0;
  int rows = 0;
  int cols = 0;

  [[nodiscard]] constexpr bool empty() const noexcept { return data == nullptr || rows == 0 || cols == 0; }
};

} // namespace hp::core::decode

// Convenience alias at hp::core level
namespace hp::core {
using decode::GpuImage;
} // namespace hp::core
