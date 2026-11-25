#pragma once
// OrderNet (Indra OCR's order.onnx, 0.73M params) on TensorRT, FP32: feats [1,N,16] + mask [1,N]
// bool -> a [1,N,N] (pairwise order), cls [1,N,10] (block class), same [1,N,N] (same block),
// tab [1,3,N,N] (table relations; computed by the graph, not used). N is dynamic, 2..kMaxLines.

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/common/cuda/cuda_ptr.h"
#include "core/engine/trt/trt_engine.h"
#include "ordernet/order.h"

namespace hp::on {

// Pages with more lines keep the geometric (XY-cut) order. Context memory at 2048: ~210 MB.
inline constexpr int kMaxLines = 2048;

class OrderNet {
 public:
  bool load(const std::string &plan_path);
  // One page; blocks until the heads are on the host.
  Heads forward(const std::vector<float> &feats, int n, cudaStream_t stream);

 private:
  void reserve(int n);

  std::unique_ptr<hp::core::engine::TrtEngine> engine_;
  int cap_ = 0;
  hp::core::CudaPtr<float> d_feats_, d_a_, d_cls_, d_same_, d_tab_;
  hp::core::CudaPtr<uint8_t> d_mask_;
};

}  // namespace hp::on
