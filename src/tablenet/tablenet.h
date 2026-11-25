#pragma once
// LayoutNet (OCRia-nano-1 tablenet.onnx: 6-layer transformer over text units + CNN image branch,
// 4.27M params) on TensorRT, FP16 or INT8 (tablenet_<p>.plan). Inputs feats [1,N,68], mask [1,N] bool, pairf [1,N,N,2],
// image [1,1,1024,1024], boxes [1,N,4], valid [1,2]; outputs node [1,N,2], pair [1,5,N,N],
// cls [1,N,13]. N is dynamic, 2..kMaxUnits (one optimization profile).

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/common/cuda/cuda_ptr.h"
#include "core/engine/trt/trt_engine.h"
#include "tablenet/decode.h"
#include "tablenet/features.h"

namespace hp::tn {

class TableNet {
 public:
  bool load(const std::string &plan_path);
  // One page; blocks until the logits are on the host.
  Logits forward(const std::vector<float> &feats, const std::vector<float> &pairf, int n, const ImgInput &img,
                 cudaStream_t stream);

 private:
  void reserve(int n);

  std::unique_ptr<hp::core::engine::TrtEngine> engine_;
  int cap_ = 0;  // units the per-N buffers hold
  int n_pair_ = 5, n_cls_ = 13;
  hp::core::CudaPtr<float> d_feats_, d_pairf_, d_image_, d_boxes_, d_valid_, d_node_, d_pair_, d_cls_;
  hp::core::CudaPtr<uint8_t> d_mask_;
};

}  // namespace hp::tn
