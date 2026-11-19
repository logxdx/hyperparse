#pragma once
// Fused ragged crop: ONE launch cuts every region a page needs (rec text lines,
// table crops, ...) out of the full-resolution u8 BGR page that is already on
// the GPU, and writes each straight into its model's input tensor.
//
// Ragged batching: each crop is tiled in 32x8 blocks; blockIdx.x is a global tile
// id and the block finds its crop by binary search over per-crop tile offsets,
// so a 48x320 rec line and a 488x488 table crop share one grid with no padding
// work between them.
//
// Per-mode arithmetic is copied verbatim from the core/kernels launches it replaces,
// so outputs are bit-identical (tools: crop_batch_test):
//   kRecWarp   == kernels::cuda_batch_roi_warp   (perspective, border replicate,
//                 zero pad past the line's own width)
//   kRectResize== table_kernels.cu slanext_pre    (half-pixel resize of a page
//                 rect, index clamp inside the rect, constant pad)

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/common/cuda/cuda_ptr.h"
#include "core/common/geometry/box.h"

namespace hp {

enum CropMode : int { kRecWarp = 0, kRectResize = 1 };

struct CropDesc {
  float M[9];          // kRecWarp: dst->src perspective (page px). kRectResize: M[0]=scale_x, M[4]=scale_y
  int rect[4];         // kRectResize: source rect x, y, w, h (sampling clamps inside it)
  float *dst;          // CHW base for this crop (its top-left pixel in plane 0)
  int dst_w, dst_h;    // crop size in the tensor
  int row_stride;      // floats per tensor row (== dst_w unless packed into a larger canvas)
  long long plane_stride;  // floats per channel plane
  int valid_w, valid_h;  // sampled region; outside -> pad
  int src_ch[3];       // page channel (0=B,1=G,2=R) feeding output channel c
  float mean[3], inv_std[3], pad[3];
  int mode;
  int tile_begin;      // filled by CropBatch::launch
};

class CropBatch {
 public:
  void clear() { h_.clear(); }
  [[nodiscard]] size_t size() const { return h_.size(); }

  // Rec line: PaddleRec geometry (compute_crop_transform to h=48, width W padded).
  void add_rec_line(const hp::core::Box &box, float *dst, int dst_w, int dst_h = 48);
  // SLANeXt table crop: ResizeByLong(dst) of page rect, ImageNet norm, BGR CHW, pad (0-mean)/std.
  void add_table(int x, int y, int w, int h, float *dst, int dst_size = 488);
  // Det region packing: page rect resized by `scale` into a [3, canvas_h, canvas_w]
  // det input at (ox, oy); BGR, ImageNet mean/std (same normalisation as det_spec).
  // Returns the placed size.
  void add_det_region(int x, int y, int w, int h, float scale, float *canvas, int canvas_w, int canvas_h,
                      int ox, int oy, int &out_w, int &out_h);
  void add(const CropDesc &d) { h_.push_back(d); }

  // One H2D of the descriptor table + one kernel launch.
  void launch(const uint8_t *page, size_t step, int page_w, int page_h, cudaStream_t stream);

 private:
  std::vector<CropDesc> h_;
  hp::core::CudaHostPtr<CropDesc> h_pinned_;
  hp::core::CudaPtr<CropDesc> d_;
  size_t cap_ = 0;
};

}  // namespace hp
