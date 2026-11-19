#include "crop_batch.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/common/cuda/cuda_check.h"
#include "core/common/geometry/perspective.h"

namespace hp {
namespace {

constexpr int kTileW = 32, kTileH = 8;

__device__ __forceinline__ float px(const uint8_t *__restrict__ src, int step, int x, int y, int c) {
  return (float)__ldg(src + (size_t)y * step + x * 3 + c);
}

__global__ void __launch_bounds__(kTileW *kTileH)
crop_batch_kernel(const uint8_t *__restrict__ src, int src_step, int src_w, int src_h,
                  const CropDesc *__restrict__ descs, int n) {
  __shared__ CropDesc d;
  __shared__ int s_idx;
  const int tid = threadIdx.y * kTileW + threadIdx.x;
  if (tid == 0) {  // last crop whose tile_begin <= blockIdx.x
    int lo = 0, hi = n - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (descs[mid].tile_begin <= (int)blockIdx.x) lo = mid;
      else hi = mid - 1;
    }
    s_idx = lo;
  }
  __syncthreads();
  if (tid < (int)(sizeof(CropDesc) / 4))
    reinterpret_cast<int *>(&d)[tid] = reinterpret_cast<const int *>(descs + s_idx)[tid];
  __syncthreads();

  const int tiles_x = (d.dst_w + kTileW - 1) / kTileW;
  const int t = blockIdx.x - d.tile_begin;
  const int x = (t % tiles_x) * kTileW + threadIdx.x;
  const int y = (t / tiles_x) * kTileH + threadIdx.y;
  if (x >= d.dst_w || y >= d.dst_h) return;
  const size_t plane = (size_t)d.plane_stride;
  float *o = d.dst + (size_t)y * d.row_stride + x;

  if (x >= d.valid_w || y >= d.valid_h) {
    o[0] = d.pad[0];
    o[plane] = d.pad[1];
    o[2 * plane] = d.pad[2];
    return;
  }

  float c3[3];
  if (d.mode == kRecWarp) {
    // == core/kernels batch_roi_warp_kernel
    float denom = d.M[6] * x + d.M[7] * y + d.M[8];
    denom = copysignf(fmaxf(fabsf(denom), 1e-7f), denom);
    const float inv_denom = 1.0f / denom;
    float sx = (d.M[0] * x + d.M[1] * y + d.M[2]) * inv_denom;
    float sy = (d.M[3] * x + d.M[4] * y + d.M[5]) * inv_denom;
    if (!isfinite(sx) || !isfinite(sy)) {
      o[0] = 0.f;
      o[plane] = 0.f;
      o[2 * plane] = 0.f;
      return;
    }
    sx = fminf(fmaxf(sx, -1.0f), (float)(src_w + 1));
    sy = fminf(fmaxf(sy, -1.0f), (float)(src_h + 1));
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    int x1 = x0 + 1, y1 = y0 + 1;
    const float dx = sx - x0, dy = sy - y0;
    x0 = min(max(x0, 0), src_w - 1);
    x1 = min(max(x1, 0), src_w - 1);
    y0 = min(max(y0, 0), src_h - 1);
    y1 = min(max(y1, 0), src_h - 1);
    const float w00 = (1.0f - dx) * (1.0f - dy), w10 = dx * (1.0f - dy);
    const float w01 = (1.0f - dx) * dy, w11 = dx * dy;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int ch = d.src_ch[c];
      c3[c] = w00 * px(src, src_step, x0, y0, ch) + w10 * px(src, src_step, x1, y0, ch) +
              w01 * px(src, src_step, x0, y1, ch) + w11 * px(src, src_step, x1, y1, ch);
    }
  } else {
    // == table_kernels.cu bilinear_sample_norm (index clamp inside the rect)
    const float sx = (x + 0.5f) * d.M[0] - 0.5f;
    const float sy = (y + 0.5f) * d.M[4] - 0.5f;
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    const float fx = sx - x0, fy = sy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    const int rw = d.rect[2], rh = d.rect[3];
    x0 = max(0, min(x0, rw - 1));
    x1 = max(0, min(x1, rw - 1));
    y0 = max(0, min(y0, rh - 1));
    y1 = max(0, min(y1, rh - 1));
    const int ax0 = d.rect[0] + x0, ax1 = d.rect[0] + x1;
    const int ay0 = d.rect[1] + y0, ay1 = d.rect[1] + y1;
    const float w00 = (1.0f - fx) * (1.0f - fy), w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy, w11 = fx * fy;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int ch = d.src_ch[c];
      c3[c] = w00 * px(src, src_step, ax0, ay0, ch) + w10 * px(src, src_step, ax1, ay0, ch) +
              w01 * px(src, src_step, ax0, ay1, ch) + w11 * px(src, src_step, ax1, ay1, ch);
    }
  }
  o[0] = (c3[0] * (1.0f / 255.0f) - d.mean[0]) * d.inv_std[0];
  o[plane] = (c3[1] * (1.0f / 255.0f) - d.mean[1]) * d.inv_std[1];
  o[2 * plane] = (c3[2] * (1.0f / 255.0f) - d.mean[2]) * d.inv_std[2];
}

}  // namespace

void CropBatch::add_rec_line(const hp::core::Box &box, float *dst, int dst_w, int dst_h) {
  const auto ct = hp::core::compute_crop_transform(box, dst_h, dst_w);
  CropDesc d{};
  std::memcpy(d.M, ct.M_inv, sizeof d.M);
  d.dst = dst;
  d.dst_w = dst_w;
  d.dst_h = dst_h;
  d.row_stride = dst_w;
  d.plane_stride = (long long)dst_w * dst_h;
  d.valid_w = ct.crop_width;
  d.valid_h = dst_h;
  for (int c = 0; c < 3; ++c) {  // RGB from BGR, (x/255 - 0.5) / 0.5, zero pad
    d.src_ch[c] = 2 - c;
    d.mean[c] = 0.5f;
    d.inv_std[c] = 2.0f;
    d.pad[c] = 0.f;
  }
  d.mode = kRecWarp;
  h_.push_back(d);
}

void CropBatch::add_table(int x, int y, int w, int h, float *dst, int dst_size) {
  // == table_kernels.cu resize_by_long
  const float s = (float)dst_size / (float)std::max(w, h);
  int nw = (int)lroundf((float)w * s), nh = (int)lroundf((float)h * s);
  nw = std::max(1, std::min(nw, dst_size));
  nh = std::max(1, std::min(nh, dst_size));
  CropDesc d{};
  d.M[0] = (float)w / (float)nw;
  d.M[4] = (float)h / (float)nh;
  d.rect[0] = x;
  d.rect[1] = y;
  d.rect[2] = w;
  d.rect[3] = h;
  d.dst = dst;
  d.dst_w = d.dst_h = dst_size;
  d.row_stride = dst_size;
  d.plane_stride = (long long)dst_size * dst_size;
  d.valid_w = nw;
  d.valid_h = nh;
  constexpr float mean[3] = {0.485f, 0.456f, 0.406f};  // channel order as core/kernels passes them
  constexpr float inv_std[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
  for (int c = 0; c < 3; ++c) {
    d.src_ch[c] = c;  // BGR CHW
    d.mean[c] = mean[c];
    d.inv_std[c] = inv_std[c];
    d.pad[c] = (-mean[c]) * inv_std[c];
  }
  d.mode = kRectResize;
  h_.push_back(d);
}

void CropBatch::add_det_region(int x, int y, int w, int h, float scale, float *canvas, int canvas_w,
                               int canvas_h, int ox, int oy, int &out_w, int &out_h) {
  const int nw = std::max(1, (int)lroundf(w * scale)), nh = std::max(1, (int)lroundf(h * scale));
  CropDesc d{};
  d.M[0] = (float)w / (float)nw;
  d.M[4] = (float)h / (float)nh;
  d.rect[0] = x;
  d.rect[1] = y;
  d.rect[2] = w;
  d.rect[3] = h;
  d.dst = canvas + (size_t)oy * canvas_w + ox;
  d.dst_w = d.valid_w = nw;
  d.dst_h = d.valid_h = nh;
  d.row_stride = canvas_w;
  d.plane_stride = (long long)canvas_w * canvas_h;
  constexpr float mean[3] = {0.485f, 0.456f, 0.406f};  // RGB
  constexpr float sd[3] = {0.229f, 0.224f, 0.225f};
  for (int c = 0; c < 3; ++c) {  // BGR output: channel c is source channel c, ImageNet stats reversed
    d.src_ch[c] = c;
    d.mean[c] = mean[2 - c];
    d.inv_std[c] = 1.0f / sd[2 - c];
  }
  d.mode = kRectResize;
  h_.push_back(d);
  out_w = nw;
  out_h = nh;
}

void CropBatch::launch(const uint8_t *page, size_t step, int page_w, int page_h, cudaStream_t stream) {
  const int n = static_cast<int>(h_.size());
  if (n == 0) return;
  int tiles = 0;
  for (auto &d : h_) {
    d.tile_begin = tiles;
    tiles += ((d.dst_w + kTileW - 1) / kTileW) * ((d.dst_h + kTileH - 1) / kTileH);
  }
  if (static_cast<size_t>(n) > cap_) {
    cap_ = std::max<size_t>(n, 2 * cap_);
    h_pinned_.reset(cap_);
    d_.reset(cap_);
  }
  // Caller must not reuse this CropBatch before the stream passes this copy.
  std::memcpy(h_pinned_.get(), h_.data(), n * sizeof(CropDesc));
  CUDA_CHECK(cudaMemcpyAsync(d_.get(), h_pinned_.get(), n * sizeof(CropDesc), cudaMemcpyHostToDevice, stream));
  crop_batch_kernel<<<tiles, dim3(kTileW, kTileH), 0, stream>>>(page, static_cast<int>(step), page_w, page_h,
                                                              d_.get(), n);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hp
