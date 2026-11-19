#include "multi_resize.h"

#include <algorithm>
#include <cmath>

#include "core/common/cuda/cuda_check.h"

namespace hp {
namespace {

__device__ __forceinline__ float bilinear(const uint8_t *__restrict__ src, size_t step, int sw,
                                          int sh, int c, float fx, float fy) {
  fx = fminf(fmaxf(fx, 0.f), (float)(sw - 1));
  fy = fminf(fmaxf(fy, 0.f), (float)(sh - 1));
  const int x0 = (int)fx, y0 = (int)fy;
  const int x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
  const float ax = fx - x0, ay = fy - y0;
  const uint8_t *r0 = src + (size_t)y0 * step;
  const uint8_t *r1 = src + (size_t)y1 * step;
  const float top = r0[x0 * 3 + c] * (1.f - ax) + r0[x1 * 3 + c] * ax;
  const float bot = r1[x0 * 3 + c] * (1.f - ax) + r1[x1 * 3 + c] * ax;
  return top * (1.f - ay) + bot * ay;
}

__global__ void __launch_bounds__(256)
multi_resize_kernel(const uint8_t *__restrict__ src, size_t step, int sw, int sh,
                    const __grid_constant__ ResizeTable table) {
  const ResizeOut &o = table.o[blockIdx.z];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= o.out_w || y >= o.out_h) return;
  const int px = x - o.x0, py = y - o.y0;
  const bool inside = px >= 0 && py >= 0 && px < o.rw && py < o.rh;
  const float sx = (px + 0.5f) * ((float)sw / o.rw) - 0.5f;
  const float sy = (py + 0.5f) * ((float)sh / o.rh) - 0.5f;
  const size_t plane = (size_t)o.out_w * o.out_h;
  const size_t idx = (size_t)y * o.out_w + x;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float v = inside ? bilinear(src, step, sw, sh, o.src_ch[c], sx, sy) : o.fill;
    o.dst[c * plane + idx] = (v * (1.f / 255.f) - o.mean[c]) * o.inv_std[c];
  }
}

constexpr float kImMean[3] = {0.485f, 0.456f, 0.406f};  // RGB
constexpr float kImStd[3] = {0.229f, 0.224f, 0.225f};

}  // namespace

void launch_multi_resize(const uint8_t *src, size_t step, int sw, int sh,
                         const ResizeTable &table, cudaStream_t stream) {
  if (table.n == 0) return;
  int mw = 0, mh = 0;
  for (int i = 0; i < table.n; ++i) {
    mw = std::max(mw, table.o[i].out_w);
    mh = std::max(mh, table.o[i].out_h);
  }
  const dim3 block(32, 8);
  const dim3 grid((mw + 31) / 32, (mh + 7) / 8, table.n);
  multi_resize_kernel<<<grid, block, 0, stream>>>(src, step, sw, sh, table);
  CUDA_CHECK(cudaGetLastError());
}

ResizeOut dfine_spec(int w, int h, int size, float *dst, Letterbox &lb) {
  ResizeOut o;
  const float r = static_cast<float>(size) / std::max(w, h);
  o.dst = dst;
  o.out_w = o.out_h = size;
  o.rw = static_cast<int>(w * r);
  o.rh = static_cast<int>(h * r);
  o.x0 = (size - o.rw) / 2;
  o.y0 = (size - o.rh) / 2;
  o.fill = 114.f;
  // RGB from BGR source; mean 0 / std 1 already the defaults.
  lb = {r, o.x0, o.y0};
  return o;
}

ResizeOut det_spec(int w, int h, int max_side, float *dst) {
  ResizeOut o;
  const double s = std::min(1.0, static_cast<double>(max_side) / std::max(w, h));
  auto r32 = [&](int v) { return std::max(32, static_cast<int>(std::lround(v * s / 32.0)) * 32); };
  o.dst = dst;
  o.out_w = o.rw = r32(w);
  o.out_h = o.rh = r32(h);
  for (int c = 0; c < 3; ++c) {  // BGR output: channel c is source channel c
    o.src_ch[c] = c;
    o.mean[c] = kImMean[2 - c];
    o.inv_std[c] = 1.f / kImStd[2 - c];
  }
  return o;
}

ResizeOut doc_ori_spec(int w, int h, float *dst) {
  ResizeOut o;
  const double k = 256.0 / std::min(w, h);
  o.dst = dst;
  o.out_w = o.out_h = 224;
  o.rw = static_cast<int>(std::lround(w * k));
  o.rh = static_cast<int>(std::lround(h * k));
  o.x0 = -((o.rw - 224) / 2);
  o.y0 = -((o.rh - 224) / 2);
  for (int c = 0; c < 3; ++c) {
    o.mean[c] = kImMean[c];
    o.inv_std[c] = 1.f / kImStd[c];
  }
  return o;
}

}  // namespace hp
