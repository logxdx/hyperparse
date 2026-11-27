#pragma once
// One kernel launch: a u8 BGR page on the GPU -> every per-page model input.
// blockIdx.z selects the output; bilinear = OpenCV INTER_LINEAR (half-pixel
// centres, edge clamp). Validated against OpenCV in trt-env/fused_preprocess.py.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace hp {

struct ResizeOut {
  float *dst = nullptr;  // CHW float32
  int out_w = 0, out_h = 0;
  int x0 = 0, y0 = 0;    // where the resized page sits inside the tensor (pad > 0, crop < 0)
  int rw = 0, rh = 0;    // resized page size before pad/crop
  float fill = 0.f;      // value outside the page, pre-normalisation (0..255)
  int src_ch[3] = {2, 1, 0};  // source channel feeding output channel c (source is BGR)
  float mean[3] = {0, 0, 0};  // after /255, output channel order
  float inv_std[3] = {1, 1, 1};
};

inline constexpr int kMaxResizeOuts = 4;

struct ResizeTable {
  ResizeOut o[kMaxResizeOuts];
  int n = 0;
  void add(const ResizeOut &r) { o[n++] = r; }
};

// src: packed u8 BGR rows with `step` bytes per row.
void launch_multi_resize(const uint8_t *src, size_t step, int sw, int sh,
                         const ResizeTable &table, cudaStream_t stream);

// u8 BGR page -> u8 grey (w*h, packed), the same integer weights as
// tn::grey_from_bgr (bit-identical).
void launch_bgr_to_grey(const uint8_t *src, size_t step, int w, int h, uint8_t *dst, cudaStream_t stream);

// ── Specs (the geometry each model was evaluated with) ──────────────────────

// D-FINE: ratio = S/max(w,h), int() truncation, centred pad, fill 114, RGB, /255.
struct Letterbox {
  float scale = 1.f;
  int x0 = 0, y0 = 0;
};
ResizeOut dfine_spec(int w, int h, int size, float *dst, Letterbox &lb);

// PP-OCR det: keep ratio, long side <= max_side, each side round()ed to x32
// (min 32), BGR, ImageNet mean/std.
ResizeOut det_spec(int w, int h, int max_side, float *dst);

// doc_ori: short side 256 (round), centre crop 224, RGB, ImageNet mean/std.
ResizeOut doc_ori_spec(int w, int h, float *dst);

}  // namespace hp
