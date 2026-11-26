#include "orient.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "core/common/cuda/cuda_check.h"

namespace hp {
namespace {

constexpr int kSide = 224;
constexpr int kShort = 256;
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};  // R, G, B
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
constexpr int kLabels[4] = {0, 90, 180, 270};

nvinfer1::Dims dims4(int64_t a, int64_t b, int64_t c, int64_t d) {
  nvinfer1::Dims x{};
  x.nbDims = 4;
  x.d[0] = a, x.d[1] = b, x.d[2] = c, x.d[3] = d;
  return x;
}

}  // namespace

bool DocOrient::load(const std::string &plan_path) {
  engine_ = std::make_unique<hp::core::engine::TrtEngine>(plan_path);
  if (!engine_->load()) return false;
  d_in_.reset(3 * kSide * kSide);
  d_out_.reset(4);
  h_in_.resize(3 * kSide * kSide);
  if (!engine_->set_input_shape("x", dims4(1, 3, kSide, kSide))) return false;
  engine_->set_tensor_address("x", d_in_.get());
  engine_->set_tensor_address("fetch_name_0", d_out_.get());
  return true;
}

OrientResult DocOrient::classify(const uint8_t *bgr, int w, int h, cudaStream_t stream) {
  // Resize so the short side is 256 (cv2.resize to (round(w*s), round(h*s)), INTER_LINEAR:
  // src = (dst + 0.5) * src_len / dst_len - 0.5, clamped), then take the centred 224 x 224.
  const double s = static_cast<double>(kShort) / std::min(w, h);
  const int rw = std::max(1, static_cast<int>(std::lround(w * s))), rh = std::max(1, static_cast<int>(std::lround(h * s)));
  const int ox = std::max(0, (rw - kSide) / 2), oy = std::max(0, (rh - kSide) / 2);
  const double fx = static_cast<double>(w) / rw, fy = static_cast<double>(h) / rh;
  const size_t plane = static_cast<size_t>(kSide) * kSide;
  std::fill(h_in_.begin(), h_in_.end(), 0.f);
  for (int y = 0; y < std::min(kSide, rh); ++y) {
    const double sy = std::clamp((y + oy + 0.5) * fy - 0.5, 0.0, static_cast<double>(h - 1));
    const int y0 = static_cast<int>(sy), y1 = std::min(y0 + 1, h - 1);
    const float wy = static_cast<float>(sy - y0);
    for (int x = 0; x < std::min(kSide, rw); ++x) {
      const double sx = std::clamp((x + ox + 0.5) * fx - 0.5, 0.0, static_cast<double>(w - 1));
      const int x0 = static_cast<int>(sx), x1 = std::min(x0 + 1, w - 1);
      const float wx = static_cast<float>(sx - x0);
      const uint8_t *p00 = bgr + (static_cast<size_t>(y0) * w + x0) * 3, *p01 = bgr + (static_cast<size_t>(y0) * w + x1) * 3;
      const uint8_t *p10 = bgr + (static_cast<size_t>(y1) * w + x0) * 3, *p11 = bgr + (static_cast<size_t>(y1) * w + x1) * 3;
      for (int c = 0; c < 3; ++c) {  // c: RGB channel; BGR source index 2 - c
        const int k = 2 - c;
        const float v = (1 - wy) * ((1 - wx) * p00[k] + wx * p01[k]) + wy * ((1 - wx) * p10[k] + wx * p11[k]);
        h_in_[c * plane + static_cast<size_t>(y) * kSide + x] = (v / 255.f - kMean[c]) / kStd[c];
      }
    }
  }
  CUDA_CHECK(cudaMemcpyAsync(d_in_.get(), h_in_.data(), h_in_.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  if (!engine_->execute(stream)) throw std::runtime_error("DocOrient: execute failed");
  float p[4];
  CUDA_CHECK(cudaMemcpyAsync(p, d_out_.get(), sizeof p, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const int k = static_cast<int>(std::max_element(p, p + 4) - p);
  return {kLabels[k], p[k]};
}

void rotate_upright_bgr(uint8_t *bgr, int &w, int &h, int degrees_cw) {
  if (degrees_cw != 90 && degrees_cw != 180 && degrees_cw != 270) return;
  const int W0 = w, H0 = h;
  const int W1 = degrees_cw == 180 ? W0 : H0, H1 = degrees_cw == 180 ? H0 : W0;
  std::vector<uint8_t> out(static_cast<size_t>(W0) * H0 * 3);
  for (int y = 0; y < H1; ++y)
    for (int x = 0; x < W1; ++x) {
      int sx, sy;  // source pixel in the original image
      if (degrees_cw == 90) sx = W0 - 1 - y, sy = x;            // turn back counter-clockwise
      else if (degrees_cw == 270) sx = y, sy = H0 - 1 - x;      // turn back clockwise
      else sx = W0 - 1 - x, sy = H0 - 1 - y;
      std::memcpy(&out[(static_cast<size_t>(y) * W1 + x) * 3], bgr + (static_cast<size_t>(sy) * W0 + sx) * 3, 3);
    }
  std::memcpy(bgr, out.data(), out.size());
  w = W1, h = H1;
}

void box_to_original(float &x0, float &y0, float &x1, float &y1, int degrees_cw, float w0, float h0) {
  const float a0 = x0, b0 = y0, a1 = x1, b1 = y1;
  if (degrees_cw == 90) x0 = w0 - b1, x1 = w0 - b0, y0 = a0, y1 = a1;
  else if (degrees_cw == 270) x0 = b0, x1 = b1, y0 = h0 - a1, y1 = h0 - a0;
  else if (degrees_cw == 180) x0 = w0 - a1, x1 = w0 - a0, y0 = h0 - b1, y1 = h0 - b0;
}

}  // namespace hp
