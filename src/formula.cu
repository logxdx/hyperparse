#include "formula.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>

#include "core/common/cuda/cuda_check.h"

namespace hp {
namespace {

// ── constants (PP-FormulaNet-S) ──────────────────────────────────────────────
constexpr int H = 16, Dh = 24, CTX = 144, VOCAB = 50000, MAXLEN = 1056, MAXIT = 342, CHECK = 16;
constexpr int MEMD = 2048;
constexpr float kMean = 0.7931f, kStd = 0.1738f, kPad = (0.0f - kMean) / kStd;

// ── preprocess: OpenCV-free port of formula_preprocess_one ──────────────────
constexpr int PREC = 22;  // PIL PRECISION_BITS
constexpr int64_t HALF = (int64_t)1 << (PREC - 1);

double pil_cubic(double x) {
  const double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

struct Axis {
  int ksize = 0;
  std::vector<int> lo, n;
  std::vector<int64_t> w;
};

void pil_axis(int out_len, int in_len, Axis &f) {  // PIL precompute_coeffs + normalize_coeffs_8bpc
  const double scale = (double)in_len / out_len, fscale = std::max(1.0, scale), support = 2.0 * fscale;
  f.ksize = (int)std::ceil(support) * 2 + 1;
  f.lo.assign(out_len, 0);
  f.n.assign(out_len, 0);
  f.w.assign((size_t)out_len * f.ksize, 0);
  std::vector<double> tmp(f.ksize);
  for (int xx = 0; xx < out_len; ++xx) {
    const double center = (xx + 0.5) * scale;
    int lo = (int)(center - support + 0.5), hi = (int)(center + support + 0.5);
    lo = std::max(lo, 0);
    hi = std::min(hi, in_len);
    const int n = hi - lo;
    double sum = 0;
    for (int k = 0; k < n; ++k) sum += tmp[k] = pil_cubic((lo + k - center + 0.5) / fscale);
    if (sum == 0.0) sum = 1.0;
    for (int k = 0; k < n; ++k) {
      const double v = tmp[k] / sum;
      f.w[(size_t)xx * f.ksize + k] = (int64_t)(v >= 0 ? v * (1 << PREC) + 0.5 : v * (1 << PREC) - 0.5);
    }
    f.lo[xx] = lo;
    f.n[xx] = n;
  }
}

inline uint8_t clip8(int64_t s) { s >>= PREC; return (uint8_t)(s < 0 ? 0 : s > 255 ? 255 : s); }

// PIL bicubic, horizontal then vertical, uint8 intermediate (order is load-bearing).
void pil_resize(const uint8_t *src, size_t step, int sw, int sh, int nw, int nh, std::vector<uint8_t> &out) {
  Axis hx, vy;
  pil_axis(nw, sw, hx);
  std::vector<uint8_t> tmp((size_t)sh * nw * 3);
  for (int y = 0; y < sh; ++y) {
    const uint8_t *row = src + (size_t)y * step;
    uint8_t *o = tmp.data() + (size_t)y * nw * 3;
    for (int xx = 0; xx < nw; ++xx) {
      const int64_t *w = &hx.w[(size_t)xx * hx.ksize];
      const uint8_t *p = row + (size_t)hx.lo[xx] * 3;
      int64_t b = HALF, g = HALF, r = HALF;
      for (int k = 0; k < hx.n[xx]; ++k, p += 3) { b += w[k] * p[0]; g += w[k] * p[1]; r += w[k] * p[2]; }
      o[xx * 3] = clip8(b); o[xx * 3 + 1] = clip8(g); o[xx * 3 + 2] = clip8(r);
    }
  }
  pil_axis(nh, sh, vy);
  out.assign((size_t)nh * nw * 3, 0);
  for (int yy = 0; yy < nh; ++yy) {
    const int64_t *w = &vy.w[(size_t)yy * vy.ksize];
    uint8_t *o = out.data() + (size_t)yy * nw * 3;
    for (int x = 0; x < nw * 3; x += 3) {
      int64_t b = HALF, g = HALF, r = HALF;
      for (int k = 0; k < vy.n[yy]; ++k) {
        const uint8_t *p = tmp.data() + ((size_t)(vy.lo[yy] + k) * nw * 3) + x;
        b += w[k] * p[0]; g += w[k] * p[1]; r += w[k] * p[2];
      }
      o[x] = clip8(b); o[x + 1] = clip8(g); o[x + 2] = clip8(r);
    }
  }
}

// ── decode kernels ──────────────────────────────────────────────────────────
__global__ void argmax_rows(const float *__restrict__ lg, int64_t *__restrict__ next, int64_t *__restrict__ hist,
                            int64_t *__restrict__ tok, int V) {
  const int row = blockIdx.x;
  const float *p = lg + (size_t)row * V;
  float m = -1e30f;
  int mi = 0;
  for (int i = threadIdx.x; i < V; i += blockDim.x)
    if (p[i] > m) { m = p[i]; mi = i; }
  for (int s = 16; s > 0; s >>= 1) {  // ties -> lower index
    const float ov = __shfl_down_sync(0xffffffffu, m, s);
    const int oi = __shfl_down_sync(0xffffffffu, mi, s);
    if (ov > m || (ov == m && oi < mi)) { m = ov; mi = oi; }
  }
  __shared__ float sm[32];
  __shared__ int si[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) { sm[warp] = m; si[warp] = mi; }
  __syncthreads();
  if (warp == 0) {
    const int nw = blockDim.x >> 5;
    m = lane < nw ? sm[lane] : -1e30f;
    mi = lane < nw ? si[lane] : 0;
    for (int s = 16; s > 0; s >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, m, s);
      const int oi = __shfl_down_sync(0xffffffffu, mi, s);
      if (ov > m || (ov == m && oi < mi)) { m = ov; mi = oi; }
    }
    if (lane == 0) { next[row] = mi; hist[row] = mi; tok[row] = mi; }
  }
}

__global__ void scan_eos(const int64_t *all, unsigned char *done, int B, int64_t eos, int s0, int s1) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B || done[b]) return;
  for (int s = s0; s < s1; ++s)
    for (int j = 0; j < 3; ++j)
      if (all[((size_t)s * B + b) * 3 + j] == eos) { done[b] = 1; return; }
}

nvinfer1::Dims dims(std::initializer_list<int64_t> v) {
  nvinfer1::Dims d{};
  d.nbDims = (int)v.size();
  int i = 0;
  for (auto x : v) d.d[i++] = x;
  return d;
}

}  // namespace

void formula_preprocess(const uint8_t *bgr, size_t step, int w, int h, float *out) {
  const int S = kFormulaS;
  if (w <= 0 || h <= 0) { std::fill(out, out + S * S, kPad); return; }
  // cv::cvtColor BGR2GRAY on u8: fixed point (R 4899, G 9617, B 1868) >> 14, rounded.
  std::vector<uint8_t> gray((size_t)w * h);
  int mn = 255, mx = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const uint8_t *p = bgr + (size_t)y * step + x * 3;
      const int g = (p[0] * 1868 + p[1] * 9617 + p[2] * 4899 + (1 << 13)) >> 14;
      gray[(size_t)y * w + x] = (uint8_t)g;
      mn = std::min(mn, g);
      mx = std::max(mx, g);
    }
  int cx0 = 0, cy0 = 0, cx1 = w, cy1 = h;  // ink trim: (g-mn)/max(1,mx-mn)*255 < 200
  if (mx > mn) {
    const double a = 255.0 / std::max(1, mx - mn), b = -mn * a;
    int x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if ((float)(gray[(size_t)y * w + x] * a + b) < 200.0f) {
          x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
        }
    if (x1 >= x0 && y1 >= y0) { cx0 = x0; cy0 = y0; cx1 = x1 + 1; cy1 = y1 + 1; }
  }
  const int W = cx1 - cx0, Hh = cy1 - cy0;
  int nw, nh;  // Python round (ties to even) == lrint
  if (W <= Hh) { nw = S; nh = std::max(1, (int)std::lrint((double)S * Hh / W)); }
  else { nh = S; nw = std::max(1, (int)std::lrint((double)S * W / Hh)); }
  if (nh > S) { nw = std::max(1, (int)std::lrint((double)nw * S / nh)); nh = S; }
  if (nw > S) { nh = std::max(1, (int)std::lrint((double)nh * S / nw)); nw = S; }
  std::vector<uint8_t> rs;
  pil_resize(bgr + (size_t)cy0 * step + cx0 * 3, step, W, Hh, nw, nh, rs);
  std::fill(out, out + S * S, kPad);
  const int top = (S - nh) / 2, left = (S - nw) / 2;
  const double inv255 = 1.0 / 255.0, inv_std = 1.0 / kStd, off = -kMean / (double)kStd;
  for (int y = 0; y < nh; ++y)
    for (int x = 0; x < nw; ++x) {
      const uint8_t *p = &rs[((size_t)y * nw + x) * 3];
      float c[3];
      for (int k = 0; k < 3; ++k) c[k] = (float)((float)(p[k] * inv255) * inv_std + off);
      out[(size_t)(top + y) * S + left + x] = c[0] * 0.114f + c[1] * 0.587f + c[2] * 0.299f;  // float BGR2GRAY
    }
}

bool FormulaRecognizer::load(const std::string &dir) {
  // Encoder + prep stay FP32: FP16 there flips tokens on ~7% of crops (measured,
  // papers60: 90/97 identical) for ~3% speed. FP16 decode step: 93/97 identical,
  // mean token edit 0.18%, 36.5 -> 25.2 ms per formula on the RTX 4050 (FP32 TRT
  // is token-identical to ONNX Runtime on 20/20 formulas; not shipped).
  enc_ = std::make_unique<hp::core::engine::TrtEngine>(dir + "/encoder.plan");
  prep_ = std::make_unique<hp::core::engine::TrtEngine>(dir + "/prep.plan");
  step_ = std::make_unique<hp::core::engine::TrtEngine>(dir + "/step_fp16.plan");
  if (!enc_->load() || !prep_->load() || !step_->load()) return false;
  tok_ = hp::core::formula::FormulaTokenizer::load(dir + "/tokenizer.json");
  if (!tok_) return false;
  enc_out_ = enc_->output_names().at(0);
  const int B = kMaxBatch;
  const size_t kv = (size_t)2 * B * H * MAXLEN * Dh, ckv = (size_t)2 * B * H * CTX * Dh;
  h_x_ = hp::core::CudaHostPtr<float>((size_t)kFormulaS * kFormulaS);
  d_x_ = hp::core::CudaPtr<float>((size_t)kFormulaS * kFormulaS);
  d_mem_ = hp::core::CudaPtr<float>((size_t)B * CTX * MEMD);
  d_ck_ = hp::core::CudaPtr<float>(ckv);
  d_cv_ = hp::core::CudaPtr<float>(ckv);
  d_kA_ = hp::core::CudaPtr<float>(kv);
  d_kB_ = hp::core::CudaPtr<float>(kv);
  d_vA_ = hp::core::CudaPtr<float>(kv);
  d_vB_ = hp::core::CudaPtr<float>(kv);
  d_logits_ = hp::core::CudaPtr<float>((size_t)B * 3 * VOCAB);
  d_tok_ = hp::core::CudaPtr<int64_t>((size_t)B * 3);
  d_all_ = hp::core::CudaPtr<int64_t>((size_t)MAXIT * B * 3);
  d_done_ = hp::core::CudaPtr<unsigned char>(B);
  std::vector<int64_t> pos(MAXIT);
  for (int i = 0; i < MAXIT; ++i) pos[i] = 3LL * i;
  d_pos_ = hp::core::CudaPtr<int64_t>(MAXIT);
  CUDA_CHECK(cudaMemcpy(d_pos_.get(), pos.data(), MAXIT * sizeof(int64_t), cudaMemcpyHostToDevice));
  return true;
}

void FormulaRecognizer::decode_chunk(int B, std::vector<std::vector<int64_t>> &out, cudaStream_t s) {
  const int64_t eos = tok_->eos_id();
  const size_t kvb = (size_t)2 * B * H * MAXLEN * Dh * sizeof(float);
  CUDA_CHECK(cudaMemsetAsync(d_kA_.get(), 0, kvb, s));
  CUDA_CHECK(cudaMemsetAsync(d_vA_.get(), 0, kvb, s));
  CUDA_CHECK(cudaMemsetAsync(d_tok_.get(), 0, (size_t)B * 3 * sizeof(int64_t), s));  // <s> = 0
  CUDA_CHECK(cudaMemsetAsync(d_done_.get(), 0, B, s));
  float *kin = d_kA_.get(), *kout = d_kB_.get(), *vin = d_vA_.get(), *vout = d_vB_.get();
  if (!step_->set_input_shape("tokens", dims({B, 3})) || !step_->set_input_shape("pos", dims({1})) ||
      !step_->set_input_shape("kb", dims({2, B, H, MAXLEN, Dh})) || !step_->set_input_shape("vb", dims({2, B, H, MAXLEN, Dh})) ||
      !step_->set_input_shape("ck", dims({2, B, H, CTX, Dh})) || !step_->set_input_shape("cv", dims({2, B, H, CTX, Dh})))
    throw std::runtime_error("formula step: bad shape");
  step_->set_tensor_address("tokens", d_tok_.get());
  step_->set_tensor_address("ck", d_ck_.get());
  step_->set_tensor_address("cv", d_cv_.get());
  step_->set_tensor_address("logits", d_logits_.get());
  std::vector<unsigned char> done(B, 0);
  int last = MAXIT, checked = 0;
  for (int it = 0; it < MAXIT; ++it) {
    step_->set_tensor_address("pos", d_pos_.get() + it);
    step_->set_tensor_address("kb", kin);
    step_->set_tensor_address("vb", vin);
    step_->set_tensor_address("kb_out", kout);
    step_->set_tensor_address("vb_out", vout);
    if (!step_->execute(s)) throw std::runtime_error("formula step failed");
    int64_t *hist = d_all_.get() + (size_t)it * B * 3;
    argmax_rows<<<B * 3, 256, 0, s>>>(d_logits_.get(), hist, hist, d_tok_.get(), VOCAB);
    std::swap(kin, kout);
    std::swap(vin, vout);
    ++steps_run;
    if ((it + 1) % CHECK == 0) {
      scan_eos<<<(B + 63) / 64, 64, 0, s>>>(d_all_.get(), d_done_.get(), B, eos, checked, it + 1);
      CUDA_CHECK(cudaMemcpyAsync(done.data(), d_done_.get(), B, cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
      checked = it + 1;
      if (std::all_of(done.begin(), done.end(), [](unsigned char d) { return d != 0; })) { last = it + 1; break; }
    }
  }
  std::vector<int64_t> hall((size_t)last * B * 3);
  CUDA_CHECK(cudaMemcpyAsync(hall.data(), d_all_.get(), hall.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  out.assign(B, {});
  for (int b = 0; b < B; ++b) {
    bool stop = false;
    for (int st = 0; st < last && !stop; ++st)
      for (int j = 0; j < 3; ++j) {
        const int64_t t = hall[((size_t)st * B + b) * 3 + j];
        if (t == eos) { stop = true; break; }
        if (t != 0 && t != 1) out[b].push_back(t);
      }
  }
}

std::vector<std::string> FormulaRecognizer::run(const std::vector<FormulaCrop> &crops, cudaStream_t s) {
  std::vector<std::string> res(crops.size());
  for (size_t c0 = 0; c0 < crops.size(); c0 += kMaxBatch) {
    const int B = (int)std::min<size_t>(kMaxBatch, crops.size() - c0);
    // Encoder one crop at a time: a batched conv drifts near-tie tokens.
    if (!enc_->set_input_shape("x", dims({1, 1, kFormulaS, kFormulaS}))) throw std::runtime_error("formula enc shape");
    enc_->set_tensor_address("x", d_x_.get());
    for (int b = 0; b < B; ++b) {
      const auto &c = crops[c0 + b];
      formula_preprocess(c.bgr, c.step, c.w, c.h, h_x_.get());
      if (!dump_dir.empty())
        std::ofstream(dump_dir + "/x_" + std::to_string(dump_n + b) + ".f32", std::ios::binary)
            .write(reinterpret_cast<const char *>(h_x_.get()), sizeof(float) * kFormulaS * kFormulaS);
      CUDA_CHECK(cudaMemcpyAsync(d_x_.get(), h_x_.get(), sizeof(float) * kFormulaS * kFormulaS, cudaMemcpyHostToDevice, s));
      enc_->set_tensor_address(enc_out_, d_mem_.get() + (size_t)b * CTX * MEMD);
      if (!enc_->execute(s)) throw std::runtime_error("formula encoder failed");
      CUDA_CHECK(cudaStreamSynchronize(s));  // h_x_ reused next crop
    }
    if (!prep_->set_input_shape("memory", dims({B, CTX, MEMD}))) throw std::runtime_error("formula prep shape");
    prep_->set_tensor_address("memory", d_mem_.get());
    prep_->set_tensor_address("ck", d_ck_.get());
    prep_->set_tensor_address("cv", d_cv_.get());
    if (!prep_->execute(s)) throw std::runtime_error("formula prep failed");
    std::vector<std::vector<int64_t>> toks;
    decode_chunk(B, toks, s);
    for (int b = 0; b < B; ++b) {
      res[c0 + b] = tok_->decode(toks[b], /*post_process=*/false);
      if (!dump_dir.empty()) {
        std::ofstream t(dump_dir + "/tok_" + std::to_string(dump_n + b) + ".txt");
        for (auto id : toks[b]) t << id << ' ';
      }
    }
    dump_n += B;
  }
  return res;
}

}  // namespace hp
