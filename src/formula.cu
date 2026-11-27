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
constexpr int H = 16, Dh = 24, CTX = 144, VOCAB = 50000, MAXLEN = 1056, MAXIT = 342;
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
// Cohort row decoded in each slot of a step batch (-1: empty).
struct Rows {
  int r[FormulaRecognizer::kBatch];
};

// Greedy pick per (slot, token): the next input token of the slot and, for a live slot, its
// cohort row's history; an EOS marks the row done in `done` (pinned host memory).
__global__ void argmax_rows(const float *__restrict__ lg, int64_t *__restrict__ tok, int64_t *__restrict__ hist,
                            Rows rows, int it, int64_t eos, int *done, int V) {
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
    if (lane == 0) {
      tok[row] = mi;
      const int r = rows.r[row / 3];
      if (r >= 0) {
        hist[((size_t)r * MAXIT + it) * 3 + row % 3] = mi;
        if (mi == eos) *(volatile int *)(done + r) = 1;
      }
    }
  }
}

nvinfer1::Dims dims(std::initializer_list<int64_t> v) {
  nvinfer1::Dims d{};
  d.nbDims = (int)v.size();
  int i = 0;
  for (auto x : v) d.d[i++] = x;
  return d;
}

constexpr int B = FormulaRecognizer::kBatch, G = FormulaRecognizer::kGroups;
constexpr size_t kX = (size_t)kFormulaS * kFormulaS, kMem = (size_t)CTX * MEMD;
constexpr size_t kKvRow = (size_t)H * MAXLEN * Dh, kKv = 2 * B * kKvRow;    // [2, B, H, MAXLEN, Dh]
constexpr size_t kCkvRow = (size_t)H * CTX * Dh, kCkv = 2 * B * kCkvRow;   // [2, B, H, CTX, Dh]

// Dark pixels of a crop (as the ink trim counts them): the cohort sort key, ~ formula length.
int ink_pixels(const FormulaCrop &c) {
  int mn = 255, mx = 0;
  auto gray = [&](int x, int y) {
    const uint8_t *p = c.bgr + (size_t)y * c.step + x * 3;
    return (p[0] * 1868 + p[1] * 9617 + p[2] * 4899 + (1 << 13)) >> 14;
  };
  for (int y = 0; y < c.h; ++y)
    for (int x = 0; x < c.w; ++x) {
      const int g = gray(x, y);
      mn = std::min(mn, g);
      mx = std::max(mx, g);
    }
  if (mx <= mn) return 0;
  const int thr = mn + (200 * (mx - mn) + 254) / 255;  // (g - mn) * 255 / (mx - mn) < 200
  int n = 0;
  for (int y = 0; y < c.h; ++y)
    for (int x = 0; x < c.w; ++x) n += gray(x, y) < thr;
  return n;
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
  h_x_ = hp::core::CudaHostPtr<float>(2 * B * kX);
  h_eos_ = hp::core::CudaHostPtr<int>(kCohort);
  d_x_ = hp::core::CudaPtr<float>(B * kX);
  d_mem_ = hp::core::CudaPtr<float>(B * kMem);
  d_ckv_ = hp::core::CudaPtr<float>(2 * G * 2 * kCkv);  // [set][group][ck, cv]
  d_kv_ = hp::core::CudaPtr<float>((G + 1) * 2 * kKv);  // [buffer][k, v]: a group's KV + one to step into
  d_logits_ = hp::core::CudaPtr<float>((size_t)B * 3 * VOCAB);
  d_tok_ = hp::core::CudaPtr<int64_t>((size_t)G * B * 3);
  d_hist_ = hp::core::CudaPtr<int64_t>((size_t)kCohort * MAXIT * 3);
  std::vector<int64_t> pos(MAXIT);
  for (int i = 0; i < MAXIT; ++i) pos[i] = 3LL * i;
  d_pos_ = hp::core::CudaPtr<int64_t>(MAXIT);
  CUDA_CHECK(cudaMemcpy(d_pos_.get(), pos.data(), MAXIT * sizeof(int64_t), cudaMemcpyHostToDevice));
  return enc_->set_input_shape("x", dims({1, 1, kFormulaS, kFormulaS})) && step_->set_input_shape("pos", dims({1})) &&
         set_prep_batch(B) && set_step_batch(B, MAXLEN) && init_len_buckets();
}

// A plan that takes the KV at a shorter length (engine_build.cpp: padded to 1056 slots inside the
// graph, so the attention and its tokens are unchanged) only moves the slots in use across the
// graph boundary: the step at position p passes the smallest bucket >= p + 3.
bool FormulaRecognizer::init_len_buckets() {
  len_buckets_.clear();
  const char *e = std::getenv("HP_FORMULA_FULL_KV");  // debug: always all 1056 slots
  if (!(e && *e == '1') && step_->set_input_shape("kb", dims({2, B, H, 64, Dh})))
    len_buckets_ = {64, 128, 256, 512};
  len_buckets_.push_back(MAXLEN);
  step_len_ = 0;
  return set_step_batch(B, MAXLEN);
}

// Prep / step at batch nb: kBatch, or 1 for a crop decoded alone (see formula.h). The next
// cohort's prep runs during this cohort's decode, so they are shaped separately.
bool FormulaRecognizer::set_prep_batch(int nb) {
  if (nb == prep_batch_) return true;
  prep_batch_ = nb;
  return prep_->set_input_shape("memory", dims({nb, CTX, MEMD}));
}

bool FormulaRecognizer::set_step_batch(int nb, int len) {
  if (nb == step_batch_ && len == step_len_) return true;
  step_batch_ = nb;
  step_len_ = len;
  return step_->set_input_shape("tokens", dims({nb, 3})) && step_->set_input_shape("kb", dims({2, nb, H, len, Dh})) &&
         step_->set_input_shape("vb", dims({2, nb, H, len, Dh})) &&
         step_->set_input_shape("ck", dims({2, nb, H, CTX, Dh})) && step_->set_input_shape("cv", dims({2, nb, H, CTX, Dh}));
}

void FormulaRecognizer::encode_step(const std::vector<FormulaCrop> &crops, const std::vector<int> &order,
                                    EncodeJob &job) {
  const int i = job.next, g = i / B, b = i % B;
  float *ck = d_ckv_.get() + (size_t)(job.set * G + g) * 2 * kCkv;
  if (b == 0) {  // a new step batch: the other pinned half, once its uploads are done
    x_half_ ^= 1;
    CUDA_CHECK(cudaEventSynchronize(x_free_[x_half_]));
    if (i == 0) CUDA_CHECK(cudaStreamWaitEvent(enc_stream_, set_free_[job.set]));  // the set's last decode is over
  }
  const int ci = order[job.c0 + i];
  const FormulaCrop &c = crops[ci];
  float *x = h_x_.get() + (x_half_ * B + b) * kX;
  formula_preprocess(c.bgr, c.step, c.w, c.h, x);
  if (!dump_dir.empty()) {
    const std::string id = std::to_string(dump_n + ci);
    std::ofstream(dump_dir + "/x_" + id + ".f32", std::ios::binary)
        .write(reinterpret_cast<const char *>(x), sizeof(float) * kX);
    std::ofstream cf(dump_dir + "/c_" + id + ".bgr", std::ios::binary);
    const int32_t wh[2] = {c.w, c.h};
    cf.write(reinterpret_cast<const char *>(wh), sizeof wh);
    for (int y = 0; y < c.h; ++y) cf.write(reinterpret_cast<const char *>(c.bgr + (size_t)y * c.step), (size_t)c.w * 3);
  }
  CUDA_CHECK(cudaMemcpyAsync(d_x_.get() + b * kX, x, sizeof(float) * kX, cudaMemcpyHostToDevice, enc_stream_));
  enc_->set_tensor_address("x", d_x_.get() + b * kX);
  enc_->set_tensor_address(enc_out_, d_mem_.get() + b * kMem);
  if (!enc_->execute(enc_stream_)) throw std::runtime_error("formula encoder failed");
  ++job.next;
  if (b == B - 1 || job.done()) {  // the step batch is complete: cross-KV prep
    CUDA_CHECK(cudaEventRecord(x_free_[x_half_], enc_stream_));
    if (!set_prep_batch(job.n == 1 ? 1 : B)) throw std::runtime_error("formula prep shape");
    for (int r = job.n == 1 ? B : b + 1; r < B; ++r)  // empty slots: row 0's memory (never read back)
      CUDA_CHECK(cudaMemcpyAsync(d_mem_.get() + r * kMem, d_mem_.get(), sizeof(float) * kMem,
                                 cudaMemcpyDeviceToDevice, enc_stream_));
    prep_->set_tensor_address("memory", d_mem_.get());
    prep_->set_tensor_address("ck", ck);
    prep_->set_tensor_address("cv", ck + kCkv);
    if (!prep_->execute(enc_stream_)) throw std::runtime_error("formula prep failed");
    if (job.done()) CUDA_CHECK(cudaEventRecord(set_ready_[job.set], enc_stream_));
  }
}

void FormulaRecognizer::decode_cohort(int n, int set, std::vector<std::vector<int64_t>> &toks, cudaStream_t s,
                                      const std::vector<FormulaCrop> &crops, const std::vector<int> &order,
                                      EncodeJob &next) {
  const int64_t eos = tok_->eos_id();
  struct Group {
    int slot;  // its cross-KV and next tokens: d_ckv_ set/slot, d_tok_ slot
    int kv;    // its self-attention KV: d_kv_ buffer
    Rows rows;
    [[nodiscard]] int live() const { return (int)std::count_if(rows.r, rows.r + B, [](int r) { return r >= 0; }); }
  };
  auto kv = [&](int buf) { return d_kv_.get() + (size_t)buf * 2 * kKv; };  // v at + kKv; [2, nb, H, len, Dh]
  auto ckv = [&](int slot) { return d_ckv_.get() + (size_t)(set * G + slot) * 2 * kCkv; };  // cv at + kCkv
  auto tok = [&](int slot) { return d_tok_.get() + (size_t)slot * B * 3; };
  volatile int *done = h_eos_.get();
  for (int i = 0; i < n; ++i) done[i] = 0;  // the previous cohort's kernels are over (its decode synced)

  CUDA_CHECK(cudaStreamWaitEvent(s, set_ready_[set]));
  const int nb = n == 1 ? 1 : B;  // step batch size
  size_t li = 0;                  // KV length bucket
  int len = len_buckets_[li];
  if (!set_step_batch(nb, len)) throw std::runtime_error("formula step shape");
  std::vector<Group> gs;
  std::vector<int> spare;  // KV buffers no group holds
  for (int g = 0; g < G + 1; ++g) {
    if (g * B >= n || g == G) { spare.push_back(g); continue; }
    Group grp{g, g, {}};
    for (int b = 0; b < B; ++b) grp.rows.r[b] = g * B + b < n ? g * B + b : -1;
    gs.push_back(grp);
    CUDA_CHECK(cudaMemsetAsync(kv(g), 0, 2 * kKv * sizeof(float), s));
    CUDA_CHECK(cudaMemsetAsync(tok(g), 0, (size_t)B * 3 * sizeof(int64_t), s));  // <s> = 0
  }
  // Row a of group `from` -> empty slot b of group `to`: everything its next step reads.
  auto move = [&](Group &from, int a, Group &to, int b) {
    const size_t row = (size_t)H * len * Dh;
    for (int l = 0; l < 2; ++l) {
      const size_t sk = (size_t)(l * B + a), dk = (size_t)(l * B + b);
      for (size_t off : {size_t{0}, kKv})
        CUDA_CHECK(cudaMemcpyAsync(kv(to.kv) + off + dk * row, kv(from.kv) + off + sk * row, row * sizeof(float),
                                   cudaMemcpyDeviceToDevice, s));
      for (size_t off : {size_t{0}, kCkv})
        CUDA_CHECK(cudaMemcpyAsync(ckv(to.slot) + off + dk * kCkvRow, ckv(from.slot) + off + sk * kCkvRow,
                                   kCkvRow * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    CUDA_CHECK(cudaMemcpyAsync(tok(to.slot) + b * 3, tok(from.slot) + a * 3, 3 * sizeof(int64_t),
                               cudaMemcpyDeviceToDevice, s));
    to.rows.r[b] = from.rows.r[a];
    from.rows.r[a] = -1;
  };

  constexpr int R = kEosLag + 1;
  int it = 0;
  for (; it < MAXIT; ++it) {
    if (it >= kEosLag) {  // step it - kEosLag has run: drop the rows that have ended by then
      CUDA_CHECK(cudaEventSynchronize(step_done_[(it - kEosLag) % R]));
      for (Group &g : gs)
        for (int &r : g.rows.r)
          if (r >= 0 && done[r]) r = -1;
      int live = 0;
      for (const Group &g : gs) live += g.live();
      // Pack the live rows into as few step batches as hold them: empty the emptiest batch.
      while ((int)gs.size() > (live + B - 1) / B) {
        auto src = std::min_element(gs.begin(), gs.end(), [](const Group &a, const Group &b) { return a.live() < b.live(); });
        for (int a = 0; a < B; ++a) {
          if (src->rows.r[a] < 0) continue;
          for (Group &dst : gs) {
            if (&dst == &*src) continue;
            const int b = (int)(std::find(dst.rows.r, dst.rows.r + B, -1) - dst.rows.r);
            if (b < B) { move(*src, a, dst, b); break; }
          }
        }
        spare.push_back(src->kv);
        gs.erase(src);
      }
      if (gs.empty()) break;
    }
    if (3 * it + 3 > len) {  // the next length bucket: re-lay each group's KV out at the longer stride
      while (3 * it + 3 > len_buckets_[li]) ++li;
      const int to = len_buckets_[li];
      for (Group &g : gs) {
        const int out = spare.back();
        spare.pop_back();
        for (size_t off : {size_t{0}, kKv}) {
          CUDA_CHECK(cudaMemsetAsync(kv(out) + off, 0, (size_t)2 * nb * H * to * Dh * sizeof(float), s));
          CUDA_CHECK(cudaMemcpy2DAsync(kv(out) + off, (size_t)to * Dh * sizeof(float), kv(g.kv) + off,
                                       (size_t)len * Dh * sizeof(float), (size_t)len * Dh * sizeof(float),
                                       (size_t)2 * nb * H, cudaMemcpyDeviceToDevice, s));
        }
        spare.push_back(g.kv);
        g.kv = out;
      }
      len = to;
      if (!set_step_batch(nb, len)) throw std::runtime_error("formula step shape");
    }
    for (Group &g : gs) {
      const int out = spare.back();
      spare.pop_back();
      step_->set_tensor_address("tokens", tok(g.slot));
      step_->set_tensor_address("pos", d_pos_.get() + it);
      step_->set_tensor_address("ck", ckv(g.slot));
      step_->set_tensor_address("cv", ckv(g.slot) + kCkv);
      step_->set_tensor_address("kb", kv(g.kv));
      step_->set_tensor_address("vb", kv(g.kv) + kKv);
      step_->set_tensor_address("kb_out", kv(out));
      step_->set_tensor_address("vb_out", kv(out) + kKv);
      step_->set_tensor_address("logits", d_logits_.get());
      if (!step_->execute(s)) throw std::runtime_error("formula step failed");
      argmax_rows<<<nb * 3, 256, 0, s>>>(d_logits_.get(), tok(g.slot), d_hist_.get(), g.rows, it, eos, h_eos_.get(), VOCAB);
      spare.push_back(g.kv);  // free once this step has read it (stream order)
      g.kv = out;
      ++steps_run;
    }
    CUDA_CHECK(cudaEventRecord(step_done_[it % R], s));
    if (!next.done()) encode_step(crops, order, next);  // the next cohort, a crop per step
  }
  CUDA_CHECK(cudaEventRecord(set_free_[set], s));
  std::vector<int64_t> hist((size_t)n * MAXIT * 3);
  CUDA_CHECK(cudaMemcpyAsync(hist.data(), d_hist_.get(), hist.size() * sizeof(int64_t), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  for (int r = 0; r < n; ++r) {
    bool stop = false;
    for (int st = 0; st < it && !stop; ++st)
      for (int j = 0; j < 3; ++j) {
        const int64_t t = hist[((size_t)r * MAXIT + st) * 3 + j];
        if (t == eos) { stop = true; break; }
        if (t != 0 && t != 1) toks[r].push_back(t);
      }
  }
}

std::vector<std::string> FormulaRecognizer::run(const std::vector<FormulaCrop> &crops, cudaStream_t s) {
  const int n = (int)crops.size();
  std::vector<std::string> res(n);
  if (n == 0) return res;
  std::vector<int> ink(n), order(n);
  for (int i = 0; i < n; ++i) ink[i] = ink_pixels(crops[i]), order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ink[a] > ink[b]; });
  auto job = [&](int k) { return EncodeJob{k * kCohort, std::clamp(n - k * kCohort, 0, kCohort), k % 2, 0}; };
  EncodeJob next = job(0);
  while (!next.done()) encode_step(crops, order, next);
  for (int k = 0; k * kCohort < n; ++k) {
    const int c0 = k * kCohort, m = std::min(kCohort, n - c0);
    next = job(k + 1);
    std::vector<std::vector<int64_t>> toks(m);
    decode_cohort(m, k % 2, toks, s, crops, order, next);
    while (!next.done()) encode_step(crops, order, next);
    for (int r = 0; r < m; ++r) {
      const int i = order[c0 + r];
      res[i] = tok_->decode(toks[r], /*post_process=*/false);
      if (!dump_dir.empty()) {
        std::ofstream t(dump_dir + "/tok_" + std::to_string(dump_n + i) + ".txt");
        for (auto id : toks[r]) t << id << ' ';
      }
    }
  }
  dump_n += n;
  return res;
}

}  // namespace hp
