#include "tablenet/features.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>

#include "tablenet/text.h"

namespace hp::tn {
namespace {

constexpr int kFourier = 8;
constexpr uint8_t kDark = 170;  // grey below this is ink

float median_f32(std::vector<float> s) {
  std::stable_sort(s.begin(), s.end());
  const size_t n = s.size();
  if (n == 0) return 0.f;
  return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2.f;
}

// tn_core._NUM: ^[\s(\[$€£¥+\-–−±<>≤≥~]*[\d.,]+[%)\]*†‡§a-z]*$ (case-insensitive)
bool is_numeric_text(const std::string &t) {
  const std::u32string cs = utf32(t);
  static const std::u32string lead_set = U"([$€£¥+-–−±<>≤≥~";
  static const std::u32string tail_set = U"%)]*†‡§";
  auto lead = [&](char32_t c) { return is_whitespace(c) || lead_set.find(c) != std::u32string::npos; };
  auto body = [](char32_t c) { return is_ascii_digit(c) || c == U'.' || c == U','; };
  auto tail = [&](char32_t c) { return tail_set.find(c) != std::u32string::npos || is_ascii_alpha(c); };
  size_t i = 0;
  while (i < cs.size() && lead(cs[i]) && !body(cs[i])) ++i;
  const size_t start = i;
  while (i < cs.size() && body(cs[i])) ++i;
  if (i == start) return false;
  for (; i < cs.size(); ++i)
    if (!tail(cs[i])) return false;
  return true;
}

// Python str.isdigit, close enough for features.
bool is_digit(char32_t c) {
  return is_ascii_digit(c) || c == U'²' || c == U'³' || c == U'¹' || (c >= 0x2070 && c <= 0x2079) ||
         (c >= 0x2080 && c <= 0x2089);
}

std::array<float, 4> f32box(const std::array<double, 4> &b) {
  return {static_cast<float>(b[0]), static_cast<float>(b[1]), static_cast<float>(b[2]), static_cast<float>(b[3])};
}

// [N, 4]: a horizontal rule just above / below the unit, a vertical rule just left / right.
std::vector<std::array<float, 4>> unit_rule_features(const std::vector<std::array<float, 4>> &boxes,
                                                     const Rules &rules, const std::vector<float> &h) {
  std::vector<std::array<float, 4>> out(boxes.size(), {0, 0, 0, 0});
  for (size_t i = 0; i < boxes.size(); ++i) {
    const auto &b = boxes[i];
    const float hh = std::max(h[i], 0.5f);
    for (const auto &r : rules.h) {
      if (!(std::min(b[2], r[2]) - std::max(b[0], r[1]) > 0.f)) continue;
      const float above = b[1] - r[0], below = r[0] - b[3];
      if (above >= -0.3f * hh && above < 1.5f * hh) out[i][0] = 1.f;
      if (below >= -0.3f * hh && below < 1.5f * hh) out[i][1] = 1.f;
    }
    for (const auto &r : rules.v) {
      if (!(std::min(b[3], r[2]) - std::max(b[1], r[1]) > 0.f)) continue;
      const float left = b[0] - r[0], right = r[0] - b[2];
      if (left >= -0.3f * hh && left < 2.f * hh) out[i][2] = 1.f;
      if (right >= -0.3f * hh && right < 2.f * hh) out[i][3] = 1.f;
    }
  }
  return out;
}

// First position at or after `from` (below w) whose bit is `set`; w when there is none.
size_t next_bit(const uint64_t *row, size_t words, size_t from, size_t w, bool set) {
  size_t k = from / 64;
  if (k >= words) return w;
  const uint64_t flip = set ? 0 : ~uint64_t{0};
  uint64_t word = (row[k] ^ flip) & (~uint64_t{0} << (from % 64));
  for (;;) {
    if (word != 0) return std::min(k * 64 + static_cast<size_t>(__builtin_ctzll(word)), w);
    if (++k >= words) return w;
    word = row[k] ^ flip;
  }
}

using Runs = std::vector<std::vector<std::pair<size_t, size_t>>>;

Rules rules_from_runs(const Runs &hruns, const Runs &vruns, float scale) {
  constexpr float kThickMaxPt = 2.5f;
  Rules out;
  for (int axis = 0; axis < 2; ++axis) {
    const Runs &runs = axis == 0 ? hruns : vruns;
    const size_t lines_n = runs.size();
    std::vector<std::array<size_t, 3>> ids;  // (line, start, end)
    std::vector<size_t> first(lines_n + 1, 0);
    for (size_t line = 0; line < lines_n; ++line) {
      first[line] = ids.size();
      for (const auto &[s, e] : runs[line]) ids.push_back({line, s, e});
    }
    first[lines_n] = ids.size();
    std::vector<size_t> parent(ids.size());
    for (size_t k = 0; k < parent.size(); ++k) parent[k] = k;
    auto root = [&](size_t k) {
      while (parent[k] != k) {
        parent[k] = parent[parent[k]];
        k = parent[k];
      }
      return k;
    };
    for (size_t line = 1; line < lines_n; ++line)
      for (size_t a = first[line - 1]; a < first[line]; ++a)
        for (size_t b = first[line]; b < first[line + 1]; ++b) {
          const auto [l0, s0, e0] = ids[a];
          const auto [l1, s1, e1] = ids[b];
          if (s1 <= e0 + 1 && e1 + 1 >= s0) {
            const size_t ra = root(a), rb = root(b);
            if (ra != rb) parent[ra] = rb;
          }
        }
    // component extents, in root order (deterministic)
    std::vector<std::optional<std::array<size_t, 4>>> comp(ids.size());
    for (size_t k = 0; k < ids.size(); ++k) {
      const size_t r = root(k);
      const auto [line, s, e] = ids[k];
      if (!comp[r]) comp[r] = std::array<size_t, 4>{std::numeric_limits<size_t>::max(), 0, std::numeric_limits<size_t>::max(), 0};
      auto &c = *comp[r];
      c[0] = std::min(c[0], line);
      c[1] = std::max(c[1], line);
      c[2] = std::min(c[2], s);
      c[3] = std::max(c[3], e);
    }
    auto round1 = [](float v) { return std::round(v * 10.f) / 10.f; };
    for (const auto &oc : comp) {
      if (!oc) continue;
      const auto [l0, l1, s, e] = *oc;
      const float thick = static_cast<float>(l1 - l0 + 1);
      if (thick > kThickMaxPt * scale) continue;
      const float along = static_cast<float>(e - s + 1);
      const float mid = (static_cast<float>(l0) + thick / 2.f) / scale;
      const float a0 = static_cast<float>(s) / scale, a1 = (static_cast<float>(s) + along) / scale;
      (axis == 0 ? out.h : out.v).push_back({round1(mid), round1(a0), round1(a1)});
    }
  }
  return out;
}

// Area-average resample (OpenCV INTER_AREA when shrinking) of a grey image to (tw, th).
std::vector<uint8_t> resample_area(const std::vector<uint8_t> &g, int w, int h, int tw, int th) {
  auto weights = [](int n_in, int n_out, std::vector<size_t> &at, std::vector<std::pair<size_t, float>> &taps) {
    const float f = static_cast<float>(n_in) / static_cast<float>(n_out);
    at.assign(1, 0);
    taps.clear();
    for (int o = 0; o < n_out; ++o) {
      const float a = static_cast<float>(o) * f, b = (static_cast<float>(o) + 1.f) * f;
      auto i = static_cast<size_t>(std::floor(a));
      while (static_cast<float>(i) < b && i < static_cast<size_t>(n_in)) {
        const float cov = std::max(std::min(b, static_cast<float>(i) + 1.f) - std::max(a, static_cast<float>(i)), 0.f);
        if (cov > 0.f) taps.push_back({i, cov / f});
        ++i;
      }
      at.push_back(taps.size());
    }
  };
  std::vector<size_t> ax, ay;
  std::vector<std::pair<size_t, float>> wx, wy;
  weights(w, tw, ax, wx);
  weights(h, th, ay, wy);
  std::vector<float> tmp(static_cast<size_t>(h) * tw);
  for (int y = 0; y < h; ++y) {
    const uint8_t *row = g.data() + static_cast<size_t>(y) * w;
    float *dst = tmp.data() + static_cast<size_t>(y) * tw;
    for (int x = 0; x < tw; ++x) {
      float s = 0.f;
      for (size_t t = ax[x]; t < ax[x + 1]; ++t) s += static_cast<float>(row[wx[t].first]) * wx[t].second;
      dst[x] = s;
    }
  }
  std::vector<uint8_t> out(static_cast<size_t>(th) * tw);
  std::vector<float> acc(tw);
  for (int y = 0; y < th; ++y) {
    std::fill(acc.begin(), acc.end(), 0.f);
    for (size_t t = ay[y]; t < ay[y + 1]; ++t) {
      const float *src = tmp.data() + wy[t].first * tw;
      const float c = wy[t].second;
      for (int x = 0; x < tw; ++x) acc[x] += src[x] * c;
    }
    uint8_t *dst = out.data() + static_cast<size_t>(y) * tw;
    for (int x = 0; x < tw; ++x) dst[x] = static_cast<uint8_t>(std::clamp(std::round(acc[x]), 0.f, 255.f));
  }
  return out;
}

}  // namespace

std::vector<uint8_t> grey_from_bgr(const uint8_t *bgr, int w, int h, size_t stride) {
  std::vector<uint8_t> g(static_cast<size_t>(w) * h);
  for (int y = 0; y < h; ++y) {
    const uint8_t *src = bgr + static_cast<size_t>(y) * stride;
    uint8_t *dst = g.data() + static_cast<size_t>(y) * w;
    for (int x = 0; x < w; ++x, src += 3)
      dst[x] = static_cast<uint8_t>((4899u * src[0] + 9617u * src[1] + 1868u * src[2] + (1u << 13)) >> 14);
  }
  return g;
}

Rules detect_rules(const std::vector<uint8_t> &grey, int w_, int h_, float scale) {
  const auto w = static_cast<size_t>(w_), h = static_cast<size_t>(h_);
  if (w == 0 || h == 0) return {};
  // ink mask as one bitset per row
  const size_t words = (w + 63) / 64;
  std::vector<uint64_t> bits(words * h, 0);
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x)
      if (grey[y * w + x] < kDark) bits[y * words + x / 64] |= uint64_t{1} << (x % 64);
  // horizontal runs per row, vertical runs per column, of at least 3% of the side (15 px floor)
  const size_t lh = std::max<size_t>(15, static_cast<size_t>(static_cast<float>(w) * 0.03f));
  const size_t lv = std::max<size_t>(15, static_cast<size_t>(static_cast<float>(h) * 0.03f));
  Runs hruns(h), vruns(w);
  std::vector<size_t> open(w, 0);
  const std::vector<uint64_t> zero(words, 0);
  for (size_t y = 0; y < h; ++y) {
    const uint64_t *cur = bits.data() + y * words;
    const uint64_t *prev = y == 0 ? zero.data() : bits.data() + (y - 1) * words;
    for (size_t k = 0; k < words; ++k) {
      uint64_t changed = cur[k] ^ prev[k];
      while (changed != 0) {
        const size_t x = k * 64 + static_cast<size_t>(__builtin_ctzll(changed));
        changed &= changed - 1;
        if ((cur[k] >> (x % 64)) & 1) open[x] = y;
        else if (y - open[x] >= lv) vruns[x].push_back({open[x], y - 1});
      }
    }
    size_t x = next_bit(cur, words, 0, w, true);
    while (x < w) {
      const size_t end = next_bit(cur, words, x, w, false);
      if (end - x >= lh) hruns[y].push_back({x, end - 1});
      x = next_bit(cur, words, end, w, true);
    }
  }
  const uint64_t *last = bits.data() + (h - 1) * words;
  for (size_t x = 0; x < w; ++x)
    if (((last[x / 64] >> (x % 64)) & 1) && h - open[x] >= lv) vruns[x].push_back({open[x], h - 1});
  return rules_from_runs(hruns, vruns, scale);
}

PageImage page_image(const std::vector<uint8_t> &grey, int w, int h, float px_per_pt) {
  PageImage im;
  if (w <= 0 || h <= 0) return im;
  const float side = static_cast<float>(kImgSide);
  const float s = side / static_cast<float>(std::max(w, h));
  im.w = std::max(static_cast<int>(std::round(static_cast<float>(w) * s)), 8);
  im.h = std::max(static_cast<int>(std::round(static_cast<float>(h) * s)), 8);
  im.px = resample_area(grey, w, h, im.w, im.h);
  im.k = px_per_pt * s;
  return im;
}

ImgInput image_input(const PageImage &im, const std::vector<Unit> &units) {
  ImgInput in;
  in.canvas.assign(static_cast<size_t>(kImgSide) * kImgSide, 0.f);
  for (int y = 0; y < std::min(im.h, kImgSide); ++y)
    for (int x = 0; x < std::min(im.w, kImgSide); ++x)
      in.canvas[static_cast<size_t>(y) * kImgSide + x] = (255.f - static_cast<float>(im.px[static_cast<size_t>(y) * im.w + x])) / 255.f;
  const float f = im.k / static_cast<float>(kImgSide);
  for (const Unit &u : units) {
    const auto b = f32box(u.pbox);
    in.boxes.push_back({std::clamp(b[0] * f, 0.f, 1.f), std::clamp(b[1] * f, 0.f, 1.f), std::clamp(b[2] * f, 0.f, 1.f),
                        std::clamp(b[3] * f, 0.f, 1.f)});
  }
  in.valid[0] = static_cast<float>(im.w) / static_cast<float>(kImgSide);
  in.valid[1] = static_cast<float>(im.h) / static_cast<float>(kImgSide);
  return in;
}

std::vector<float> unit_features(const std::vector<Unit> &units, double page_w, double page_h, const Rules *rules) {
  const auto w_pg = static_cast<float>(page_w), h_pg = static_cast<float>(page_h);
  const size_t n = units.size();
  std::vector<float> f(n * kFeatDim, 0.f);
  if (n == 0) return f;
  std::vector<std::array<float, 4>> b, pb;
  std::vector<float> h, hraw;
  for (const Unit &u : units) {
    b.push_back(f32box(u.bx));
    pb.push_back(f32box(u.pbox));
    hraw.push_back(static_cast<float>(u.h));
    h.push_back(std::max(static_cast<float>(u.h), 0.5f));
  }
  const float hm = median_f32(h);
  std::vector<float> mid(n);
  for (size_t i = 0; i < n; ++i) mid[i] = (b[i][1] + b[i][3]) / 2.f;
  std::vector<std::array<float, 4>> rule_f;
  if (rules) rule_f = unit_rule_features(pb, *rules, hraw);
  constexpr float kInf = std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < n; ++i) {
    float n_row = 0.f, gap_l = 20.f, gap_r = 20.f, lmax = -kInf, rmin = kInf;
    bool has_l = false, has_r = false;
    for (size_t j = 0; j < n; ++j) {
      if (!(std::abs(mid[j] - mid[i]) < 0.5f * std::max(h[j], h[i]))) continue;
      n_row += 1.f;
      if (j == i) continue;
      if (b[j][2] <= b[i][0] + 0.5f * h[i]) {
        has_l = true;
        lmax = std::max(lmax, b[j][2]);
      }
      if (b[j][0] >= b[i][2] - 0.5f * h[i]) {
        has_r = true;
        rmin = std::min(rmin, b[j][0]);
      }
    }
    if (has_l) gap_l = std::min(20.f, (b[i][0] - lmax) / hm);
    if (has_r) gap_r = std::min(20.f, (rmin - b[i][2]) / hm);
    const float cx = (b[i][0] + b[i][2]) / 2.f / w_pg;
    const float cy = (b[i][1] + b[i][3]) / 2.f / h_pg;
    const std::string &t = units[i].text;
    const std::u32string chars = utf32(t);
    const auto len = static_cast<float>(chars.size());
    float n_alpha = 0, n_dig = 0, n_up = 0;
    for (char32_t c : chars) {
      n_alpha += is_alphabetic(c);
      n_dig += is_digit(c);
      n_up += is_uppercase(c);
    }
    const float dig = n_dig / std::max(len, 1.f);
    const float alpha = n_alpha / std::max(len, 1.f);
    const float upper = n_up / std::max(n_alpha, 1.f);
    const auto words = static_cast<float>(split_whitespace(t).size());
    const float numeric = is_numeric_text(t) ? 1.f : 0.f;
    const std::string_view tr = trim_end(t);
    const float endp = !tr.empty() && (tr.back() == '.' || tr.back() == ':' || tr.back() == ';' || tr.back() == ',') ? 1.f : 0.f;
    float *row = f.data() + i * kFeatDim;
    const float base[21] = {cx,
                            cy,
                            (b[i][2] - b[i][0]) / w_pg,
                            (b[i][3] - b[i][1]) / h_pg,
                            b[i][0] / w_pg,
                            b[i][2] / w_pg,
                            b[i][1] / h_pg,
                            b[i][3] / h_pg,
                            h[i] / hm,
                            std::log1p(len) / 5.f,
                            std::log1p(words) / 3.f,
                            dig,
                            alpha,
                            upper,
                            numeric,
                            endp,
                            std::log1p(n_row) / 3.f,
                            gap_l / 20.f,
                            gap_r / 20.f,
                            units[i].bold == 1 ? 1.f : 0.f,
                            units[i].bold < 0 ? 0.f : 1.f};
    std::copy(base, base + 21, row);
    int k = 21;
    for (int q = 0; q < kFourier; ++q) {
      const float fr = std::pow(2.f, static_cast<float>(q)) * std::numbers::pi_v<float>;
      row[k] = std::sin(fr * cx);
      row[k + 1] = std::cos(fr * cx);
      row[k + 2] = std::sin(fr * cy);
      row[k + 3] = std::cos(fr * cy);
      k += 4;
    }
    k += 8;  // projection signals: absent (all 0)
    if (rules) {
      std::copy(rule_f[i].begin(), rule_f[i].end(), row + k);
      row[k + 4] = 1.f;
    }
    k += 5;
    row[k] = -1.f;      // projection block id: absent
    row[k + 1] = -1.f;  // XY-cut leaf id: absent
  }
  return f;
}

std::vector<float> pair_rule_features(const std::vector<Unit> &units, const Rules &rules) {
  const size_t n = units.size();
  std::vector<float> out(n * n * 2, 0.f);
  constexpr float cap = 3.f;
  std::vector<std::array<float, 4>> boxes;
  for (const Unit &u : units) boxes.push_back(f32box(u.pbox));
  std::vector<float> cy(n), cx(n);
  for (size_t i = 0; i < n; ++i) {
    cy[i] = (boxes[i][1] + boxes[i][3]) / 2.f;
    cx[i] = (boxes[i][0] + boxes[i][2]) / 2.f;
  }
  // covers[i][r]: rule r runs under unit i's x-range (h) / beside its y-range (v)
  const size_t nh = rules.h.size(), nv = rules.v.size();
  std::vector<uint8_t> hcov(n * nh), vcov(n * nv);
  for (size_t i = 0; i < n; ++i) {
    const auto &b = boxes[i];
    for (size_t r = 0; r < nh; ++r) hcov[i * nh + r] = rules.h[r][1] <= b[2] && rules.h[r][2] >= b[0];
    for (size_t r = 0; r < nv; ++r) vcov[i * nv + r] = rules.v[r][1] <= b[3] && rules.v[r][2] >= b[1];
  }
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) {
      if (nh) {
        const float lo = std::min(cy[i], cy[j]), hi = std::max(cy[i], cy[j]);
        int c = 0;
        for (size_t r = 0; r < nh; ++r)
          c += rules.h[r][0] > lo && rules.h[r][0] < hi && hcov[i * nh + r] && hcov[j * nh + r];
        out[(i * n + j) * 2] = std::min(static_cast<float>(c), cap) / cap;
      }
      if (nv) {
        const float lo = std::min(cx[i], cx[j]), hi = std::max(cx[i], cx[j]);
        int c = 0;
        for (size_t r = 0; r < nv; ++r)
          c += rules.v[r][0] > lo && rules.v[r][0] < hi && vcov[i * nv + r] && vcov[j * nv + r];
        out[(i * n + j) * 2 + 1] = std::min(static_cast<float>(c), cap) / cap;
      }
    }
  return out;
}

}  // namespace hp::tn
