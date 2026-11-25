#include "tablenet/page.h"

#include <chrono>
#include <limits>
#include <optional>

#include "tablenet/text.h"

namespace hp::tn {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// The rotated reading direction (1..3) carrying at least 60% of a text-layer page's characters.
std::optional<int> main_rotation(const PageInput &in) {
  if (in.ocr) return std::nullopt;
  size_t by_q[4] = {0, 0, 0, 0}, total = 0;
  for (const Unit &u : in.units) {
    size_t c = 0;
    for (char32_t ch : utf32(u.text)) c += !is_whitespace(ch);
    by_q[u.q & 3] += c;
    total += c;
  }
  for (int q = 1; q < 4; ++q)
    if (total > 0 && by_q[q] * 10 >= total * 6) return q;
  return std::nullopt;
}

// Ruling lines in reading direction q's frame (a quarter turn swaps horizontal and vertical).
Rules rotate_rules(const Rules &r, int q, double w, double h) {
  Rules out;
  std::vector<std::array<double, 4>> segs;
  for (const auto &s : r.h) segs.push_back({s[1], s[0], s[2], s[0]});
  for (const auto &s : r.v) segs.push_back({s[0], s[1], s[0], s[2]});
  for (const auto &sg : segs) {
    const auto b = to_frame(sg, q, w, h);
    const auto b0 = static_cast<float>(b[0]), b1 = static_cast<float>(b[1]), b2 = static_cast<float>(b[2]),
               b3 = static_cast<float>(b[3]);
    if (std::abs(b3 - b1) <= std::abs(b2 - b0)) out.h.push_back({(b1 + b3) / 2.f, std::min(b0, b2), std::max(b0, b2)});
    else out.v.push_back({(b0 + b2) / 2.f, std::min(b1, b3), std::max(b1, b3)});
  }
  return out;
}

// The image-branch page turned into reading direction q's frame.
PageImage rotate_image(const PageImage &im, int q) {
  const int w = im.w, h = im.h;
  PageImage out;
  out.w = q % 2 == 1 ? h : w;
  out.h = q % 2 == 1 ? w : h;
  out.k = im.k;
  out.px.assign(static_cast<size_t>(out.w) * out.h, 255);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int nx, ny;
      if (q == 1) nx = h - 1 - y, ny = x;
      else if (q == 2) nx = w - 1 - x, ny = h - 1 - y;
      else nx = y, ny = w - 1 - x;
      out.px[static_cast<size_t>(ny) * out.w + nx] = im.px[static_cast<size_t>(y) * w + x];
    }
  return out;
}

std::vector<Table> run_framed(TableNet &net, const std::vector<Unit> &units, double w, double h, const Rules &rules,
                              const PageImage &img, cudaStream_t stream, PageStats *stats) {
  const size_t n = units.size();
  size_t chars = 0;
  for (const Unit &u : units) chars += char_count(u.text);
  if (n < 2 || n > kMaxUnits || chars < 20) return {};
  auto t = Clock::now();
  const std::vector<float> feats = unit_features(units, w, h, &rules);
  const std::vector<float> pairf = pair_rule_features(units, rules);
  const ImgInput ii = image_input(img, units);
  if (stats) stats->features_ms += ms_since(t), stats->units += static_cast<int>(n);
  t = Clock::now();
  Logits l = net.forward(feats, pairf, static_cast<int>(n), ii, stream);
  if (stats) stats->net_ms += ms_since(t);
  t = Clock::now();
  std::vector<bool> minor(n);
  for (size_t i = 0; i < n; ++i) minor[i] = units[i].minor;
  mask_minor(l, minor);
  std::vector<DUnit> du;
  du.reserve(n);
  for (const Unit &u : units) du.push_back({u.bx, u.text, u.h});
  std::vector<Table> out;
  for (std::vector<int> g : decode_tables(du, l)) {
    std::erase_if(g, [&](int i) { return minor[i]; });
    if (g.size() < 2) continue;
    Grid grid = decode_grid(du, g, l);
    // one column is a list, not a table; tables of contents, charts and multi-column prose read
    // as grids are no tables either
    if (grid.n_cols < 2 || grid.is_toc() || grid.is_chart() || grid.is_prose()) continue;
    constexpr double inf = std::numeric_limits<double>::infinity();
    std::array<double, 4> b{inf, inf, -inf, -inf};
    for (int i : g) {
      const auto &p = units[i].pbox;
      b = {std::min(b[0], p[0]), std::min(b[1], p[1]), std::max(b[2], p[2]), std::max(b[3], p[3])};
    }
    out.push_back({b, std::move(g), std::move(grid)});
  }
  if (stats) stats->decode_ms += ms_since(t);
  return out;
}

}  // namespace

std::vector<Table> run_page(TableNet &net, const PageInput &in, cudaStream_t stream, PageStats *stats) {
  if (in.units.size() < 2 || in.units.size() > kMaxUnits || in.grey.empty()) return {};
  const Rules rules = detect_rules(in.grey, in.px_w, in.px_h, in.px_per_pt);
  const PageImage img = page_image(in.grey, in.px_w, in.px_h, in.px_per_pt);
  const std::optional<int> rq = main_rotation(in);
  if (!rq) return run_framed(net, in.units, in.w, in.h, rules, img, stream, stats);
  // A page whose text mostly runs in one rotated direction is turned upright first; every box
  // comes back to the page afterwards.
  const int q = *rq;
  std::vector<Unit> upright = in.units;
  for (Unit &u : upright) u.pbox = to_frame(u.pbox, q, in.w, in.h);
  const double fw = q % 2 == 1 ? in.h : in.w, fh = q % 2 == 1 ? in.w : in.h;
  std::vector<Table> tables =
      run_framed(net, upright, fw, fh, rotate_rules(rules, q, in.w, in.h), rotate_image(img, q), stream, stats);
  for (Table &t : tables) {
    t.bbox = from_frame(t.bbox, q, in.w, in.h);
    for (GridCell &c : t.grid.cells) c.bx = from_frame(c.bx, q, in.w, in.h);
  }
  return tables;
}

}  // namespace hp::tn
