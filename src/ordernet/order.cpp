// OrderNet post-processing: a port of indra_ocr's order_post.py (line features, reading-order
// decoding) and layout_post.py (blocks -> Markdown). See order.h.
//
// The reference is numpy code; where its arithmetic decides the output, it is reproduced:
// float32 where numpy computes in float32 (NEP 50: a Python float operand does not widen a
// float32 array), numpy's pairwise summation for float32 sums over a contiguous axis
// (sequential sums over a strided axis and in cumsum), np.median (mean of the two middle
// values, in the array's dtype), argmax returning the first maximum. numpy's default argsort
// is unstable (introsort / SIMD sort); std::stable_sort is used, so the two agree except on
// equal keys. This file is built with -ffp-contract=off: no FMA contraction.

#include "ordernet/order.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <string_view>

#include "ordernet/text.h"

namespace hp::on {
namespace {

using Box = std::array<float, 4>;
constexpr double kInf = std::numeric_limits<double>::infinity();

// numpy's pairwise summation (float32 add.reduce over a contiguous run): blocks of up to 128
// summed with 8 accumulators, larger runs split in two at a multiple of 8.
float pairwise_sum(const float *a, ptrdiff_t n) {
  if (n < 8) {
    float res = -0.0f;
    for (ptrdiff_t i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= 128) {
    float r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    ptrdiff_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  ptrdiff_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// np.sum / np.mean of a 1-D float32 array.
float np_sum(const std::vector<float> &v) { return 0.0f + pairwise_sum(v.data(), static_cast<ptrdiff_t>(v.size())); }
float np_mean(const std::vector<float> &v) { return np_sum(v) / static_cast<float>(v.size()); }

// np.median of a float32 array (float32 result: the middle value, or the float32 mean of the
// two middle values). v is taken by value and reordered.
float median_f32(std::vector<float> v) {
  const size_t n = v.size(), m = n / 2;
  std::nth_element(v.begin(), v.begin() + m, v.end());
  if (n % 2) return v[m];
  const float lo = *std::max_element(v.begin(), v.begin() + m);
  return (lo + v[m]) / 2.0f;
}

// np.median of float64 values.
double median_f64(std::vector<double> v) {
  const size_t n = v.size(), m = n / 2;
  std::nth_element(v.begin(), v.begin() + m, v.end());
  if (n % 2) return v[m];
  const double lo = *std::max_element(v.begin(), v.begin() + m);
  return (lo + v[m]) / 2.0;
}

// np.argmax: index of the first maximum.
template <class T>
int argmax(const T *v, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (v[i] > v[best]) best = i;
  return best;
}

// np.argsort, but stable (see the note at the top).
template <class T>
std::vector<int> argsort(const std::vector<T> &v) {
  std::vector<int> o(v.size());
  std::iota(o.begin(), o.end(), 0);
  std::stable_sort(o.begin(), o.end(), [&](int i, int j) { return v[i] < v[j]; });
  return o;
}

// Python round() of a float: half to even (the default rounding mode).
double py_round(double v) { return std::nearbyint(v); }

// Block class of a set of lines: argmax of the column sums of their class logits
// (cls_logits[blk].sum(0).argmax(); a sum over axis 0 adds the rows in order).
int block_class(const Heads &h, const std::vector<int> &blk) {
  std::array<float, kClasses> s{};
  for (int i : blk)
    for (int c = 0; c < kClasses; ++c) s[c] += h.cls[static_cast<size_t>(i) * kClasses + c];
  return argmax(s.data(), kClasses);
}

constexpr int kSection = 1, kHeader = 2, kFooter = 3, kList = 4, kTable = 5, kToc = 8, kForm = 9;

bool is_grid(int c) { return c == kTable || c == kForm || c == kToc; }  // GRID_LABELS

std::vector<Box> boxes_of(const std::vector<Line> &lines, const std::vector<int> &idx) {
  std::vector<Box> b;
  b.reserve(idx.size());
  for (int i : idx) b.push_back(lines[i].box);
  return b;
}

// ---------------------------------------------------------------------------
// order_post: XY-cut
// ---------------------------------------------------------------------------

struct Cut {
  float gap = 0.0f;
  int k = -1;
  std::vector<int> o;
};

// The widest whitespace gap along one axis (1: between rows, 0: between columns): lines
// sorted by their low edge, gap = next low edge - furthest high edge so far.
Cut best_cut(const std::vector<Box> &b, int axis) {
  const int lo_i = axis == 1 ? 1 : 0, hi_i = axis == 1 ? 3 : 2;
  std::vector<float> lo(b.size());
  for (size_t i = 0; i < b.size(); ++i) lo[i] = b[i][lo_i];
  Cut c;
  c.o = argsort(lo);
  float end = b[c.o[0]][hi_i];
  for (size_t k = 1; k < c.o.size(); ++k) {
    const float gap = lo[c.o[k]] - end;
    if (gap > c.gap) {
      c.gap = gap;
      c.k = static_cast<int>(k);
    }
    end = std::max(end, b[c.o[k]][hi_i]);
  }
  return c;
}

void xycut_rec(const std::vector<Box> &boxes, const std::vector<int> &ids, std::vector<int> &out) {
  if (ids.size() <= 1) {
    out.insert(out.end(), ids.begin(), ids.end());
    return;
  }
  std::vector<Box> b(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) b[i] = boxes[ids[i]];
  Cut cy = best_cut(b, 1), cx = best_cut(b, 0);
  if (cy.k < 0 && cx.k < 0) {
    for (int i : order_top_down(b)) out.push_back(ids[i]);
    return;
  }
  const Cut &c = cy.gap >= cx.gap ? cy : cx;
  std::vector<int> first, second;
  for (int k = 0; k < static_cast<int>(c.o.size()); ++k) (k < c.k ? first : second).push_back(ids[c.o[k]]);
  xycut_rec(boxes, first, out);
  xycut_rec(boxes, second, out);
}

}  // namespace

// ---------------------------------------------------------------------------
// order_post: average-linkage clustering
// ---------------------------------------------------------------------------

// Not in order.h: tests/ordernet_parity_test declares it to rebuild the blocks.
namespace detail {
std::vector<std::vector<int>> cluster(const std::vector<float> &logits, int n);
}

// Average-linkage clustering of pairwise "same group" logits [n, n]: merge the two groups with
// the highest mean logit between them while that mean is positive (probability above one
// half). Unlike connected components, one wrong pair cannot join two groups that the other
// pairs keep apart.
//
// A merge changes only the merged group's row and column of the mean matrix, and each row
// keeps its best partner, so a page of n lines costs about n^2 instead of n^3. Groups come
// out by their smallest member, members sorted.
std::vector<std::vector<int>> detail::cluster(const std::vector<float> &logits, int n) {
  if (n == 0) return {};
  const size_t N = static_cast<size_t>(n);
  std::vector<double> S(N * N), M(N * N);
  for (size_t i = 0; i < N * N; ++i) S[i] = logits[i];
  for (size_t i = 0; i < N; ++i) S[i * N + i] = 0.0;
  M = S;  // mean logit between groups; -inf on the diagonal and dead groups
  for (size_t i = 0; i < N; ++i) M[i * N + i] = -kInf;
  std::vector<double> size(N, 1.0), best_v(N), row(N);
  std::vector<char> alive(N, 1);
  std::vector<int> best(N);
  std::vector<std::vector<int>> members(N);
  for (int i = 0; i < n; ++i) {
    members[i] = {i};
    best[i] = argmax(&M[i * N], n);
    best_v[i] = M[i * N + best[i]];
  }
  for (int it = 0; it < n - 1; ++it) {
    int a = argmax(best_v.data(), n);
    int b = best[a];
    if (best_v[a] <= 0) break;
    if (a > b) std::swap(a, b);
    double *Sa = &S[a * N];
    const double *Sb = &S[b * N];
    for (size_t j = 0; j < N; ++j) Sa[j] += Sb[j];
    for (size_t j = 0; j < N; ++j) S[j * N + a] += S[j * N + b];
    Sa[a] = 0.0;
    size[a] += size[b];
    alive[b] = 0;
    members[a].insert(members[a].end(), members[b].begin(), members[b].end());
    for (size_t j = 0; j < N; ++j) row[j] = alive[j] ? Sa[j] / (size[a] * size[j]) : -kInf;
    row[a] = -kInf;
    for (size_t j = 0; j < N; ++j) {
      M[a * N + j] = M[j * N + a] = row[j];
      M[b * N + j] = M[j * N + b] = -kInf;
    }
    best_v[b] = -kInf;
    best[a] = argmax(row.data(), n);
    best_v[a] = row[best[a]];
    // other rows: (j, a) changed and (j, b) is gone
    for (int j = 0; j < n; ++j) {
      if (j == a || !alive[j] || (best[j] != a && best[j] != b)) continue;
      best[j] = argmax(&M[j * N], n);
      best_v[j] = M[j * N + best[j]];
    }
    for (int j = 0; j < n; ++j) {
      if (j != a && alive[j] && row[j] > best_v[j]) {
        best[j] = a;
        best_v[j] = row[j];
      }
    }
  }
  std::vector<std::vector<int>> out;
  for (int i = 0; i < n; ++i) {
    if (!alive[i]) continue;
    std::sort(members[i].begin(), members[i].end());
    out.push_back(std::move(members[i]));
  }
  return out;
}

namespace {

// ---------------------------------------------------------------------------
// layout_post: text joining and tables
// ---------------------------------------------------------------------------

constexpr std::u32string_view kBullets = U"•-–▪◦*·→";

// Han, kana, CJK punctuation and fullwidth forms: written without spaces (Korean Hangul is
// not: Korean separates words with spaces).
bool cjk(char32_t o) {
  return (0x3000 <= o && o <= 0x30FF) || (0x3400 <= o && o <= 0x9FFF) || (0xF900 <= o && o <= 0xFAFF) ||
         (0xFF00 <= o && o <= 0xFFEF) || (0x20000 <= o && o <= 0x2FA1F);
}

// Join the lines of one paragraph. 'tumori-' + 'genesis' -> 'tumorigenesis'; a line break
// between two CJK characters takes no space.
std::string join_lines(const std::vector<std::u32string> &texts) {
  std::u32string out;
  for (const auto &raw : texts) {
    const std::u32string_view t = py::strip(raw);
    if (t.empty()) continue;
    if (out.size() > 1 && out.back() == U'-' && py::isalpha(out[out.size() - 2]) && py::isalpha(t[0]) &&
        py::islower(t[0])) {
      out.pop_back();
      out += t;
    } else if (!out.empty() && cjk(out.back()) && cjk(t[0])) {
      out += t;
    } else {
      if (!out.empty()) out += U' ';
      out += t;
    }
  }
  return py::utf8(out);
}

// _WHOLE_MATH = ^\$[^$]+\$$ on a stripped line: a line that is one formula.
bool whole_math(std::u32string_view s) {
  return s.size() >= 3 && s.front() == U'$' && s.back() == U'$' &&
         s.substr(1, s.size() - 2).find(U'$') == std::u32string_view::npos;
}

// _ITEM = ^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$): a list marker, followed by
// text or alone on its line (the detector often cuts a bullet off as a line of its own; the
// next line then joins its item). \s and \d are Python's Unicode classes.
bool item_start(std::u32string_view t) {
  const size_t n = t.size();
  auto tail = [&](size_t q) { return q == n || (q < n && py::isspace(t[q])); };
  auto closer = [&](size_t q) { return q < n && (t[q] == U'.' || t[q] == U')'); };
  size_t p = 0;
  while (p < n && py::isspace(t[p])) ++p;
  if (p == n) return false;
  if (py::contains(kBullets, t[p]) && tail(p + 1)) return true;
  const size_t q = p + (t[p] == U'(' ? 1 : 0);
  size_t run = 0;
  while (q + run < n && run < 4 && py::isdecimal(t[q + run])) ++run;
  if (run >= 1 && run <= 3 && closer(q + run) && tail(q + run + 1)) return true;
  const bool ascii_letter = q < n && ((t[q] >= U'a' && t[q] <= U'z') || (t[q] >= U'A' && t[q] <= U'Z'));
  return ascii_letter && closer(q + 1) && tail(q + 2);
}

// np.searchsorted(edges, x, side="right") with numpy 2.x's branchless binary search. The
// edges are not always sorted (overlapping column bands), and then the probe sequence decides
// the answer; checked against numpy on random unsorted arrays.
int searchsorted_right(const std::vector<double> &edges, double x) {
  size_t base = 0, n = edges.size();
  if (n == 0) return 0;
  while (n > 1) {
    const size_t half = n / 2;
    if (edges[base + half] <= x) base += half;
    n -= half;
  }
  return static_cast<int>(base + (edges[base] <= x ? 1 : 0));
}

using Rows = std::vector<std::vector<int>>;

// Group cell lines into rows by vertical centre, top to bottom, left to right.
Rows rows_of(const std::vector<Box> &boxes) {
  const size_t n = boxes.size();
  std::vector<float> cy(n), hs(n);
  for (size_t i = 0; i < n; ++i) {
    cy[i] = (boxes[i][1] + boxes[i][3]) / 2.0f;
    hs[i] = boxes[i][3] - boxes[i][1];
  }
  const float half_h = static_cast<float>(0.5 * static_cast<double>(median_f32(hs)));
  Rows rows;
  std::vector<float> cur;  // cy of the last row's lines
  for (int i : argsort(cy)) {
    if (!rows.empty() && std::fabs(cy[i] - np_mean(cur)) < half_h) {
      rows.back().push_back(i);
      cur.push_back(cy[i]);
    } else {
      rows.push_back({i});
      cur.assign(1, cy[i]);
    }
  }
  for (auto &r : rows)
    std::stable_sort(r.begin(), r.end(), [&](int a, int b) { return boxes[a][0] < boxes[b][0]; });
  return rows;
}

// Join the line rows of wrapped cells into one table row.
//
// Inside a wrapped cell the gap to the next line is the font's line gap; a new table row adds
// cell padding. A row joins the row above when its gap is under half the table's median row
// gap and under 0.6 line heights, and it has fewer lines than the row above (a continuation
// fills only the wrapped columns).
Rows merge_wrapped(const Rows &rows, const std::vector<Box> &boxes) {
  if (rows.size() < 3) return rows;
  std::vector<float> hs(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i) hs[i] = boxes[i][3] - boxes[i][1];
  const double h = median_f32(hs);
  std::vector<double> gaps;
  for (size_t r = 0; r + 1 < rows.size(); ++r) {
    float top = boxes[rows[r + 1][0]][1], bottom = boxes[rows[r][0]][3];
    for (int k : rows[r + 1]) top = std::min(top, boxes[k][1]);
    for (int k : rows[r]) bottom = std::max(bottom, boxes[k][3]);
    gaps.push_back(top - bottom);
  }
  const double med = median_f64(gaps);
  Rows out = {rows[0]};
  for (size_t r = 1; r < rows.size(); ++r) {
    const double g = gaps[r - 1];
    // A continuation fills only the wrapped columns: fewer lines than the row above.
    if (g < 0.5 * med && g < 0.6 * h && rows[r].size() < rows[r - 1].size())
      out.back().insert(out.back().end(), rows[r].begin(), rows[r].end());
    else
      out.push_back(rows[r]);
  }
  return out;
}

// An HTML table from cell lines (one text line per cell, as born-digital PDFs give).
//
// A first row that is one line across most of the table is a title: it goes before the table
// as text. The first remaining row is the header (<th>).
std::string table_html(const std::vector<Box> &boxes, const std::vector<std::string> &texts) {
  Rows rows = rows_of(boxes);
  float xmax = boxes[0][2], xmin = boxes[0][0];
  for (const Box &b : boxes) {
    xmax = std::max(xmax, b[2]);
    xmin = std::min(xmin, b[0]);
  }
  const float span_x = xmax - xmin;
  size_t width = 0;
  for (const auto &r : rows) width = std::max(width, r.size());
  if (rows.size() > 2 && rows[0].size() == 1 && width > 1) {
    const Box &t = boxes[rows[0][0]];
    if (t[2] - t[0] > 0.6f * span_x) {
      std::vector<Box> kb;
      std::vector<std::string> kt;
      for (size_t k = 0; k < texts.size(); ++k) {
        if (static_cast<int>(k) == rows[0][0]) continue;
        kb.push_back(boxes[k]);
        kt.push_back(texts[k]);
      }
      return texts[rows[0][0]] + "\n\n" + table_html(kb, kt);
    }
  }
  Rows full;
  for (const auto &r : rows)
    if (r.size() == width) full.push_back(r);
  rows = merge_wrapped(rows, boxes);
  // Column bands from the rows with the most cells: those rows rarely span.
  const int W = static_cast<int>(width);
  std::vector<std::array<float, 2>> bands(W);
  for (int c = 0; c < W; ++c) {
    std::vector<float> x0s, x1s;
    for (const auto &r : full) {
      x0s.push_back(boxes[r[c]][0]);
      x1s.push_back(boxes[r[c]][2]);
    }
    bands[c] = {median_f32(x0s), median_f32(x1s)};
  }
  // Widen each band to the midpoints between neighbours, so every x falls in one band.
  std::vector<double> edges = {-kInf};
  for (int c = 0; c + 1 < W; ++c) edges.push_back((bands[c][1] + bands[c + 1][0]) / 2.0f);
  edges.push_back(kInf);

  std::string out = "<table>";
  for (size_t ri = 0; ri < rows.size(); ++ri) {
    const char *tag = ri == 0 && rows.size() > 1 ? "th" : "td";
    std::map<int, std::pair<int, std::string>> cells;  // first band -> (span, text)
    std::vector<int> r = rows[ri];
    std::stable_sort(r.begin(), r.end(), [&](int a, int b) {
      return boxes[a][1] != boxes[b][1] ? boxes[a][1] < boxes[b][1] : boxes[a][0] < boxes[b][0];
    });
    for (int k : r) {
      const float x0 = boxes[k][0], x1 = boxes[k][2];
      const int first = searchsorted_right(edges, x0) - 1;
      const int last = searchsorted_right(edges, x1) - 1;
      // A cell spans every band it covers by more than half of the band.
      std::vector<int> span;
      for (int c = std::max(first, 0); c <= std::min(last, W - 1); ++c)
        if (std::min(x1, bands[c][1]) - std::max(x0, bands[c][0]) > 0.5f * (bands[c][1] - bands[c][0]))
          span.push_back(c);
      if (span.empty()) span.push_back(std::min(std::max(first, 0), W - 1));
      const int n_span = static_cast<int>(span.size());
      auto it = cells.find(span[0]);
      if (it == cells.end())
        cells[span[0]] = {n_span, texts[k]};
      else
        it->second = {std::max(it->second.first, n_span), it->second.second + " " + texts[k]};
    }
    out += "\n<tr>";
    for (int c = 0; c < W;) {
      auto it = cells.find(c);
      if (it != cells.end()) {
        const int n = it->second.first;
        out += "<";
        out += tag;
        if (n > 1) out += " colspan=\"" + std::to_string(n) + "\"";
        out += ">" + py::html_escape(it->second.second) + "</" + tag + ">";
        c += n;
      } else {
        out += "<" + std::string(tag) + "></" + tag + ">";
        c += 1;
      }
    }
    out += "</tr>";
  }
  out += "\n</table>";
  return out;
}

// _block_part: one block's markdown appended to parts (nothing for dropped furniture).
void block_part(std::vector<std::string> &parts, const std::vector<int> &blk, const std::vector<Line> &lines,
                const Heads &h, bool keep_furniture) {
  const int cls = block_class(h, blk);
  if ((cls == kHeader || cls == kFooter) && !keep_furniture) return;
  std::vector<std::u32string> texts;
  texts.reserve(blk.size());
  for (int i : blk) texts.push_back(py::utf32(lines[i].text));
  if (std::all_of(texts.begin(), texts.end(), [](const auto &t) { return whole_math(py::strip(t)); })) {
    // display equations
    std::u32string s;
    for (size_t k = 0; k < texts.size(); ++k) {
      const std::u32string_view t = py::strip(texts[k]);
      if (k) s += U'\n';
      s += U"$$";
      s += t.substr(1, t.size() - 2);
      s += U"$$";
    }
    parts.push_back(py::utf8(s));
    return;
  }
  if (cls == kSection) {
    parts.push_back("## " + join_lines(texts));
  } else if (cls == kTable && blk.size() > 1) {
    std::vector<std::string> ct;
    for (int i : blk) ct.push_back(lines[i].text);
    parts.push_back(table_html(boxes_of(lines, blk), ct));
  } else if (cls == kList) {
    std::vector<std::vector<std::u32string>> items;
    for (const auto &t : texts) {
      if (items.empty() || item_start(t))
        items.push_back({t});
      else
        items.back().push_back(t);
    }
    std::string s;
    for (size_t k = 0; k < items.size(); ++k) {
      if (k) s += '\n';
      s += join_lines(items[k]);
    }
    parts.push_back(std::move(s));
  } else {
    parts.push_back(join_lines(texts));
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// order_post
// ---------------------------------------------------------------------------

// [N, 4] boxes (x0, y0, x1, y1) -> [N, kFeats] float32 features: 10 geometric (box / page size,
// centre, width, log height over the median line height, log aspect, median height / page
// height), 6 from the stripped text (log display length, digit share, list marker, trailing
// hyphen, sentence end, initial capital).
std::vector<float> line_features(const std::vector<Line> &lines, float W, float H) {
  const size_t n = lines.size();
  std::vector<float> f(n * kFeats, 0.0f);
  if (n == 0) return f;
  std::vector<float> w(n), h(n);
  for (size_t i = 0; i < n; ++i) {
    const Box &b = lines[i].box;
    w[i] = std::max(b[2] - b[0], 1e-3f);
    h[i] = std::max(b[3] - b[1], 1e-3f);
  }
  const float med_h = median_f32(h);
  // numpy: a float32 array op a Python float stays float32 (2 * W is exact in float32), except
  // the last column, np.full(n, med_h / H * 20.0), which is float64 until the final cast.
  const float W2 = 2.0f * W, H2 = 2.0f * H;
  const float scale = static_cast<float>(static_cast<double>(med_h) / static_cast<double>(H) * 20.0);
  for (size_t i = 0; i < n; ++i) {
    const Box &b = lines[i].box;
    float *g = &f[i * kFeats];
    g[0] = b[0] / W;
    g[1] = b[1] / H;
    g[2] = b[2] / W;
    g[3] = b[3] / H;
    g[4] = (b[0] + b[2]) / W2;
    g[5] = (b[1] + b[3]) / H2;
    g[6] = w[i] / W;
    // numpy's float32 log (SIMD) and logf differ by a few ulp at most
    g[7] = std::log(h[i] / med_h);
    g[8] = std::log(w[i] / h[i]) / 4.0f;
    g[9] = scale;
  }
  constexpr std::u32string_view kSentenceEnd = U".!?:;。！？：；．";
  for (size_t i = 0; i < n; ++i) {
    const std::u32string s = py::utf32(lines[i].text);
    const std::u32string_view t = py::strip(s);
    if (t.empty()) continue;
    float *x = &f[i * kFeats + 10];
    int wide = 0, digits = 0;
    for (char32_t c : t) {
      wide += py::is_wide(c) ? 2 : 1;
      digits += py::isdigit(c) ? 1 : 0;
    }
    // length in Latin-letter widths: a CJK character is about two wide, so a full CJK line
    // does not read as a short table cell
    x[0] = static_cast<float>(std::log1p(static_cast<double>(wide)) / 5.0);
    x[1] = static_cast<float>(static_cast<double>(digits) / static_cast<double>(t.size()));
    x[2] = py::contains(kBullets, t[0]) || (py::isdigit(py::rstrip(t.substr(0, 3), U".)")) && t.size() > 3);
    x[3] = t.back() == U'-';
    x[4] = py::contains(kSentenceEnd, t.back());
    x[5] = py::isupper(t[0]);
  }
  return f;
}

// Order from a pairwise logit matrix [n, n] (a[i, j] > 0: i before j).
//
// Start from the row-sum ranking, then local search with insertion moves: take each line out
// and put it back at the position that most increases the summed log-probability of the
// pairwise relations. Unlike adjacent swaps, an insertion can move a line far (e.g. a short
// last line of a column that the ranking put after the next column). O(n) per move.
std::vector<int> decode_order(const std::vector<float> &a, int n, int sweeps) {
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  if (n <= 1) return order;
  const size_t N = static_cast<size_t>(n);
  // lp = -np.logaddexp(0, -a) = log P(i before j), float32 as npy_logaddexpf computes it
  constexpr float kLn2 = 0.693147180559945309417232121458176568f;
  std::vector<float> lp(N * N);
  for (size_t i = 0; i < N * N; ++i) {
    const float x = 0.0f, y = -a[i];
    float r;
    if (x == y) {
      r = x + kLn2;
    } else {
      const float t = x - y;
      if (t > 0)
        r = x + std::log1p(std::exp(-t));
      else if (t <= 0)
        r = y + std::log1p(std::exp(t));
      else
        r = t;  // NaN
    }
    lp[i] = -r;
  }
  // np.argsort(-lp.sum(1) + lp.sum(0)): row sums pairwise, column sums row after row
  std::vector<float> col(N, 0.0f), score(N);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j) col[j] += lp[i * N + j];
  for (size_t i = 0; i < N; ++i) score[i] = -(0.0f + pairwise_sum(&lp[i * N], n)) + col[i];
  order = argsort(score);

  std::vector<int> rest(N - 1);
  std::vector<float> cb(N), ca(N);
  std::vector<double> gains(N);
  for (int s = 0; s < sweeps; ++s) {
    bool improved = false;
    const std::vector<int> snapshot = order;
    for (int x : snapshot) {
      int cur = 0;
      for (int k = 0, m = 0; k < n; ++k) {
        if (order[k] == x)
          cur = k;
        else
          rest[m++] = order[k];
      }
      // gains[k]: x inserted at k is after rest[:k] (cumsum of lp[rest, x]) and before
      // rest[k:] (reverse cumsum of lp[x, rest]); float32 cumsums, added in float64.
      cb[0] = 0.0f;
      for (int k = 0; k < n - 1; ++k) cb[k + 1] = cb[k] + lp[static_cast<size_t>(rest[k]) * N + x];
      ca[n - 1] = 0.0f;
      float acc = 0.0f;
      for (int k = n - 2; k >= 0; --k) {
        acc += lp[static_cast<size_t>(x) * N + rest[k]];
        ca[k] = acc;
      }
      for (int k = 0; k < n; ++k) gains[k] = static_cast<double>(cb[k]) + static_cast<double>(ca[k]);
      const int k = argmax(gains.data(), n);
      if (gains[k] > gains[cur] + 1e-6) {
        std::copy(rest.begin(), rest.begin() + k, order.begin());
        order[k] = x;
        std::copy(rest.begin() + k, rest.end(), order.begin() + k + 1);
        improved = true;
      }
    }
    if (!improved) break;
  }
  return order;
}

// Rows top to bottom (centres within half a line height share a row), left to right.
std::vector<int> order_top_down(const std::vector<std::array<float, 4>> &boxes) {
  const size_t n = boxes.size();
  if (n == 0) return {};
  std::vector<float> hs(n);
  for (size_t i = 0; i < n; ++i) hs[i] = boxes[i][3] - boxes[i][1];
  const float unit = std::max(median_f32(hs) * 0.5f, 1e-3f);
  std::vector<double> row(n);
  for (size_t i = 0; i < n; ++i) row[i] = py_round((boxes[i][1] + boxes[i][3]) / 2.0f / unit);
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    return row[a] != row[b] ? row[a] < row[b] : boxes[a][0] < boxes[b][0];
  });
  return idx;
}

// Recursive XY-cut: split at the widest whitespace gap on either axis. A horizontal cut
// (between rows) puts the upper part first; a vertical cut (between columns) puts the left
// part first. Without a gap, fall back to top-down order.
std::vector<int> order_xycut(const std::vector<std::array<float, 4>> &boxes) {
  std::vector<int> ids(boxes.size()), out;
  std::iota(ids.begin(), ids.end(), 0);
  out.reserve(ids.size());
  xycut_rec(boxes, ids, out);
  return out;
}

// Reading order with the layout heads: blocks first, then lines inside blocks.
//
// Blocks come from average-linkage clustering of the same-block logits (independent of the
// order). Inside a block, lines go by XY-cut, or row by row in a table / form /
// table-of-contents block. Blocks go by the mean pairwise order logit between them.
std::vector<int> decode_blocks_order(const Heads &h, const std::vector<Line> &lines) {
  const int n = h.n;
  if (n <= 1) {
    std::vector<int> o(std::max(n, 0));
    std::iota(o.begin(), o.end(), 0);
    return o;
  }
  const size_t N = static_cast<size_t>(n);
  const auto blocks = detail::cluster(h.same, n);
  std::vector<std::vector<int>> inner;
  for (const auto &b : blocks) {
    const auto bx = boxes_of(lines, b);
    const auto local = !h.cls.empty() && is_grid(block_class(h, b)) ? order_top_down(bx) : order_xycut(bx);
    std::vector<int> in;
    for (int k : local) in.push_back(b[k]);
    inner.push_back(std::move(in));
  }
  // Mean order logit between blocks p < q, antisymmetric: G a G^T with G the block-membership
  // matrix scaled by 1 / block size. (numpy runs it through BLAS; the sums here go in index
  // order, which can differ from BLAS in the last bits of a float64.)
  const int m = static_cast<int>(blocks.size());
  std::vector<double> T(static_cast<size_t>(m) * N, 0.0);  // G a
  for (int k = 0; k < m; ++k) {
    const double g = 1.0 / static_cast<double>(blocks[k].size());
    double *t = &T[k * N];
    for (int i : blocks[k])
      for (size_t j = 0; j < N; ++j) t[j] += g * static_cast<double>(h.a[i * N + j]);
  }
  std::vector<float> ab(static_cast<size_t>(m) * m, 0.0f);
  for (int p = 0; p < m; ++p) {
    for (int q = p + 1; q < m; ++q) {
      const double g = 1.0 / static_cast<double>(blocks[q].size());
      double v = 0.0;
      for (int j : blocks[q]) v += T[p * N + j] * g;
      ab[p * m + q] = static_cast<float>(v);
      ab[q * m + p] = static_cast<float>(-v);
    }
  }
  std::vector<int> out;
  out.reserve(N);
  for (int k : decode_order(ab, m)) out.insert(out.end(), inner[k].begin(), inner[k].end());
  return out;
}

// ---------------------------------------------------------------------------
// layout_post
// ---------------------------------------------------------------------------

// Runs of the reading order with a positive same-block logit between neighbours.
std::vector<std::vector<int>> split_blocks(const std::vector<int> &order, const Heads &h) {
  std::vector<std::vector<int>> blocks;
  const size_t N = static_cast<size_t>(h.n);
  for (int i : order) {
    if (!blocks.empty() && h.same[blocks.back().back() * N + i] > 0)
      blocks.back().push_back(i);
    else
      blocks.push_back({i});
  }
  return blocks;
}

// Markdown for one page: blocks split from the reading order and rendered by class.
//   Section-Header        "## " + joined text
//   Text, Caption, ...    one paragraph; a word split by a hyphen at a line end is joined
//   List-Group            one line per item (a new item starts at a bullet or number)
//   Table                 an HTML table assembled from the cell lines (rows by vertical
//                         centre, columns anchored on the fullest rows)
//   Page-Header / Footer  left out unless keep_furniture
// A block whose lines are all one formula ($...$) becomes display math, one $$...$$ per line.
std::string page_markdown(const std::vector<Line> &lines, const std::vector<int> &order, const Heads &h,
                          bool keep_furniture) {
  std::vector<std::string> parts;
  for (const auto &blk : split_blocks(order, h)) block_part(parts, blk, lines, h, keep_furniture);
  std::string out;
  for (const auto &p : parts) {
    if (p.empty()) continue;
    if (!out.empty()) out += "\n\n";
    out += p;
  }
  return out;
}

}  // namespace hp::on
