// Decoding LayoutNet logits into tables: a C++ port of liteparse table_model/decode.rs (itself a
// port of bench/tables/tn_core.py), plus mask_minor and the Grid filters (is_toc / is_prose /
// is_chart) of table_model/mod.rs.
//
// decode_tables: unit groups from in_table + same_table (average linkage), then absorption of
// stacked pieces, small groups and stray units inside a table's box, then growth over a header
// band. decode_grid: one table's cells (same_cell, with stacked row-and-column pairs forced
// together), rows and columns over cells (span-aware linkage), header rows, the near-empty column
// fold, wrapped-row folds, and the grid as HTML. Order-sensitive steps (argmax ties, stable sorts,
// insertion order) and the f32 / f64 split of the arithmetic follow the Rust exactly, so the
// parity test (tests/tablenet_decode_test.cpp) compares against reference dumps byte for byte.

#include "tablenet/decode.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string_view>
#include <utility>

#include "tablenet/text.h"

// Rust never fuses a * b + c; with -march=native GCC would (gnu++20 defaults to
// -ffp-contract=fast), which changes rounding and so the decoded tables.
#pragma GCC optimize("fp-contract=off")

namespace hp::tn {
namespace {

constexpr float kCellMargin = 1.5f;
constexpr double kStackGap = 0.8;
constexpr double kStackGapHdr = 0.8;
constexpr double kGate = 3.0;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr size_t kNone = std::numeric_limits<size_t>::max();

using Box = std::array<double, 4>;

template <class T>
bool contains(const std::vector<T> &v, const T &x) {
  return std::find(v.begin(), v.end(), x) != v.end();
}

bool ends_with_any(std::string_view s, std::string_view set) {
  return !s.empty() && set.find(s.back()) != std::string_view::npos;
}

double median(std::vector<double> &v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// Median unit height, or 1 when it is 0 (`float(np.median(...)) or 1.0`).
double median_h(std::vector<double> hs) {
  const double m = median(hs);
  return (m == 0.0 || std::isnan(m)) ? 1.0 : m;
}
double median_h_all(const std::vector<DUnit> &units) {
  std::vector<double> hs;
  hs.reserve(units.size());
  for (const auto &u : units) hs.push_back(u.h);
  return median_h(std::move(hs));
}
double median_h_of(const std::vector<DUnit> &units, const std::vector<int> &idx) {
  std::vector<double> hs;
  hs.reserve(idx.size());
  for (int i : idx) hs.push_back(units[i].h);
  return median_h(std::move(hs));
}

// Units positive with two groups that are negative with each other (a spanning header).
std::vector<int> bridges(const std::vector<double> &p, size_t n) {
  const double neg = -1.0, frac = 0.3;
  std::vector<int> out;
  for (size_t i = 0; i < n; ++i) {
    std::vector<size_t> nb;
    for (size_t j = 0; j < n; ++j)
      if (j != i && p[i * n + j] > 0.0) nb.push_back(j);
    if (nb.size() < 2) continue;
    size_t tot = 0, negc = 0;
    for (size_t a = 0; a < nb.size(); ++a)
      for (size_t b = a + 1; b < nb.size(); ++b) {
        ++tot;
        if (p[nb[a] * n + nb[b]] < neg) ++negc;
      }
    if (tot > 0 && static_cast<double>(negc) / static_cast<double>(tot) > frac)
      out.push_back(static_cast<int>(i));
  }
  return out;
}

// Rows or columns: linkage over the non-bridging units; each bridging unit then joins every group
// it is positive with on average (a span), or its best group.
std::vector<std::vector<int>> span_link(const std::vector<double> &p, size_t n) {
  const std::vector<int> br = bridges(p, n);
  std::vector<int> core;
  for (size_t i = 0; i < n; ++i)
    if (!contains(br, static_cast<int>(i))) core.push_back(static_cast<int>(i));
  if (core.empty()) return avg_link(p, static_cast<int>(n), 0.0);
  const size_t m = core.size();
  std::vector<double> sub(m * m, 0.0);
  for (size_t a = 0; a < m; ++a)
    for (size_t b = 0; b < m; ++b) sub[a * m + b] = p[core[a] * n + core[b]];
  std::vector<std::vector<int>> groups = avg_link(sub, static_cast<int>(m), 0.0);
  for (auto &g : groups)
    for (int &i : g) i = core[i];
  for (int i : br) {
    std::vector<double> means;
    means.reserve(groups.size());
    for (const auto &g : groups) {
      double s = 0.0;
      for (int j : g) s += p[i * n + j];
      means.push_back(s / static_cast<double>(g.size()));
    }
    std::vector<size_t> hit;
    for (size_t k = 0; k < groups.size(); ++k)
      if (means[k] > 0.0) hit.push_back(k);
    if (hit.empty()) {
      size_t best = 0;
      for (size_t k = 1; k < means.size(); ++k)
        if (means[k] > means[best]) best = k;
      hit = {best};
    }
    for (size_t k : hit) groups[k].push_back(i);
  }
  for (auto &g : groups) std::sort(g.begin(), g.end());
  return groups;
}

// Box indices in reading order: lines (vertical overlap over half the shorter box), top to
// bottom, left to right within a line.
std::vector<size_t> reading_order(const std::vector<Box> &boxes) {
  std::vector<size_t> order(boxes.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return boxes[a][1] < boxes[b][1]; });
  struct Line {
    double top, bot;
    std::vector<size_t> ks;
  };
  std::vector<Line> lines;
  for (size_t k : order) {
    const Box &b = boxes[k];
    bool placed = false;
    for (auto &ln : lines) {
      const double ov = std::min(ln.bot, b[3]) - std::max(ln.top, b[1]);
      if (ov > 0.5 * std::min(ln.bot - ln.top, b[3] - b[1])) {
        ln.ks.push_back(k);
        ln.top = std::min(ln.top, b[1]);
        ln.bot = std::max(ln.bot, b[3]);
        placed = true;
        break;
      }
    }
    if (!placed) lines.push_back({b[1], b[3], {k}});
  }
  std::stable_sort(lines.begin(), lines.end(),
                   [](const Line &a, const Line &b) { return a.top < b.top; });
  std::vector<size_t> out;
  out.reserve(boxes.size());
  for (auto &ln : lines) {
    std::stable_sort(ln.ks.begin(), ln.ks.end(),
                     [&](size_t a, size_t b) { return boxes[a][0] < boxes[b][0]; });
    out.insert(out.end(), ln.ks.begin(), ln.ks.end());
  }
  return out;
}

Box bbox_of(const std::vector<DUnit> &units, const std::vector<int> &g) {
  Box b{kInf, kInf, -kInf, -kInf};
  for (int i : g) {
    const Box &u = units[i].bx;
    b[0] = std::min(b[0], u[0]);
    b[1] = std::min(b[1], u[1]);
    b[2] = std::max(b[2], u[2]);
    b[3] = std::max(b[3], u[3]);
  }
  return b;
}

// `re.match(r"^\s*(table|tab\.|figure|fig\.|exhibit|chart|source|note)s?\b", t, re.I)`: the
// alternatives in order, the optional "s" tried first (greedy), then a word boundary. Rust
// lowercases the whole text first; only ASCII letters (and the Kelvin sign, 'k') lowercase to
// the ASCII the words are made of, and lowercasing keeps a character alphanumeric, so comparing
// ASCII-lowercased code points is the same test.
bool is_caption(const std::string &text) {
  std::u32string t = utf32(trim_start(text));
  for (char32_t &c : t) {
    if (c >= U'A' && c <= U'Z') c += 32;
    else if (c == 0x212A) c = U'k';
  }
  auto word = [](char32_t c) { return is_alphanumeric(c) || c == U'_'; };
  auto boundary = [&](size_t pos) {
    const bool prev = pos > 0 && pos - 1 < t.size() && word(t[pos - 1]);
    const bool next = pos < t.size() && word(t[pos]);
    return prev != next;
  };
  for (std::u32string_view w :
       {U"table", U"tab.", U"figure", U"fig.", U"exhibit", U"chart", U"source", U"note"}) {
    if (t.size() < w.size() || std::u32string_view(t).substr(0, w.size()) != w) continue;
    const size_t end = w.size();
    if (end < t.size() && t[end] == U's' && boundary(end + 1)) return true;
    if (boundary(end)) return true;
  }
  return false;
}

// Grow each table up over its header band (see tn_core._head_band).
void head_band(const std::vector<DUnit> &units, const Logits &l,
               std::vector<std::vector<int>> &groups, std::vector<bool> &taken) {
  const double hm = median_h_all(units);
  const int n = static_cast<int>(units.size());
  for (auto &g : groups) {
    for (int it = 0; it < 3; ++it) {
      const Box b = bbox_of(units, g);
      const double x0 = b[0], y0 = b[1], x1 = b[2];
      std::vector<int> above;
      for (int i = 0; i < n; ++i) {
        const Box &u = units[i].bx;
        if (!taken[i] && l.node_at(i, 0) > -8.0f && u[3] <= y0 + 0.2 * hm &&
            y0 - u[3] < 4.0 * 1.4 * hm)
          above.push_back(i);
      }
      if (above.empty()) break;
      // the nearest line above: units overlapping vertically with the lowest one (max()'s first)
      int low = above[0];
      for (size_t a = 1; a < above.size(); ++a)
        if (units[above[a]].bx[3] > units[low].bx[3]) low = above[a];
      const Box lb = units[low].bx;
      std::vector<int> line;
      for (int i : above)
        if (std::min(units[i].bx[3], lb[3]) - std::max(units[i].bx[1], lb[1]) > 0.3 * hm)
          line.push_back(i);
      const double w = x1 - x0;
      bool ok = std::all_of(line.begin(), line.end(), [&](int i) {
        return units[i].bx[0] >= x0 - 0.05 * w && units[i].bx[2] <= x1 + 0.05 * w;
      });
      double line_bot = -kInf;
      for (int i : line) line_bot = std::max(line_bot, units[i].bx[3]);
      const bool near = y0 - line_bot < 1.5 * 1.4 * hm;
      const bool strong =
          std::all_of(line.begin(), line.end(), [&](int i) { return l.node_at(i, 0) > -1.5f; });
      const bool over = std::all_of(line.begin(), line.end(), [&](int i) {
        return std::any_of(g.begin(), g.end(), [&](int j) {
          return std::min(units[i].bx[2], units[j].bx[2]) -
                     std::max(units[i].bx[0], units[j].bx[0]) >
                 0.0;
        });
      });
      ok = ok && (strong || (near && over && std::all_of(line.begin(), line.end(), [&](int i) {
                               return l.node_at(i, 0) > -4.0f;
                             })));
      ok = ok && !std::any_of(line.begin(), line.end(),
                              [&](int i) { return is_caption(units[i].text); });
      const double hdr =
          static_cast<double>(std::count_if(line.begin(), line.end(),
                                            [&](int i) { return l.node_at(i, 1) > 0.0f; })) /
          static_cast<double>(std::max<size_t>(line.size(), 1));
      ok = ok && line.size() >= 2 && hdr > 0.5;
      ok = ok && !std::any_of(line.begin(), line.end(), [&](int i) {
        const std::string &t = units[i].text;
        const std::string_view te = trim_end(t);
        return split_whitespace(t).size() > 6 || (!te.empty() && te.back() == '.') ||
               (!te.empty() && te.back() == ':');
      });
      if (!ok) break;
      for (int i : line) {
        g.push_back(i);
        taken[i] = true;
      }
    }
    std::sort(g.begin(), g.end());
  }
}

// Fold column k into column t: its cells join the cell covering (row, t), or move there; spans
// through k shrink and the columns right of k move left.
void fold_col(std::vector<GridCell> &cells, int k, int t) {
  std::vector<size_t> movers;
  for (size_t i = 0; i < cells.size(); ++i)
    if (cells[i].c0 == k && cells[i].c1 == k) movers.push_back(i);
  std::vector<size_t> remove;
  for (size_t xi : movers) {
    const int r = cells[xi].r0;
    size_t tgt = kNone;
    for (size_t j = 0; j < cells.size(); ++j)
      if (cells[j].r0 <= r && r <= cells[j].r1 && cells[j].c0 <= t && t <= cells[j].c1) {
        tgt = j;
        break;
      }
    if (tgt != kNone && tgt != xi) {
      if (!cells[xi].text.empty()) {
        const std::string joined = t < k ? cells[tgt].text + " " + cells[xi].text
                                         : cells[xi].text + " " + cells[tgt].text;
        cells[tgt].text = std::string(trim(joined));
      }
      remove.push_back(xi);
    } else {
      cells[xi].c0 = t;
      cells[xi].c1 = t;
    }
  }
  std::sort(remove.begin(), remove.end());
  for (auto it = remove.rbegin(); it != remove.rend(); ++it)
    cells.erase(cells.begin() + static_cast<std::ptrdiff_t>(*it));
  for (auto &x : cells) {
    if (x.c0 <= k && k <= x.c1 && x.c0 < x.c1) {
      x.c1 -= 1;
    } else if (x.c0 > k) {
      x.c0 -= 1;
      x.c1 -= 1;
    }
  }
}

// Fold near-empty columns into their nearest neighbour (tn_core._squeeze, column rule).
void squeeze_cols(std::vector<GridCell> &cells, const std::vector<int> &head, int &n_cols,
                  double hh) {
  bool changed = true;
  while (changed && n_cols >= 3) {
    changed = false;
    for (int q = 0; q < n_cols; ++q) {
      std::vector<size_t> col;
      for (size_t k = 0; k < cells.size(); ++k)
        if (cells[k].c0 == q && cells[k].c1 == q && !cells[k].text.empty()) col.push_back(k);
      if (std::count_if(col.begin(), col.end(),
                        [&](size_t k) { return !contains(head, cells[k].r0); }) > 1)
        continue;
      std::vector<Box> boxes;
      for (size_t k : col) boxes.push_back(cells[k].bx);
      bool have_best = false;
      double best_gap = 0.0;
      int best_t = 0;
      for (int t : {q - 1, q + 1}) {
        if (t < 0 || t >= n_cols) continue;
        std::vector<Box> nb;
        for (const auto &x : cells)
          if (x.c0 <= t && t <= x.c1 && x.c0 == x.c1 && !x.text.empty()) nb.push_back(x.bx);
        if (boxes.empty() || nb.empty()) continue;
        double gap = kInf;
        for (const Box &b : boxes)
          for (const Box &nn : nb)
            gap = std::min(gap, std::max(std::max(b[0] - nn[2], nn[0] - b[2]), 0.0));
        if (gap < 1.5 * hh && (!have_best || gap < best_gap)) {
          have_best = true;
          best_gap = gap;
          best_t = t;
        }
      }
      if (have_best) {
        fold_col(cells, q, best_t);
        n_cols -= 1;
        changed = true;
        break;
      }
    }
  }
}

// A lone number (thousands separators, sign, decimal, percent): values don't wrap.
bool is_number(const std::string &text) {
  std::string_view v = trim(text);
  while (!v.empty() && v.back() == '%') v.remove_suffix(1);
  std::string t;
  for (char ch : v)
    if (ch != ',' && ch != ' ') t.push_back(ch);
  if (t.empty()) return false;
  const std::u32string u = utf32(t);
  if (!std::any_of(u.begin(), u.end(), is_ascii_digit)) return false;
  static constexpr std::u32string_view kOk = U"+-.()$€£";
  return std::all_of(u.begin(), u.end(), [](char32_t c) {
    return is_ascii_digit(c) || kOk.find(c) != std::u32string_view::npos;
  });
}

// Text opening a new list item ("1.2 ...", "(a) ...", "• ..."): a sub-row, never a wrapped line.
bool starts_item(const std::string &text) {
  const std::string_view t = trim_start(text);
  const std::vector<std::string> words = split_whitespace(t);
  if (words.empty()) return false;
  const std::string &first = words[0];
  const std::u32string fu = utf32(first);
  static constexpr std::u32string_view kBullets = U"•·▪◦‣-–—*●○■□";
  if (fu.size() == 1 && kBullets.find(fu[0]) != std::u32string_view::npos) return true;
  std::string_view core = first;
  while (!core.empty() && core.front() == '(') core.remove_prefix(1);
  while (!core.empty() && (core.back() == ')' || core.back() == '.' || core.back() == ':'))
    core.remove_suffix(1);
  bool dotted_digits = true;  // core.split('.').all(non-empty, all ASCII digits)
  {
    size_t start = 0;
    while (true) {
      const size_t dot = core.find('.', start);
      const std::string_view part =
          core.substr(start, dot == std::string_view::npos ? std::string_view::npos : dot - start);
      if (part.empty() ||
          !std::all_of(part.begin(), part.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        dotted_digits = false;
        break;
      }
      if (dot == std::string_view::npos) break;
      start = dot + 1;
    }
  }
  const std::u32string cu = utf32(core);
  const bool letter = cu.size() == 1 && is_ascii_alpha(cu[0]) && std::string_view(first) != core;
  const bool enumerator = !core.empty() && (dotted_digits || letter);
  return enumerator && std::string_view(first) != t;
}

// Non-empty-text cells lying in row r alone (no row span).
std::vector<size_t> filled_row(const std::vector<GridCell> &cells, int r) {
  std::vector<size_t> out;
  for (size_t k = 0; k < cells.size(); ++k)
    if (cells[k].r0 == r && cells[k].r1 == r && !trim(cells[k].text).empty()) out.push_back(k);
  return out;
}

// Drop row `r` after its text moved up: its (empty) cells go, spans through it shrink, rows
// below move up.
void remove_row(std::vector<GridCell> &cells, std::vector<int> &head, int &n_rows, int r) {
  std::erase_if(cells, [&](const GridCell &c) { return c.r0 == r && c.r1 == r; });
  for (auto &c : cells) {
    if (c.r0 > r) {
      c.r0 -= 1;
      c.r1 -= 1;
    } else if (c.r1 >= r) {
      c.r1 -= 1;
    }
  }
  std::erase(head, r);
  for (int &h : head)
    if (h > r) h -= 1;
  n_rows -= 1;
}

// Move cell k's text (trimmed) and box into cell j.
void merge_into(std::vector<GridCell> &cells, size_t k, size_t j) {
  const std::string t(trim(cells[k].text));
  const Box b = cells[k].bx;
  GridCell &c = cells[j];
  c.text = std::string(trim(c.text)) + " " + t;
  c.bx = {std::min(c.bx[0], b[0]), std::min(c.bx[1], b[1]), std::max(c.bx[2], b[2]),
          std::max(c.bx[3], b[3])};
}

// Rows that are a wrapped cell's continuation lines fold into the row above: a row with fewer
// filled cells than the one above, each under a filled cell, none a number, sitting clearly
// tighter (under 0.8x) than the table's own row pitch (measured between rows of 2+ filled cells,
// so a dense table with uniform pitch never folds).
void fold_wrapped_rows(std::vector<GridCell> &cells, std::vector<int> &head, int &n_rows) {
  auto top = [&](const std::vector<size_t> &ks) {
    double v = kInf;
    for (size_t k : ks) v = std::min(v, cells[k].bx[1]);
    return v;
  };
  std::vector<double> pitches;
  for (int r = 1; r < n_rows; ++r) {
    const auto a = filled_row(cells, r - 1), b = filled_row(cells, r);
    if (a.size() >= 2 && b.size() >= 2) pitches.push_back(top(b) - top(a));
  }
  if (pitches.size() < 2) return;
  const double pitch = median(pitches);
  int r = 1;
  while (r < n_rows) {
    const auto above = filled_row(cells, r - 1), here = filled_row(cells, r);
    const bool same_kind = contains(head, r) == contains(head, r - 1);
    std::vector<size_t> targets;
    const bool span_starts =
        std::any_of(cells.begin(), cells.end(), [&](const GridCell &c) { return c.r0 == r && c.r1 > r; });
    const bool ok =
        same_kind && !span_starts && !here.empty() && here.size() < above.size() &&
        std::all_of(here.begin(), here.end(), [&](size_t k) {
          const GridCell &c = cells[k];
          // the cell covering this column in the row above
          size_t t = kNone;
          for (size_t j = 0; j < cells.size(); ++j)
            if (cells[j].r0 < r && r - 1 <= cells[j].r1 && cells[j].c0 <= c.c0 &&
                c.c1 <= cells[j].c1) {
              t = j;
              break;
            }
          if (t == kNone || trim(cells[t].text).empty() || is_number(c.text) ||
              starts_item(c.text))
            return false;
          const bool fits = c.bx[1] - cells[t].bx[1] < 0.8 * pitch &&
                            c.bx[1] - cells[t].bx[3] < 0.5 * pitch;
          targets.push_back(t);
          return fits;
        });
    if (!ok) {
      ++r;
      continue;
    }
    for (size_t a = 0; a < here.size() && a < targets.size(); ++a)
      merge_into(cells, here[a], targets[a]);
    remove_row(cells, head, n_rows, r);
  }
}

// Whether `below` reads as the next line of the same phrase as `above`: it opens lowercase or
// with a bracket, or `above` breaks off mid-phrase (a hyphen, slash, comma, open bracket, `&`, or
// a connective). A list of names ("Buick Reatta" under "Alfa Romeo") gives no such cue.
bool continues(const std::string &above, const std::string &below) {
  const std::string_view a = trim(above), b = trim(below);
  const std::u32string bu = utf32(b);
  const bool opens = !bu.empty() && (is_lowercase(bu[0]) || bu[0] == U'(' || bu[0] == U'[' ||
                                     bu[0] == U'{');
  const bool open_brackets = std::count(a.begin(), a.end(), '(') > std::count(a.begin(), a.end(), ')');
  // the text after the last whitespace character, lowercased (only ASCII can match the list)
  const std::u32string au = utf32(a);
  size_t start = au.size();
  while (start > 0 && !is_whitespace(au[start - 1])) --start;
  std::u32string last = au.substr(start);
  for (char32_t &c : last)
    if (c >= U'A' && c <= U'Z') c += 32;
  static constexpr std::u32string_view kWords[] = {U"and", U"of",  U"the", U"for",
                                                   U"to",  U"in",  U"on",  U"or",
                                                   U"per", U"by",  U"with", U"from"};
  const bool connective = std::find(std::begin(kWords), std::end(kWords),
                                    std::u32string_view(last)) != std::end(kWords);
  const bool breaks = ends_with_any(a, "-/,(&") || open_brackets || connective;
  return opens || breaks;
}

// Header cells wrapped over several lines ("Product" / "(active substance," / "strains)") come
// out as one header row per line, each with as many filled cells as the line above, which
// fold_wrapped_rows (fewer cells, looser than the body pitch) leaves alone. In the header area
// (the leading rows without numbers, and any marked header rows; at most four), a row folds into
// the one above when every filled cell sits under a filled cell, none is a number, each starts
// within 0.6x its own line height of the cell above's bottom, and each reads as a continuation
// (continues): wrapped lines, not a second header level (that one adds cells under empty slots)
// nor a text-only table's next row.
void fold_header_wraps(std::vector<GridCell> &cells, std::vector<int> &head, int &n_rows) {
  // the header area: the leading rows without numbers (at most four), and any marked header rows
  // at the top
  auto zone = [&]() {
    int marked = 0;
    while (marked < n_rows && contains(head, marked)) ++marked;
    int z = 0;
    while (z < std::min(n_rows, 4)) {
      const auto f = filled_row(cells, z);
      if (f.empty() ||
          std::any_of(f.begin(), f.end(), [&](size_t k) { return is_number(cells[k].text); }))
        break;
      ++z;
    }
    return std::min(std::max(z, marked), 4);
  };
  int r = 1;
  while (r < zone()) {
    const auto here = filled_row(cells, r);
    const bool span_starts =
        std::any_of(cells.begin(), cells.end(), [&](const GridCell &c) { return c.r0 == r && c.r1 > r; });
    std::vector<std::pair<size_t, size_t>> pairs;
    // one lone filled cell in a wider row is a body cell's wrapped tail more often than a header
    // line (a header wraps in several columns at once)
    int n_cols = 0;
    for (const auto &c : cells) n_cols = std::max(n_cols, c.c1 + 1);
    const bool ok =
        (here.size() >= 2 || n_cols <= 2) && !span_starts &&
        std::all_of(here.begin(), here.end(), [&](size_t k) {
          const GridCell &c = cells[k];
          size_t above = kNone;
          for (size_t j = 0; j < cells.size(); ++j)
            if (cells[j].r0 < r && r - 1 <= cells[j].r1 && cells[j].c0 <= c.c0 &&
                c.c1 <= cells[j].c1 && !trim(cells[j].text).empty()) {
              above = j;
              break;
            }
          if (above == kNone || is_number(c.text) || starts_item(c.text)) return false;
          const double line_h = std::max(c.bx[3] - c.bx[1], 1e-6);
          pairs.emplace_back(k, above);
          return c.bx[1] - cells[above].bx[3] < 0.6 * line_h &&
                 continues(cells[above].text, c.text);
        });
    if (!ok) {
      ++r;
      continue;
    }
    for (auto [k, j] : pairs) merge_into(cells, k, j);
    remove_row(cells, head, n_rows, r);
  }
}

// Python `html.escape(text, quote=True)`.
std::string escape(const std::string &t) {
  std::string s;
  s.reserve(t.size());
  for (char ch : t) {
    switch (ch) {
      case '&': s += "&amp;"; break;
      case '<': s += "&lt;"; break;
      case '>': s += "&gt;"; break;
      case '"': s += "&quot;"; break;
      case '\'': s += "&#x27;"; break;
      default: s.push_back(ch);
    }
  }
  return s;
}

// Rust `str::parse::<f64>` on text already restricted to ASCII digits, '.', '-' and '+': an
// optional sign, then digits with at most one '.', at least one digit.
bool parse_f64(std::string_view t, double &out) {
  size_t i = 0;
  if (i < t.size() && (t[i] == '+' || t[i] == '-')) ++i;
  size_t digits = 0, dots = 0;
  for (size_t k = i; k < t.size(); ++k) {
    if (t[k] >= '0' && t[k] <= '9') ++digits;
    else if (t[k] == '.') ++dots;
    else return false;
  }
  if (digits == 0 || dots > 1) return false;
  out = std::strtod(std::string(t).c_str(), nullptr);
  return true;
}

}  // namespace

// Units running against the page's main direction take part in the forward pass but are never
// cells: their logits are set to -9 (as tablelab.run_model).
void mask_minor(Logits &l, const std::vector<bool> &minor) {
  const size_t n = static_cast<size_t>(l.n);
  for (size_t i = 0; i < minor.size(); ++i) {
    if (!minor[i]) continue;
    l.node[i * 2] = -9.0f;
    l.node[i * 2 + 1] = -9.0f;
    for (size_t k = 0; k < 4; ++k)
      for (size_t j = 0; j < n; ++j) {
        l.pair[(k * n + i) * n + j] = -9.0f;
        l.pair[(k * n + j) * n + i] = -9.0f;
      }
  }
}

// Average-linkage clusters of a symmetric n x n matrix: merge while the mean logit > thresh.
// Groups come back sorted, in the order of their surviving representative.
//
// Each merge joins the pair with the highest mean (ties: the first in row-major order of the live
// groups) and adds the absorbed group's sums into the survivor's row and column. Every live group
// caches its best partner, so a merge rescans only the rows whose cached partner it touched:
// O(n^2) on typical pages instead of rescanning every pair per merge.
std::vector<std::vector<int>> avg_link(const std::vector<double> &p, int n_, double thresh) {
  const size_t n = static_cast<size_t>(n_);
  std::vector<std::vector<int>> groups(n);
  for (size_t i = 0; i < n; ++i) groups[i] = {static_cast<int>(i)};
  std::vector<double> s = p;
  std::vector<double> cnt(n * n, 1.0);
  std::vector<size_t> alive(n);
  std::iota(alive.begin(), alive.end(), size_t{0});
  // best partner of each live group g: (mean, partner), the first maximum in `alive` order
  auto row_best = [&](size_t g) {
    std::pair<double, size_t> best{-kInf, kNone};
    for (size_t h : alive) {
      if (h == g) continue;
      const double v = s[g * n + h] / cnt[g * n + h];
      if (v > best.first) best = {v, h};
    }
    return best;
  };
  std::vector<std::pair<double, size_t>> best(n);
  for (size_t g = 0; g < n; ++g) best[g] = row_best(g);
  while (alive.size() > 1) {
    // the first live group (in `alive` order) holding the overall maximum
    size_t ba = 0;
    for (size_t a = 1; a < alive.size(); ++a)
      if (best[alive[a]].first > best[alive[ba]].first) ba = a;
    const auto [v, gb] = best[alive[ba]];
    if (v <= thresh) break;
    const size_t ga = alive[ba];
    const size_t bb = static_cast<size_t>(std::find(alive.begin(), alive.end(), gb) - alive.begin());
    groups[ga].insert(groups[ga].end(), groups[gb].begin(), groups[gb].end());
    groups[gb].clear();
    for (size_t k = 0; k < n; ++k) {
      s[ga * n + k] += s[gb * n + k];
      cnt[ga * n + k] += cnt[gb * n + k];
    }
    for (size_t k = 0; k < n; ++k) {
      s[k * n + ga] += s[k * n + gb];
      cnt[k * n + ga] += cnt[k * n + gb];
    }
    alive.erase(alive.begin() + static_cast<std::ptrdiff_t>(bb));
    // Only row ga and column ga changed (and gb is gone): a row whose partner was ga or gb is
    // rescanned; any other row compares its old best with its new entry for ga.
    best[ga] = row_best(ga);
    for (size_t r : alive) {
      if (r == ga) continue;
      const auto [bv, bc] = best[r];
      if (bc == ga || bc == gb) {
        best[r] = row_best(r);
      } else {
        const double c = s[r * n + ga] / cnt[r * n + ga];
        if (c > bv || (c == bv && ga < bc)) best[r] = {c, ga};
      }
    }
  }
  std::vector<std::vector<int>> out;
  out.reserve(alive.size());
  for (size_t g : alive) {
    std::vector<int> v = groups[g];
    std::sort(v.begin(), v.end());
    out.push_back(std::move(v));
  }
  return out;
}

// Tables as lists of unit indices (tn_core.decode_tables, absorb=True).
std::vector<std::vector<int>> decode_tables(const std::vector<DUnit> &units, const Logits &l,
                                            const DecodeOptions &opt) {
  const int n = static_cast<int>(units.size());
  std::vector<int> on;
  for (int i = 0; i < n; ++i)
    if (l.node_at(i, 0) > opt.table_thresh) on.push_back(i);
  if (on.size() < 2) return {};
  const size_t m = on.size();
  std::vector<double> p(m * m, 0.0);
  for (size_t a = 0; a < m; ++a)
    for (size_t b = 0; b < m; ++b) p[a * m + b] = static_cast<double>(l.pair_at(0, on[a], on[b]));
  std::vector<std::vector<int>> groups;
  for (auto &g : avg_link(p, static_cast<int>(m), 0.0)) {
    if (g.size() < 2) continue;
    for (int &i : g) i = on[i];
    groups.push_back(std::move(g));
  }
  // stacked pieces that line up across and sit within 3 lines merge back
  if (groups.size() > 1) {
    const double hm = median_h_all(units);
    while (true) {
      std::vector<Box> ext;
      for (const auto &g : groups) ext.push_back(bbox_of(units, g));
      size_t ma = kNone, mb = kNone;
      for (size_t a = 0; a < groups.size() && ma == kNone; ++a)
        for (size_t b = 0; b < groups.size(); ++b) {
          if (a == b) continue;
          const Box &aa = ext[a], &bb = ext[b];
          const double gap = bb[1] - aa[3];
          const double ov = std::min(aa[2], bb[2]) - std::max(aa[0], bb[0]);
          if (gap >= 0.0 && gap < 3.0 * 1.4 * hm &&
              ov > 0.7 * std::min(aa[2] - aa[0], bb[2] - bb[0])) {
            ma = a;
            mb = b;
            break;
          }
        }
      if (ma == kNone) break;
      const std::vector<int> gb = groups[mb];
      groups[ma].insert(groups[ma].end(), gb.begin(), gb.end());
      std::sort(groups[ma].begin(), groups[ma].end());
      groups.erase(groups.begin() + static_cast<std::ptrdiff_t>(mb));
    }
  }
  // a small group lying mostly inside a bigger table's box (grown by a line) joins it
  if (groups.size() > 1) {
    const double hm = median_h_all(units);
    while (true) {
      std::stable_sort(groups.begin(), groups.end(),
                       [](const auto &a, const auto &b) { return b.size() < a.size(); });
      std::vector<Box> ext;
      for (const auto &g : groups) {
        const Box b = bbox_of(units, g);
        ext.push_back({b[0] - 1.4 * hm, b[1] - 1.4 * hm, b[2] + 1.4 * hm, b[3] + 1.4 * hm});
      }
      size_t ms = kNone, mb = kNone;
      for (size_t s = groups.size() - 1; s >= 1 && ms == kNone; --s) {
        const Box sb = bbox_of(units, groups[s]);
        const double area = std::max((sb[2] - sb[0]) * (sb[3] - sb[1]), 1e-6);
        for (size_t b = 0; b < s; ++b) {
          if (static_cast<double>(groups[s].size()) > 0.5 * static_cast<double>(groups[b].size()))
            continue;
          const Box &e = ext[b];
          const double ov = std::max(std::min(sb[2], e[2]) - std::max(sb[0], e[0]), 0.0) *
                            std::max(std::min(sb[3], e[3]) - std::max(sb[1], e[1]), 0.0);
          if (ov >= 0.5 * area) {
            ms = s;
            mb = b;
            break;
          }
        }
      }
      if (ms == kNone) break;
      const std::vector<int> gs = groups[ms];
      groups[mb].insert(groups[mb].end(), gs.begin(), gs.end());
      std::sort(groups[mb].begin(), groups[mb].end());
      groups.erase(groups.begin() + static_cast<std::ptrdiff_t>(ms));
    }
  }
  // stray units whose centre lies inside a table's box join it
  if (!groups.empty()) {
    std::vector<bool> taken(n, false);
    for (const auto &g : groups)
      for (int i : g) taken[i] = true;
    for (auto &g : groups) {
      const Box b = bbox_of(units, g);
      for (int i = 0; i < n; ++i) {
        if (taken[i] || l.node_at(i, 0) < -8.0f) continue;
        const Box &u = units[i].bx;
        const double cx = (u[0] + u[2]) / 2.0, cy = (u[1] + u[3]) / 2.0;
        if (b[0] <= cx && cx <= b[2] && b[1] <= cy && cy <= b[3]) {
          g.push_back(i);
          taken[i] = true;
        }
      }
      std::sort(g.begin(), g.end());
    }
    if (opt.head_band) head_band(units, l, groups, taken);
  }
  return groups;
}

// One table's grid from its units (global indices `idx`) and the row / col / cell heads.
Grid decode_grid(const std::vector<DUnit> &units, const std::vector<int> &idx, const Logits &l,
                 const DecodeOptions &opt) {
  const size_t m = idx.size();
  if (m == 0) return {};
  std::vector<std::array<float, 4>> bxs;
  bxs.reserve(m);
  for (int i : idx) {
    const Box &b = units[i].bx;
    bxs.push_back({static_cast<float>(b[0]), static_cast<float>(b[1]), static_cast<float>(b[2]),
                   static_cast<float>(b[3])});
  }
  std::vector<float> hh;
  hh.reserve(m);
  for (const auto &b : bxs) hh.push_back(std::max(b[3] - b[1], 1e-3f));
  auto pr = [&](int k, size_t a, size_t b) { return l.pair_at(k, idx[a], idx[b]); };
  // cell logits: same_cell, or for stacked neighbours the weaker of same_row / same_col - margin
  std::vector<double> cell_p(m * m, 0.0);
  std::vector<bool> forced(m * m, false);
  for (size_t a = 0; a < m; ++a)
    for (size_t b = 0; b < m; ++b) {
      const float vgap = std::max(bxs[b][1] - bxs[a][3], bxs[a][1] - bxs[b][3]);
      const float xov = std::min(bxs[a][2], bxs[b][2]) - std::max(bxs[a][0], bxs[b][0]);
      const bool both_hdr = l.node_at(idx[a], 1) > 0.0f && l.node_at(idx[b], 1) > 0.0f;
      const double lim =
          (both_hdr ? kStackGapHdr : kStackGap) * static_cast<double>(std::min(hh[a], hh[b]));
      const bool stacked = static_cast<double>(vgap) < lim && vgap > -0.5f * hh[a] && xov > 0.0f;
      const float f = stacked ? std::min(pr(1, a, b), pr(2, a, b)) - kCellMargin
                              : -std::numeric_limits<float>::infinity();
      forced[a * m + b] = f > 0.0f;
      cell_p[a * m + b] = static_cast<double>(std::max(pr(3, a, b), f));
    }
  std::vector<std::vector<int>> cells = avg_link(cell_p, static_cast<int>(m), 0.0);
  if (opt.forced_link && cells.size() > 1) {
    std::vector<size_t> of(m, 0);
    for (size_t k = 0; k < cells.size(); ++k)
      for (int i : cells[k]) of[i] = k;
    std::vector<size_t> parent(cells.size());
    std::iota(parent.begin(), parent.end(), size_t{0});
    auto root = [&](size_t k) {
      while (parent[k] != k) {
        parent[k] = parent[parent[k]];
        k = parent[k];
      }
      return k;
    };
    for (size_t a = 0; a < m; ++a)
      for (size_t b = 0; b < m; ++b)
        if (forced[a * m + b]) {
          const size_t ra = root(of[a]);
          const size_t rb = root(of[b]);
          if (ra != rb) parent[ra] = rb;
        }
    std::vector<size_t> keys;
    std::vector<std::vector<int>> merged;
    for (size_t k = 0; k < cells.size(); ++k) {
      const size_t r = root(k);
      const auto it = std::find(keys.begin(), keys.end(), r);
      if (it != keys.end()) {
        auto &dst = merged[static_cast<size_t>(it - keys.begin())];
        dst.insert(dst.end(), cells[k].begin(), cells[k].end());
      } else {
        keys.push_back(r);
        merged.push_back(cells[k]);
      }
    }
    for (auto &g : merged) std::sort(g.begin(), g.end());
    cells = std::move(merged);
  }
  const size_t nc = cells.size();
  std::vector<Box> cb;
  cb.reserve(nc);
  for (const auto &g : cells) {
    Box b{kInf, kInf, -kInf, -kInf};
    for (int i : g) {
      const Box &u = units[idx[i]].bx;
      b[0] = std::min(b[0], u[0]);
      b[1] = std::min(b[1], u[1]);
      b[2] = std::max(b[2], u[2]);
      b[3] = std::max(b[3], u[3]);
    }
    cb.push_back(b);
  }
  // cell-pair logits on an axis: the mean of their unit pairs, minus GATE without overlap
  auto cell_logits = [&](int k, int lo, int hi) {
    std::vector<double> out(nc * nc, 0.0);
    for (size_t a = 0; a < nc; ++a)
      for (size_t b = a + 1; b < nc; ++b) {
        double s = 0.0;
        for (int i : cells[a])
          for (int j : cells[b]) s += static_cast<double>(pr(k, i, j));
        double v = s / static_cast<double>(cells[a].size() * cells[b].size());
        if (std::min(cb[a][hi], cb[b][hi]) - std::max(cb[a][lo], cb[b][lo]) <= 0.0) v -= kGate;
        out[a * nc + b] = v;
        out[b * nc + a] = v;
      }
    return out;
  };
  std::vector<std::vector<int>> rows, cols;
  if (nc > 1) {
    rows = span_link(cell_logits(1, 1, 3), nc);
    cols = span_link(cell_logits(2, 0, 2), nc);
  } else {
    rows = {{0}};
    cols = {{0}};
  }
  auto mean_of = [&](const std::vector<int> &g, int lo, int hi) {
    double s = 0.0;
    for (int k : g) s += (cb[k][lo] + cb[k][hi]) / 2.0;
    return s / static_cast<double>(g.size());
  };
  std::stable_sort(rows.begin(), rows.end(), [&](const auto &a, const auto &b) {
    return mean_of(a, 1, 3) < mean_of(b, 1, 3);
  });
  std::stable_sort(cols.begin(), cols.end(), [&](const auto &a, const auto &b) {
    return mean_of(a, 0, 2) < mean_of(b, 0, 2);
  });
  std::vector<std::vector<int>> r_in(nc), c_in(nc);
  for (size_t k = 0; k < rows.size(); ++k)
    for (int c : rows[k]) r_in[c].push_back(static_cast<int>(k));
  for (size_t k = 0; k < cols.size(); ++k)
    for (int c : cols[k]) c_in[c].push_back(static_cast<int>(k));
  const int n_rows0 = static_cast<int>(rows.size());
  const int n_cols0 = static_cast<int>(cols.size());
  std::vector<GridCell> placed;
  std::vector<size_t> owner(static_cast<size_t>(n_rows0) * n_cols0, kNone);  // slot -> placed
  auto slot = [&](int r, int q) { return static_cast<size_t>(r) * n_cols0 + q; };
  auto rmin = [&](size_t c) { return *std::min_element(r_in[c].begin(), r_in[c].end()); };
  auto cmin = [&](size_t c) { return *std::min_element(c_in[c].begin(), c_in[c].end()); };
  std::vector<size_t> order(nc);
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return std::make_pair(rmin(a), cmin(a)) < std::make_pair(rmin(b), cmin(b));
  });
  for (size_t c : order) {
    int r0 = rmin(c), r1 = *std::max_element(r_in[c].begin(), r_in[c].end());
    int c0 = cmin(c), c1 = *std::max_element(c_in[c].begin(), c_in[c].end());
    std::vector<Box> boxes;
    for (int i : cells[c]) boxes.push_back(units[idx[i]].bx);
    std::string text;
    bool first = true;
    for (size_t k : reading_order(boxes)) {
      if (!first) text += ' ';
      first = false;
      text += units[idx[cells[c][k]]].text;
    }
    bool clash = false;
    int fr = -1, fq = -1;  // the first free slot of the span
    for (int r = r0; r <= r1; ++r)
      for (int q = c0; q <= c1; ++q) {
        if (owner[slot(r, q)] != kNone) clash = true;
        else if (fr < 0) fr = r, fq = q;
      }
    if (clash) {
      if (fr >= 0) {
        r0 = r1 = fr;
        c0 = c1 = fq;
      } else {
        const size_t o = owner[slot(r0, c0)];
        placed[o].text = placed[o].text + " " + text;
        continue;
      }
    }
    const size_t pi = placed.size();
    placed.push_back({r0, c0, r1, c1, std::move(text), cb[c]});
    for (int r = r0; r <= r1; ++r)
      for (int q = c0; q <= c1; ++q) owner[slot(r, q)] = pi;
  }
  std::vector<int> head_rows;
  for (size_t k = 0; k < rows.size(); ++k) {
    size_t total = 0, pos = 0;
    for (int c : rows[k])
      for (int i : cells[c]) {
        ++total;
        if (l.node_at(idx[i], 1) > 0.0f) ++pos;
      }
    if (static_cast<double>(pos) / static_cast<double>(total) > 0.5)
      head_rows.push_back(static_cast<int>(k));
  }
  int n_cols = n_cols0;
  if (opt.squeeze) squeeze_cols(placed, head_rows, n_cols, median_h_of(units, idx));
  int n_rows = n_rows0;
  if (opt.fold_wraps) {
    fold_wrapped_rows(placed, head_rows, n_rows);
    fold_header_wraps(placed, head_rows, n_rows);
  }
  Grid g;
  g.cells = std::move(placed);
  g.head_rows = std::move(head_rows);
  g.n_rows = n_rows;
  g.n_cols = n_cols;
  return g;
}

// The grid as HTML, exactly as tn_core.decode_grid writes it.
std::string Grid::to_html() const {
  std::map<std::pair<int, int>, size_t> at;  // top-left slot -> cell (the last one wins)
  std::set<std::pair<int, int>> taken;
  for (size_t k = 0; k < cells.size(); ++k) {
    const GridCell &c = cells[k];
    at[{c.r0, c.c0}] = k;
    for (int r = c.r0; r <= c.r1; ++r)
      for (int q = c.c0; q <= c.c1; ++q) taken.insert({r, q});
  }
  std::string out = "<table>";
  for (int r = 0; r < n_rows; ++r) {
    const std::string tag = contains(head_rows, r) ? "th" : "td";
    std::string tds;
    int q = 0;
    while (q < n_cols) {
      const auto it = at.find({r, q});
      if (it != at.end()) {
        const GridCell &c = cells[it->second];
        std::string attr;
        if (c.r1 > r) attr += " rowspan=\"" + std::to_string(c.r1 - r + 1) + "\"";
        if (c.c1 > q) attr += " colspan=\"" + std::to_string(c.c1 - q + 1) + "\"";
        tds += "<" + tag + attr + ">" + escape(c.text) + "</" + tag + ">";
        q = c.c1 + 1;
      } else if (taken.count({r, q})) {
        ++q;
      } else {
        tds += "<" + tag + "></" + tag + ">";
        ++q;
      }
    }
    out += "\n<tr>" + tds + "</tr>";
  }
  out += "\n</table>";
  return out;
}

// A table of contents: at most three columns, and most rows end in a page number (arabic or
// roman) with the arabic ones non-decreasing down the table. Rendered as text, not a table.
bool Grid::is_toc() const {
  if (n_cols > 3) return false;
  std::vector<std::vector<std::pair<int, std::string_view>>> rows(static_cast<size_t>(n_rows));
  for (const auto &c : cells) {
    const std::string_view t = trim(c.text);
    if (!t.empty()) rows[c.r0].emplace_back(c.c0, t);
  }
  std::erase_if(rows, [](const auto &r) { return r.empty(); });
  size_t hits = 0;
  std::vector<unsigned> arabic;
  for (const auto &r : rows) {
    // max_by_key: the last of the rightmost
    size_t li = 0;
    for (size_t k = 1; k < r.size(); ++k)
      if (r[k].first >= r[li].first) li = k;
    const std::string_view last = r[li].second;
    const std::u32string lu = utf32(last);
    static constexpr std::u32string_view kRoman = U"ivxlcdmIVXLCDM";
    const bool roman = last.size() <= 7 && std::all_of(lu.begin(), lu.end(), [](char32_t ch) {
                         return kRoman.find(ch) != std::u32string_view::npos;
                       });
    const bool num = last.size() <= 4 && std::all_of(lu.begin(), lu.end(), is_ascii_digit);
    if (r.size() >= 2 && (roman || num)) {
      ++hits;
      if (num) {
        // `parse::<u32>().unwrap_or(0)`: at most four ASCII digits, so it parses
        unsigned v = 0;
        for (char ch : last) v = v * 10 + static_cast<unsigned>(ch - '0');
        arabic.push_back(v);
      }
    }
  }
  size_t falls = 0;
  for (size_t k = 1; k < arabic.size(); ++k)
    if (arabic[k] < arabic[k - 1]) ++falls;
  const bool rising = falls <= arabic.size() / 10;
  return hits >= 3 && hits * 5 >= rows.size() * 3 && rising;
}

// Multi-column prose read as a table: long, mostly non-numeric cells whose text runs on down each
// column, a cell that doesn't end a sentence followed by one that starts lowercase ("calendar
// 2024, more than 70%" / "will be from recycled content"). A real table's cells are independent,
// so its columns rarely flow even when one column is long.
bool Grid::is_prose() const {
  auto words = [](const std::string &t) { return split_whitespace(t).size(); };
  std::vector<const GridCell *> cs;
  for (const auto &c : cells)
    if (!trim(c.text).empty()) cs.push_back(&c);
  if (cs.size() < 6) return false;
  std::vector<size_t> wc;
  for (const auto *c : cs) wc.push_back(words(c->text));
  std::sort(wc.begin(), wc.end());
  if (wc[wc.size() / 2] < 4) return false;
  const size_t numeric = static_cast<size_t>(std::count_if(cs.begin(), cs.end(), [&](const GridCell *c) {
    return words(c->text) <= 3 &&
           std::any_of(c->text.begin(), c->text.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
  }));
  if (numeric * 10 >= cs.size() * 3) return false;
  std::map<int, std::vector<const GridCell *>> cols;
  for (const auto *c : cs)
    if (c->c0 == c->c1) cols[c->c0].push_back(c);
  size_t pairs = 0, flow = 0;
  for (auto &[q, col] : cols) {
    std::stable_sort(col.begin(), col.end(),
                     [](const GridCell *a, const GridCell *b) { return a->r0 < b->r0; });
    for (size_t k = 1; k < col.size(); ++k) {
      const GridCell *w0 = col[k - 1], *w1 = col[k];
      if (w1->r0 > w0->r1 + 2) continue;  // not adjacent: more than one empty row between them
      ++pairs;
      const std::string_view up = trim_end(w0->text);
      const bool open = !ends_with_any(up, ".!?:;");
      const std::u32string down = utf32(trim_start(w1->text));
      const bool lower = !down.empty() && is_lowercase(down[0]);
      flow += (open && lower) ? 1 : 0;
    }
  }
  return pairs >= 3 && flow * 10 >= pairs * 4;
}

// A chart read as a table: a sparse, mostly numeric grid with an axis - some column holding at
// least three lone numbers in an evenly stepped, monotonic run (the tick labels).
bool Grid::is_chart() const {
  const long slots = static_cast<long>(n_rows) * n_cols;
  if (slots == 0) return false;
  auto parse = [](const std::string &text, double &v) {
    std::string_view s = trim(text);
    while (!s.empty() && s.back() == '%') s.remove_suffix(1);
    std::string t;
    for (char ch : s)
      if (ch != ',') t.push_back(ch);
    if (t.empty() || !std::all_of(t.begin(), t.end(), [](char ch) {
          return (ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == '+';
        }))
      return false;
    return parse_f64(t, v);
  };
  long filled = 0, nonempty = 0, numeric = 0;
  std::map<int, std::vector<std::pair<int, double>>> cols;
  std::vector<std::pair<std::pair<int, int>, double>> flow;
  for (const auto &c : cells) {
    if (trim(c.text).empty()) continue;
    ++nonempty;
    filled += static_cast<long>(c.r1 - c.r0 + 1) * (c.c1 - c.c0 + 1);
    double v;
    if (parse(c.text, v)) {
      ++numeric;
      if (c.c0 == c.c1) cols[c.c0].emplace_back(c.r0, v);
      flow.push_back({{c.r0, c.c0}, v});
    }
  }
  if (filled * 2 >= slots || numeric * 2 < nonempty) return false;
  auto stepped = [](const std::vector<double> &vs) {
    if (vs.size() < 3) return false;
    std::vector<double> d;
    for (size_t k = 1; k < vs.size(); ++k) d.push_back(vs[k] - vs[k - 1]);
    return d[0] != 0.0 && std::all_of(d.begin(), d.end(), [&](double x) {
             return std::abs(x - d[0]) <= 1e-6 * std::max(std::abs(d[0]), 1.0);
           });
  };
  // a column of ticks (y axis), or every number on the grid in reading order (an x axis of years
  // strewn diagonally, one per row)
  std::stable_sort(flow.begin(), flow.end(),
                   [](const auto &a, const auto &b) { return a.first < b.first; });
  std::vector<double> all;
  for (const auto &f : flow) all.push_back(f.second);
  if (stepped(all)) return true;
  for (auto &[q, vs] : cols) {
    std::stable_sort(vs.begin(), vs.end(),
                     [](const auto &a, const auto &b) { return a.first < b.first; });
    std::vector<double> col;
    for (const auto &e : vs) col.push_back(e.second);
    if (stepped(col)) return true;
  }
  return false;
}

}  // namespace hp::tn
