#include "tablenet/units.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

#include <fpdf_edit.h>
#include <fpdf_text.h>
#include <fpdfview.h>

#include "tablenet/text.h"

namespace hp::tn {
namespace {

constexpr double kCellGap = 1.8;   // CELL_GAP glyph heights
constexpr size_t kLeaderMin = 3;   // LEADER_MIN
constexpr double kNumGap = 0.5;    // NUM_GAP
constexpr double kAdaptX = 2.5;    // ADAPT_X
constexpr double kAdaptMin = 0.6;  // ADAPT_MIN
constexpr double kInf = std::numeric_limits<double>::infinity();

using Box = std::array<double, 4>;

// One character: code unit(s) (UTF-16 halves stay separate, as PDFium reports them), page box,
// font size, bold (-1 unknown).
struct Ch {
  std::vector<uint32_t> c;
  Box pb;
  double size;
  int8_t bold;
  [[nodiscard]] bool is_space() const {
    return c.size() == 1 && c[0] <= 0x10FFFF && !(c[0] >= 0xD800 && c[0] <= 0xDFFF) && is_whitespace(c[0]);
  }
};
using FCh = std::pair<Ch, Box>;  // char + reading-frame box

Box rotate_page(const Box &b, int rot, double w0, double h0) {
  const auto [x0, y0, x1, y1] = b;
  switch (rot) {
    case 1: return {h0 - y1, x0, h0 - y0, x1};
    case 2: return {w0 - x1, h0 - y1, w0 - x0, h0 - y0};
    case 3: return {y0, w0 - x1, y1, w0 - x0};
    default: return b;
  }
}

// Rust median over partial_cmp-sorted values (callers never pass an empty vector).
double median(std::vector<double> v) {
  std::stable_sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// Code units -> UTF-8, pairing UTF-16 surrogate halves and dropping lone ones.
std::string decode(const std::vector<uint32_t> &units) {
  std::string s;
  std::vector<uint16_t> pending;
  auto flush = [&] {
    for (size_t i = 0; i < pending.size(); ++i) {
      const uint16_t a = pending[i];
      if (a >= 0xD800 && a <= 0xDBFF && i + 1 < pending.size() && pending[i + 1] >= 0xDC00 && pending[i + 1] <= 0xDFFF) {
        append_utf8(s, 0x10000 + ((char32_t(a) - 0xD800) << 10) + (pending[i + 1] - 0xDC00));
        ++i;
      }  // a lone half is dropped (char::decode_utf16(..).flatten())
    }
    pending.clear();
  };
  for (uint32_t u : units) {
    if (u >= 0xD800 && u <= 0xDFFF) {
      pending.push_back(static_cast<uint16_t>(u));
    } else {
      flush();
      if (u <= 0x10FFFF) append_utf8(s, u);
    }
  }
  flush();
  return s;
}

bool is_dec(char32_t c) { return is_ascii_digit(c); }  // Python \d as the Rust port reads it

// units.py _NUM: ^[-–−]?\$?\d[\d,.]*%?(?:[-–−]\d[\d,.]*%?)?$
bool is_num(const std::string &t) {
  const std::u32string cs = utf32(t);
  size_t i = 0;
  auto dash = [](char32_t c) { return c == U'-' || c == U'–' || c == U'−'; };
  if (i < cs.size() && dash(cs[i])) ++i;
  if (i < cs.size() && cs[i] == U'$') ++i;
  auto number = [&](size_t &k) {
    if (k >= cs.size() || !is_dec(cs[k])) return false;
    ++k;
    while (k < cs.size() && (is_dec(cs[k]) || cs[k] == U',' || cs[k] == U'.')) ++k;
    if (k < cs.size() && cs[k] == U'%') ++k;
    return true;
  };
  if (!number(i)) return false;
  if (i == cs.size()) return true;
  if (dash(cs[i])) {
    ++i;
    if (number(i) && i == cs.size()) return true;
  }
  return false;
}

bool is_leader_char(const Ch &c) {
  if (c.c.size() != 1) return false;
  const uint32_t u = c.c[0];
  return u == U'.' || u == U'·' || u == U'…' || u == U'•' || u == 0xFFFD || u == U'_';
}

// True for characters in a dot-leader run (kLeaderMin+ leader characters, spaces between).
std::vector<bool> leader_mask(const std::vector<FCh> &chars) {
  std::vector<bool> mask(chars.size(), false);
  size_t i = 0;
  while (i < chars.size()) {
    if (is_leader_char(chars[i].first)) {
      size_t j = i, k = 0;
      while (j < chars.size() && (is_leader_char(chars[j].first) || chars[j].first.is_space())) {
        k += is_leader_char(chars[j].first);
        ++j;
      }
      if (k >= kLeaderMin)
        for (size_t m = i; m < j; ++m) mask[m] = true;
      i = j;
    } else {
      ++i;
    }
  }
  return mask;
}

// The gap above which a space in this run is a column gutter (nullopt: too few word gaps).
std::optional<double> gutter_gap(const std::vector<FCh> &chars, double h) {
  std::vector<double> gaps;
  std::optional<double> prev;
  bool spaced = false;
  for (const auto &[c, b] : chars) {
    if (c.is_space()) {
      spaced = true;
      continue;
    }
    if (prev && spaced) gaps.push_back(b[0] - *prev);
    prev = b[2];
    spaced = false;
  }
  if (gaps.size() < 3) return std::nullopt;
  return std::max(kAdaptX * median(gaps), kAdaptMin * h);
}

// Split one run (chars with reading-frame boxes) into units.
std::vector<Unit> cut(const std::vector<FCh> &chars, double h) {
  const std::vector<bool> lead = leader_mask(chars);
  const std::optional<double> gut = gutter_gap(chars, h);
  std::vector<std::vector<size_t>> groups;
  std::vector<size_t> cur;
  std::optional<double> prev;
  std::string word, last_word;
  bool spaced = false;
  for (size_t k = 0; k < chars.size(); ++k) {
    const auto &[c, b] = chars[k];
    if (lead[k]) {
      if (!cur.empty()) {
        groups.push_back(std::move(cur));
        cur.clear();
        prev.reset();
        word.clear();
        last_word.clear();
        spaced = false;
      }
      continue;
    }
    if (c.is_space()) {
      if (!cur.empty()) {
        cur.push_back(k);
        if (!word.empty()) last_word = std::exchange(word, std::string());
        spaced = true;
      }
      continue;
    }
    bool is_cut = prev && b[0] - *prev > kCellGap * h;
    if (!is_cut && spaced && gut && prev && b[0] - *prev > *gut) is_cut = true;
    if (!is_cut && spaced && prev && b[0] - *prev > kNumGap * h && is_num(last_word)) {
      std::string nxt = decode(c.c);
      for (size_t x = k + 1; x < chars.size() && x < k + 1 + 23; ++x) {
        if (chars[x].first.is_space()) break;
        nxt += decode(chars[x].first.c);
      }
      is_cut = is_num(nxt);
    }
    if (is_cut) {
      groups.push_back(std::move(cur));
      cur.clear();
      last_word.clear();
    }
    cur.push_back(k);
    word += decode(c.c);
    spaced = false;
    prev = b[2];
  }
  if (!cur.empty()) groups.push_back(std::move(cur));

  std::vector<Unit> out;
  for (const auto &g : groups) {
    std::vector<size_t> ink;
    for (size_t k : g)
      if (!chars[k].first.is_space()) ink.push_back(k);
    if (ink.empty()) continue;
    std::vector<uint32_t> raw;
    for (size_t k : g) raw.insert(raw.end(), chars[k].first.c.begin(), chars[k].first.c.end());
    std::string text;
    for (const auto &piece : split_whitespace(decode(raw))) {
      if (!text.empty()) text += ' ';
      text += piece;
    }
    if (text.empty()) continue;
    Box bx{kInf, kInf, -kInf, -kInf};
    std::vector<double> hs, sizes;
    size_t nbold = 0, nknown = 0;
    for (size_t k : ink) {
      const Box &b = chars[k].second;
      bx[0] = std::min(bx[0], b[0]);
      bx[1] = std::min(bx[1], b[1]);
      bx[2] = std::max(bx[2], b[2]);
      bx[3] = std::max(bx[3], b[3]);
      hs.push_back(b[3] - b[1]);
      sizes.push_back(chars[k].first.size);
      if (chars[k].first.bold >= 0) {
        ++nknown;
        nbold += chars[k].first.bold == 1;
      }
    }
    Unit u;
    u.bx = u.pbox = bx;
    u.text = std::move(text);
    u.h = median(hs);
    u.size = median(sizes);
    u.bold = nknown == 0 ? -1 : (static_cast<double>(nbold) > static_cast<double>(nknown) / 2.0 ? 1 : 0);
    std::vector<size_t> w;
    auto flush_word = [&] {
      if (w.empty()) return;
      double x0 = kInf, x1 = -kInf;
      std::vector<uint32_t> wr;
      for (size_t k : w) {
        x0 = std::min(x0, chars[k].second[0]);
        x1 = std::max(x1, chars[k].second[2]);
        wr.insert(wr.end(), chars[k].first.c.begin(), chars[k].first.c.end());
      }
      u.words.push_back({x0, x1, decode(wr)});
      w.clear();
    };
    for (size_t k : g) {
      if (!chars[k].first.is_space()) w.push_back(k);
      else flush_word();
    }
    flush_word();
    out.push_back(std::move(u));
  }
  return out;
}

// Units of one raw run, built in the reading frame of direction q.
std::vector<Unit> run_units(const std::vector<Ch> &chars, int q, double w, double h, bool minor) {
  std::vector<std::vector<FCh>> runs;
  std::vector<FCh> cur;
  for (const Ch &c : chars) {
    const Box bx = to_frame(c.pb, q, w, h);
    if (!cur.empty() && !c.is_space()) {
      const FCh *last = nullptr;
      for (auto it = cur.rbegin(); it != cur.rend(); ++it)
        if (!it->first.is_space()) {
          last = &*it;
          break;
        }
      if (last) {
        const Box &bb = last->second;
        const double hh = std::max({bb[3] - bb[1], bx[3] - bx[1], 1.0});
        if (std::abs((bx[1] + bx[3]) / 2.0 - (bb[1] + bb[3]) / 2.0) > 0.6 * hh || bx[0] < bb[0] - hh) {
          runs.push_back(std::move(cur));
          cur.clear();
        }
      }
    }
    cur.emplace_back(c, bx);
  }
  if (!cur.empty()) runs.push_back(std::move(cur));
  std::vector<Unit> units;
  for (const auto &run : runs) {
    // the gap scale is a glyph height, from letters and digits only
    std::vector<double> hs;
    for (const auto &[c, b] : run)
      if (c.c.size() == 1 && c.c[0] <= 0x10FFFF && !(c.c[0] >= 0xD800 && c.c[0] <= 0xDFFF) && is_alphanumeric(c.c[0]))
        hs.push_back(b[3] - b[1]);
    if (hs.empty())
      for (const auto &[c, b] : run)
        if (!c.is_space()) hs.push_back(b[3] - b[1]);
    if (hs.empty()) continue;
    for (Unit &u : cut(run, median(hs))) units.push_back(std::move(u));
  }
  for (Unit &u : units) {
    u.pbox = from_frame(u.bx, q, w, h);
    u.q = static_cast<uint8_t>(q);
    u.minor = minor;
  }
  return units;
}

// Re-join pieces the baseline rule split but that touch (a superscript, a symbol in another font).
std::vector<Unit> join_touching(const std::vector<Unit> &units) {
  std::vector<size_t> order(units.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return units[a].bx[0] < units[b].bx[0]; });
  std::vector<bool> used(units.size(), false);
  std::vector<Unit> out;
  for (size_t i : order) {
    if (used[i]) continue;
    const Unit &u = units[i];
    Box bx = u.bx;
    std::string text = u.text;
    double h = u.h;
    std::vector<Word> words = u.words;
    bool grew = true;
    while (grew) {
      grew = false;
      for (size_t j : order) {
        if (used[j] || j == i) continue;
        const Unit &v = units[j];
        if (v.q != u.q) continue;
        const double hh = std::max(std::min(h, v.h), 1.0);
        const double gap = v.bx[0] - bx[2];
        if (-0.5 * hh <= gap && gap < 0.1 * hh && std::min(bx[3], v.bx[3]) - std::max(bx[1], v.bx[1]) > 0.0) {
          bx = {std::min(bx[0], v.bx[0]), std::min(bx[1], v.bx[1]), std::max(bx[2], v.bx[2]), std::max(bx[3], v.bx[3])};
          text += v.text;
          h = std::max(h, v.h);
          // touching: v's first word continues u's last
          size_t from = 0;
          if (!words.empty() && !v.words.empty()) {
            Word &a = words.back();
            const Word &b = v.words[0];
            a = {std::min(a.x0, b.x0), std::max(a.x1, b.x1), a.text + b.text};
            from = 1;
          }
          words.insert(words.end(), v.words.begin() + static_cast<long>(from), v.words.end());
          used[j] = true;
          grew = true;
        }
      }
    }
    used[i] = true;
    Unit o = u;
    o.bx = o.pbox = bx;
    o.text = std::move(text);
    o.h = h;
    o.words = std::move(words);
    out.push_back(std::move(o));
  }
  std::stable_sort(out.begin(), out.end(), [](const Unit &a, const Unit &b) {
    const auto ka = static_cast<int64_t>(round_ties_even(a.bx[1])), kb = static_cast<int64_t>(round_ties_even(b.bx[1]));
    if (ka != kb) return ka < kb;
    return a.bx[0] < b.bx[0];
  });
  return out;
}

std::optional<std::vector<Unit>> aligned_pieces(size_t i, const std::vector<Unit> &units) {
  const Unit &u = units[i];
  const auto [x0, y0, x1, y1] = u.bx;
  const double h = std::max(u.h, 1.0);
  const double tol = std::max(0.25 * h, 1.5);
  auto cy = [](const Unit &v) { return (v.bx[1] + v.bx[3]) / 2.0; };
  // the body must look tabular: 2+ rows holding 2+ units each, or one row of 3+
  std::vector<std::pair<int64_t, std::vector<const Unit *>>> rows;
  for (size_t j = 0; j < units.size(); ++j) {
    const Unit &v = units[j];
    const bool near = j != i && v.q == u.q && !v.minor && v.bx[0] < x1 + 2.0 * h && v.bx[2] > x0 - 2.0 * h &&
                      ((y1 - 0.2 * h <= v.bx[1] && v.bx[1] <= y1 + 6.0 * h) ||
                       (y0 - 6.0 * h <= v.bx[3] && v.bx[3] <= y0 + 0.2 * h));
    const size_t nw = v.words.empty() ? 1 : v.words.size();
    if (near && nw <= 4 && v.bx[2] - v.bx[0] < 0.6 * (x1 - x0) && v.bx[0] >= x0 - tol && v.bx[2] <= x1 + 2.0 * h) {
      const auto key = static_cast<int64_t>(round_ties_even(cy(v) / h));
      auto it = std::find_if(rows.begin(), rows.end(), [&](const auto &r) { return r.first == key; });
      if (it != rows.end()) it->second.push_back(&v);
      else rows.push_back({key, {&v}});
    }
  }
  std::vector<const std::vector<const Unit *> *> tab;
  for (const auto &r : rows)
    if (r.second.size() >= 2) tab.push_back(&r.second);
  // a lone body row must be 3+ cells, mostly numbers or short marks ("N/A", "-")
  auto datum = [](const Unit *v) {
    for (char32_t c : utf32(v->text))
      if (is_numeric(c)) return true;
    return char_count(v->text) <= 3;
  };
  bool lone_ok = false;
  if (tab.size() == 1 && tab[0]->size() >= 3) {
    size_t d = 0;
    for (const Unit *v : *tab[0]) d += datum(v);
    lone_ok = 2 * d > tab[0]->size();
  }
  if (!(tab.size() >= 2 || lone_ok)) return std::nullopt;
  std::vector<const Unit *> cells;
  for (const auto *r : tab) cells.insert(cells.end(), r->begin(), r->end());
  // a column edge: shared by cells of two rows (any cell's edge, when one row is the body)
  auto anchors = [&](auto f) {
    std::vector<std::pair<double, double>> vals;
    for (const Unit *v : cells) vals.push_back({f(*v), cy(*v)});
    std::vector<double> out;
    if (tab.size() == 1) {
      for (const auto &x : vals) out.push_back(x.first);
      return out;
    }
    for (const auto &[a, ya] : vals)
      for (const auto &[b, yb] : vals)
        if (std::abs(a - b) <= tol && std::abs(ya - yb) > 0.5 * h) {
          out.push_back(a);
          break;
        }
    return out;
  };
  const auto lefts = anchors([](const Unit &v) { return v.bx[0]; });
  const auto rights = anchors([](const Unit &v) { return v.bx[2]; });
  const auto mids = anchors([](const Unit &v) { return (v.bx[0] + v.bx[2]) / 2.0; });
  const auto &words = u.words;
  auto aligned = [&](size_t a, size_t b) {
    const double l = words[a].x0, r = words[b - 1].x1;
    for (double x : lefts)
      if (std::abs(l - x) <= tol) return true;
    for (double x : rights)
      if (std::abs(r - x) <= tol) return true;
    for (double x : mids)
      if (std::abs((l + r) / 2.0 - x) <= tol) return true;
    return false;
  };
  // no cell crosses mid, and every body row holding cells on both sides leaves a wide gap there
  auto gutter = [&](double mid) {
    for (const Unit *v : cells)
      if (v->bx[0] < mid && mid < v->bx[2]) return false;
    size_t spans = 0;
    for (const auto *r : tab) {
      double left = -kInf, right = kInf;
      for (const Unit *v : *r)
        if (v->bx[2] <= mid) left = std::max(left, v->bx[2]);
      for (const Unit *v : *r)
        if (v->bx[0] >= mid) right = std::min(right, v->bx[0]);
      if (std::isfinite(left) && std::isfinite(right)) {
        if (right - left < 2.0 * h) return false;
        ++spans;
      }
    }
    return spans >= std::min<size_t>(tab.size(), 2);
  };
  std::vector<size_t> cuts{0};
  for (size_t k = 1; k < words.size(); ++k)
    if (gutter((words[k - 1].x1 + words[k].x0) / 2.0)) cuts.push_back(k);
  cuts.push_back(words.size());
  // pieces as consecutive cut pairs; fold unaligned pieces into an aligned merge
  while (cuts.size() >= 3) {
    const size_t n = cuts.size() - 1;
    size_t bad = n;
    for (size_t p = 0; p < n; ++p)
      if (!aligned(cuts[p], cuts[p + 1])) {
        bad = p;
        break;
      }
    if (bad == n) break;
    if (bad > 0 && aligned(cuts[bad - 1], cuts[bad + 1])) cuts.erase(cuts.begin() + static_cast<long>(bad));
    else if (bad + 1 < n && aligned(cuts[bad], cuts[bad + 2])) cuts.erase(cuts.begin() + static_cast<long>(bad) + 1);
    else return std::nullopt;
  }
  const size_t n = cuts.size() - 1;
  if (n < 2) return std::nullopt;
  for (size_t p = 0; p < n; ++p)
    if (cuts[p + 1] - cuts[p] > 5) return std::nullopt;
  // a piece opening in lower case continues a sentence
  for (size_t p = 1; p < n; ++p) {
    const std::u32string s = utf32(words[cuts[p]].text);
    if (!s.empty() && is_lowercase(s[0])) return std::nullopt;
  }
  // each piece over its own column: no cell lies under two pieces
  for (const Unit *v : cells) {
    size_t over = 0;
    for (size_t p = 0; p < n; ++p)
      over += std::min(v->bx[2], words[cuts[p + 1] - 1].x1) > std::max(v->bx[0], words[cuts[p]].x0);
    if (over > 1) return std::nullopt;
  }
  std::vector<Unit> out;
  for (size_t p = 0; p < n; ++p) {
    Unit o;
    o.words.assign(words.begin() + static_cast<long>(cuts[p]), words.begin() + static_cast<long>(cuts[p + 1]));
    o.bx = o.pbox = {o.words.front().x0, y0, o.words.back().x1, y1};
    for (const Word &w : o.words) {
      if (!o.text.empty()) o.text += ' ';
      o.text += w.text;
    }
    o.h = u.h;
    o.size = u.size;
    o.bold = u.bold;
    o.q = u.q;
    o.minor = u.minor;
    out.push_back(std::move(o));
  }
  return out;
}

// A header row PDFium emits as one run splits where its words line up with the table body.
std::vector<Unit> split_aligned(const std::vector<Unit> &units) {
  std::vector<Unit> out;
  out.reserve(units.size());
  for (size_t i = 0; i < units.size(); ++i) {
    const Unit &u = units[i];
    std::optional<std::vector<Unit>> pieces;
    if (!u.minor && u.words.size() >= 2 && u.words.size() <= 16) pieces = aligned_pieces(i, units);
    if (pieces) {
      for (Unit &p : *pieces) out.push_back(std::move(p));
    } else {
      out.push_back(u);
    }
  }
  return out;
}

// Whether at least 80% of `chain` have another unit (not in the chain) centred on their line.
bool row_mates(const std::vector<Unit> &units, const std::vector<size_t> &chain) {
  auto cy = [](const Unit &u) { return (u.pbox[1] + u.pbox[3]) / 2.0; };
  size_t mated = 0;
  for (size_t v : chain) {
    const Box &b = units[v].pbox;
    const double tol = 0.3 * std::max(b[3] - b[1], 1.0);
    for (size_t w = 0; w < units.size(); ++w) {
      if (std::find(chain.begin(), chain.end(), w) != chain.end() || units[w].minor) continue;
      if (std::abs(cy(units[w]) - cy(units[v])) < tol) {
        ++mated;
        break;
      }
    }
  }
  return mated * 5 >= chain.size() * 4;
}

// Single characters stacked in a column (vertical text emitted one glyph per run) are minor -
// unless most of them have row-mates: that is a table column of one-glyph cells.
void flag_letter_stacks(std::vector<Unit> &units) {
  std::vector<size_t> singles;
  for (size_t i = 0; i < units.size(); ++i)
    if (char_count(units[i].text) == 1 && !units[i].minor) singles.push_back(i);
  auto cx = [](const Unit &u) { return (u.pbox[0] + u.pbox[2]) / 2.0; };
  std::stable_sort(singles.begin(), singles.end(), [&](size_t a, size_t b) {
    const auto ka = static_cast<int64_t>(round_ties_even(cx(units[a]))), kb = static_cast<int64_t>(round_ties_even(cx(units[b])));
    if (ka != kb) return ka < kb;
    return units[a].pbox[1] < units[b].pbox[1];
  });
  std::vector<bool> used(units.size(), false);
  for (size_t s = 0; s < singles.size(); ++s) {
    const size_t u = singles[s];
    if (used[u]) continue;
    std::vector<size_t> chain{u};
    size_t last = u;
    for (size_t t = s + 1; t < singles.size(); ++t) {
      const size_t v = singles[t];
      const double cl = cx(units[last]), cv = cx(units[v]);
      const Box &lb = units[last].pbox;
      const double hh = std::max({lb[3] - lb[1], lb[2] - lb[0], 1.0});
      const double dy = units[v].pbox[1] - lb[3];
      if (std::abs(cv - cl) < 0.6 * hh && 0.0 <= dy && dy < 1.5 * hh) {
        chain.push_back(v);
        last = v;
      }
    }
    if (chain.size() >= 3 && !row_mates(units, chain))
      for (size_t v : chain) {
        units[v].minor = true;
        used[v] = true;
      }
  }
}

// Bold from the character's font: weight >= 600, or a bold-ish font name.
int8_t char_bold(FPDF_TEXTPAGE tp, int i) {
  const int w = FPDFText_GetFontWeight(tp, i);
  int flags = 0;
  const unsigned long len = FPDFText_GetFontInfo(tp, i, nullptr, 0, &flags);
  std::optional<std::string> name;
  if (len > 0) {
    std::string buf(len, '\0');
    const unsigned long written = FPDFText_GetFontInfo(tp, i, buf.data(), len, &flags);
    if (written > 0) {
      const size_t n = buf[written - 1] == '\0' ? written - 1 : written;
      name = buf.substr(0, n);
    }
  }
  if (w < 0 && !name) return -1;
  const std::string nm = name.value_or("");
  for (const char *k : {"Bold", "Black", "Heavy", "Semibold", ",B"})
    if (nm.find(k) != std::string::npos) return 1;
  return w >= 600 ? 1 : 0;
}

bool is_char_whitespace(uint32_t u) {
  return u <= 0x10FFFF && !(u >= 0xD800 && u <= 0xDFFF) && is_whitespace(u);
}

}  // namespace

std::array<double, 4> to_frame(const std::array<double, 4> &b, int q, double w, double h) {
  const auto [x0, y0, x1, y1] = b;
  switch (q) {
    case 1: return {h - y1, x0, h - y0, x1};
    case 2: return {w - x1, h - y1, w - x0, h - y0};
    case 3: return {y0, w - x1, y1, w - x0};
    default: return b;
  }
}

std::array<double, 4> from_frame(const std::array<double, 4> &b, int q, double w, double h) {
  const auto [x0, y0, x1, y1] = b;
  switch (q) {
    case 1: return {y0, h - x1, y1, h - x0};
    case 2: return {w - x1, h - y1, w - x0, h - y0};
    case 3: return {w - y1, x0, w - y0, x1};
    default: return b;
  }
}

PageUnits native_units(void *fpdf_page) {
  PageUnits pu;
  auto page = static_cast<FPDF_PAGE>(fpdf_page);
  FS_RECTF vb{};
  if (!FPDF_GetPageBoundingBox(page, &vb)) return pu;
  const double cl = vb.left, cb = vb.bottom, cr = vb.right, ct = vb.top;
  const double w0 = cr - cl, h0 = ct - cb;
  const int rot = ((FPDFPage_GetRotation(page) % 4) + 4) % 4;
  const double w = rot % 2 == 1 ? h0 : w0, h = rot % 2 == 1 ? w0 : h0;
  pu.w = w;
  pu.h = h;
  FPDF_TEXTPAGE tp = FPDFText_LoadPage(page);
  if (!tp) return pu;
  // one entry per char; nullopt for a generated line break
  std::vector<std::optional<Ch>> chars;
  const int n = FPDFText_CountChars(tp);
  for (int i = 0; i < n; ++i) {
    const uint32_t u = FPDFText_GetUnicode(tp, i);
    const bool valid = u <= 0x10FFFF && !(u >= 0xD800 && u <= 0xDFFF);  // char::from_u32
    if (FPDFText_IsGenerated(tp, i) == 1 && valid && (u == U'\r' || u == U'\n')) {
      chars.emplace_back(std::nullopt);
      continue;
    }
    if (u == 0 || (valid && (u == U'\r' || u == U'\n' || u == 0x2 || u == 0xFFFE))) continue;
    double left = 0, right = 0, bottom = 0, top = 0;
    if (!FPDFText_GetCharBox(tp, i, &left, &right, &bottom, &top)) continue;
    const Box pb = rotate_page({left - cl, ct - top, right - cl, ct - bottom}, rot, w0, h0);
    const bool space = is_char_whitespace(u);
    if (pb[2] <= pb[0] && pb[3] <= pb[1] && !space) continue;
    chars.emplace_back(Ch{{u}, pb, FPDFText_GetFontSize(tp, i), char_bold(tp, i)});
  }
  FPDFText_ClosePage(tp);

  // raw runs between PDFium's breaks
  std::vector<std::vector<Ch>> raw_runs;
  std::vector<Ch> cur;
  for (auto &c : chars) {
    if (!c) {
      if (!cur.empty()) raw_runs.push_back(std::move(cur)), cur.clear();
    } else {
      cur.push_back(std::move(*c));
    }
  }
  if (!cur.empty()) raw_runs.push_back(std::move(cur));

  // reading direction per run by how consecutive glyphs advance; votes kept in first-seen order
  using Votes = std::vector<std::pair<int, size_t>>;
  auto direction = [](const std::vector<Ch> &run) {
    Votes v;
    const Ch *prev = nullptr;
    for (const Ch &c : run) {
      if (c.is_space()) continue;
      if (prev) {
        const double dx = (c.pb[0] + c.pb[2] - prev->pb[0] - prev->pb[2]) / 2.0;
        const double dy = (c.pb[1] + c.pb[3] - prev->pb[1] - prev->pb[3]) / 2.0;
        const double sz = std::max({c.pb[2] - c.pb[0], c.pb[3] - c.pb[1], 1.0});
        const double d = std::abs(dx) + std::abs(dy);
        if (0.0 < d && d < 3.0 * sz) {
          const int q = std::abs(dx) >= std::abs(dy) ? (dx > 0.0 ? 0 : 2) : (dy < 0.0 ? 1 : 3);
          auto it = std::find_if(v.begin(), v.end(), [&](const auto &e) { return e.first == q; });
          if (it != v.end()) ++it->second;
          else v.push_back({q, 1});
        }
      }
      prev = &c;
    }
    return v;
  };
  auto most = [](const Votes &v) -> std::optional<int> {
    std::optional<std::pair<int, size_t>> best;
    for (const auto &e : v)
      if (!best || e.second > best->second) best = e;
    if (!best) return std::nullopt;
    return best->first;
  };
  std::vector<Votes> dirs;
  for (const auto &r : raw_runs) dirs.push_back(direction(r));
  Votes page_votes;
  for (const auto &d : dirs)
    for (const auto &[k, cnt] : d) {
      auto it = std::find_if(page_votes.begin(), page_votes.end(), [&](const auto &e) { return e.first == k; });
      if (it != page_votes.end()) it->second += cnt;
      else page_votes.push_back({k, cnt});
    }
  const int q = most(page_votes).value_or(0);
  std::vector<Unit> units;
  for (size_t r = 0; r < raw_runs.size(); ++r) {
    size_t total = 0;
    for (const auto &e : dirs[r]) total += e.second;
    const int rq = !dirs[r].empty() && total >= 2 ? *most(dirs[r]) : q;
    for (Unit &u : run_units(raw_runs[r], rq, w, h, rq != q)) units.push_back(std::move(u));
  }
  units = join_touching(units);
  units = split_aligned(units);
  for (Unit &u : units) u.pbox = from_frame(u.bx, u.q, w, h);
  flag_letter_stacks(units);
  pu.units = std::move(units);
  return pu;
}

PageUnits native_units_doc(void *fpdf_document, int page_index) {
  FPDF_PAGE page = FPDF_LoadPage(static_cast<FPDF_DOCUMENT>(fpdf_document), page_index);
  if (!page) return {};
  PageUnits pu = native_units(page);
  FPDF_ClosePage(page);
  return pu;
}

Unit ocr_unit(std::string_view text, double x0, double y0, double x1, double y1) {
  Unit u;
  u.bx = u.pbox = {x0, y0, x1, y1};
  u.text = std::string(trim(text));
  u.h = (y1 - y0) / 1.4;
  return u;
}

// ---- serialization ----------------------------------------------------------

namespace {
template <class T>
void put(std::vector<uint8_t> &o, const T &v) {
  const auto *p = reinterpret_cast<const uint8_t *>(&v);
  o.insert(o.end(), p, p + sizeof(T));
}
void put_str(std::vector<uint8_t> &o, const std::string &s) {
  put(o, static_cast<uint32_t>(s.size()));
  o.insert(o.end(), s.begin(), s.end());
}
struct Reader {
  const uint8_t *p;
  size_t n, off = 0;
  bool ok = true;
  template <class T>
  T get() {
    T v{};
    if (off + sizeof(T) > n) {
      ok = false;
      return v;
    }
    std::memcpy(&v, p + off, sizeof(T));
    off += sizeof(T);
    return v;
  }
  std::string str() {
    const auto len = get<uint32_t>();
    if (!ok || off + len > n) {
      ok = false;
      return {};
    }
    std::string s(reinterpret_cast<const char *>(p + off), len);
    off += len;
    return s;
  }
};
}  // namespace

void serialize(const PageUnits &pu, std::vector<uint8_t> &out) {
  put(out, pu.w);
  put(out, pu.h);
  put(out, static_cast<uint32_t>(pu.units.size()));
  for (const Unit &u : pu.units) {
    for (double v : u.bx) put(out, v);
    for (double v : u.pbox) put(out, v);
    put_str(out, u.text);
    put(out, u.h);
    put(out, u.size);
    put(out, u.bold);
    put(out, u.q);
    put(out, static_cast<uint8_t>(u.minor));
    put(out, static_cast<uint32_t>(u.words.size()));
    for (const Word &w : u.words) {
      put(out, w.x0);
      put(out, w.x1);
      put_str(out, w.text);
    }
  }
}

size_t deserialize(const uint8_t *p, size_t n, PageUnits &pu) {
  Reader r{p, n};
  pu.w = r.get<double>();
  pu.h = r.get<double>();
  const auto count = r.get<uint32_t>();
  if (!r.ok || count > 1'000'000) return 0;
  pu.units.resize(count);
  for (Unit &u : pu.units) {
    for (double &v : u.bx) v = r.get<double>();
    for (double &v : u.pbox) v = r.get<double>();
    u.text = r.str();
    u.h = r.get<double>();
    u.size = r.get<double>();
    u.bold = r.get<int8_t>();
    u.q = r.get<uint8_t>();
    u.minor = r.get<uint8_t>() != 0;
    const auto nw = r.get<uint32_t>();
    if (!r.ok || nw > 100'000) return 0;
    u.words.resize(nw);
    for (Word &w : u.words) {
      w.x0 = r.get<double>();
      w.x1 = r.get<double>();
      w.text = r.str();
    }
    if (!r.ok) return 0;
  }
  return r.ok ? r.off : 0;
}

}  // namespace hp::tn
