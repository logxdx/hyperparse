#include "table_fix.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string_view>

#include "tablenet/text.h"

namespace hp::tfix {
namespace {

using tn::trim;
using tn::trim_end;
using tn::trim_start;

// A cell on the dense grid: top-left slot, spans, text, box.
struct DCell {
  int r0 = 0, c0 = 0, rs = 1, cs = 1;
  std::string text;
  std::optional<Rect> bbox;
};

struct Dense {
  std::vector<DCell> cells;
  int n_rows = 0, n_cols = 0, header_rows = 0;
  std::vector<char> th;  // hp: per row, written as <th> (SpanTable::th); empty = none
};

bool blank(std::string_view t) { return trim(t).empty(); }

Rect unite(const Rect &a, const Rect &b) {
  const float x0 = std::min(a.x, b.x), y0 = std::min(a.y, b.y);
  const float x1 = std::max(a.x + a.width, b.x + b.width), y1 = std::max(a.y + a.height, b.y + b.height);
  return {x0, y0, x1 - x0, y1 - y0};
}

// ── dense grid <-> span rows ────────────────────────────────────────────────

std::optional<Dense> from_span_rows(const std::vector<std::vector<SpanCell>> &rows, int header_rows,
                                    const std::vector<char> &th = {}) {
  const int n_rows = static_cast<int>(rows.size());
  if (n_rows == 0) return std::nullopt;
  // place cells HTML-style: each row's cells fill the next free slots left to right
  std::vector<std::vector<char>> occ(n_rows);
  Dense d;
  for (int r = 0; r < n_rows; ++r) {
    int c = 0;
    for (const SpanCell &sc : rows[r]) {
      while (c < static_cast<int>(occ[r].size()) && occ[r][c]) ++c;
      const int rs = std::max(sc.rowspan, 1), cs = std::max(sc.colspan, 1);
      for (int rr = r; rr < std::min(r + rs, n_rows); ++rr) {
        if (static_cast<int>(occ[rr].size()) < c + cs) occ[rr].resize(c + cs, 0);
        for (int cc = c; cc < c + cs; ++cc) occ[rr][cc] = 1;
      }
      d.cells.push_back({r, c, std::min(rs, n_rows - r), cs, sc.text, sc.bbox});
      c += cs;
    }
  }
  int n_cols = 0;
  for (const auto &o : occ) n_cols = std::max(n_cols, static_cast<int>(o.size()));
  if (n_cols == 0) return std::nullopt;
  // ragged rows: fill missing slots with empty cells so every slot has an owner
  for (int r = 0; r < n_rows; ++r)
    for (int c = 0; c < n_cols; ++c)
      if (!(c < static_cast<int>(occ[r].size()) && occ[r][c])) d.cells.push_back({r, c, 1, 1, {}, std::nullopt});
  d.n_rows = n_rows;
  d.n_cols = n_cols;
  d.header_rows = std::min(header_rows, n_rows);
  if (!th.empty()) {
    d.th = th;
    d.th.resize(n_rows, 0);
  }
  return d;
}

SpanTable to_span_rows(const Dense &d) {
  std::vector<std::vector<std::pair<int, SpanCell>>> rows(d.n_rows);
  for (const DCell &c : d.cells) rows[c.r0].push_back({c.c0, SpanCell{c.text, c.cs, c.rs, c.bbox}});
  SpanTable t;
  for (auto &r : rows) {
    std::stable_sort(r.begin(), r.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    std::vector<SpanCell> out;
    for (auto &p : r) out.push_back(std::move(p.second));
    t.rows.push_back(std::move(out));
  }
  t.header_rows = d.header_rows;
  t.th = d.th;
  return t;
}

// Index of the cell covering slot (r, c), -1 if none.
int at(const Dense &d, int r, int c) {
  for (size_t k = 0; k < d.cells.size(); ++k) {
    const DCell &x = d.cells[k];
    if (x.r0 <= r && r < x.r0 + x.rs && x.c0 <= c && c < x.c0 + x.cs) return static_cast<int>(k);
  }
  return -1;
}

// Index of the 1x1 cell at (r, c), if the slot holds one (-1 otherwise).
int single(const Dense &d, int r, int c) {
  const int k = at(d, r, c);
  return k >= 0 && d.cells[k].rs == 1 && d.cells[k].cs == 1 ? k : -1;
}

// ── rules ───────────────────────────────────────────────────────────────────

constexpr std::array<std::string_view, 6> kCurrency = {"$", "€", "£", "¥", "US$", "R$"};

bool is_currency(std::string_view t) { return std::find(kCurrency.begin(), kCurrency.end(), t) != kCurrency.end(); }

bool starts_numeric(std::string_view t) {
  const std::u32string u = tn::utf32(trim_start(t));
  if (u.empty()) return false;
  if (tn::is_ascii_digit(u[0])) return true;
  static constexpr std::u32string_view kLead = U"(-−–.";
  return kLead.find(u[0]) != std::u32string_view::npos && u.size() > 1 && tn::is_ascii_digit(u[1]);
}

bool has_digit(std::string_view t) {
  return std::any_of(t.begin(), t.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
}

bool ends_with_wrap(std::string_view t) { return !t.empty() && (t.back() == '-' || t.back() == '/'); }

void strip_leaders(Dense &d) {
  for (DCell &c : d.cells)
    if (c.text.find("....") != std::string::npos || c.text.find("____") != std::string::npos ||
        c.text.find(". . . .") != std::string::npos || c.text.find("…") != std::string::npos)
      c.text = strip_leader_runs(c.text);
}

// `$ | 14,046` -> `| $ 14,046`; `8.8 % $ | 13,917` -> `8.8 % | $ 13,917`.
std::vector<int> move_currency(Dense &d) {
  std::vector<int> emptied;
  for (int r = 0; r < d.n_rows; ++r) {
    for (int c = 0; c + 1 < d.n_cols; ++c) {
      const int a = single(d, r, c), b = single(d, r, c + 1);
      if (a < 0 || b < 0) continue;
      if (!starts_numeric(d.cells[b].text)) continue;
      const std::string ta(trim(d.cells[a].text));
      if (is_currency(ta)) {
        d.cells[b].text = ta + " " + std::string(trim(d.cells[b].text));
        d.cells[a].text.clear();
        if (std::find(emptied.begin(), emptied.end(), c) == emptied.end()) emptied.push_back(c);
        continue;
      }
      // a value with the next column's symbol glued to its end
      const size_t sp = ta.rfind(' ');
      if (sp == std::string::npos) continue;
      const std::string_view head = std::string_view(ta).substr(0, sp), sym = std::string_view(ta).substr(sp + 1);
      if (is_currency(sym) && has_digit(head)) {
        const std::string s(sym);
        d.cells[a].text = std::string(trim_end(head));
        d.cells[b].text = s + " " + std::string(trim(d.cells[b].text));
      }
    }
  }
  return emptied;
}

// Remove a column that moving currency symbols out of left with only empty 1x1 cells or slots
// covered by spans that also cover a neighbouring column (spans shrink by one). Columns that were
// empty to begin with are left alone.
void drop_empty_columns(Dense &d, std::vector<int> emptied) {
  std::sort(emptied.begin(), emptied.end());
  int c = 0;
  while (c < d.n_cols && d.n_cols > 1) {
    if (std::find(emptied.begin(), emptied.end(), c) == emptied.end()) {
      ++c;
      continue;
    }
    bool removable = true;
    for (int r = 0; r < d.n_rows && removable; ++r) {
      const int k = at(d, r, c);
      if (k < 0) continue;
      const DCell &x = d.cells[k];
      if (x.cs == 1) removable = blank(x.text) && x.rs == 1;
    }
    // keep a column that only spans cover (a structural slot) unless it has no own cell
    const bool owns_any =
        std::any_of(d.cells.begin(), d.cells.end(), [&](const DCell &x) { return x.cs == 1 && x.c0 == c; });
    if (!removable || !owns_any) {
      ++c;
      continue;
    }
    std::erase_if(d.cells, [&](const DCell &x) { return x.cs == 1 && x.c0 == c; });
    for (DCell &x : d.cells) {
      if (x.c0 > c) x.c0 -= 1;
      else if (x.c0 + x.cs > c) x.cs -= 1;
    }
    d.n_cols -= 1;
    // later emptied columns moved left with the removal
    std::erase(emptied, c);
    for (int &e : emptied)
      if (e > c) e -= 1;
  }
}

// Delete row r (its own cells go; spans through it shrink; rows below move up).
void remove_row(Dense &d, int r) {
  std::erase_if(d.cells, [&](const DCell &x) { return x.r0 == r && x.rs == 1; });
  for (DCell &x : d.cells) {
    if (x.r0 > r) x.r0 -= 1;
    else if (x.r0 + x.rs > r) x.rs -= 1;
  }
  d.n_rows -= 1;
  if (!d.th.empty()) d.th.erase(d.th.begin() + r);
}

// Append `t` to `up` as a wrapped line: no space after a trailing '-' or '/'.
std::string join_wrap(std::string_view up, std::string_view t) {
  return ends_with_wrap(up) ? std::string(up) + std::string(t) : std::string(up) + " " + std::string(t);
}

// "12." / "3.x": a numbered-list marker at the start of the text (table_fix.rs, `t.split('.')`).
bool numbered_marker(std::string_view t) {
  const size_t dot = t.find('.');
  const std::string_view p = t.substr(0, dot);
  return !p.empty() && p.size() <= 3 && std::all_of(p.begin(), p.end(), [](char ch) { return ch >= '0' && ch <= '9'; }) &&
         t.size() > p.size() && t[p.size()] == '.';
}

bool starts_bullet(std::string_view t) {
  return t.starts_with("•") || t.starts_with("-") || t.starts_with("*") || t.starts_with("–");
}

// Fold a body row that continues the cells of the row above (a wrapped cell emitted as a row).
void fold_continuation_rows(Dense &d) {
  if (d.n_cols < 2) return;
  const int body0 = d.header_rows;
  auto text = [](const Dense &d, int r, int c) -> std::optional<std::string> {
    const int k = single(d, r, c);
    if (k < 0) return std::nullopt;
    return std::string(trim(d.cells[k].text));
  };
  // tables whose records carry a label in the first column
  const int body = std::max(d.n_rows - body0, 0);
  int labelled = 0;
  for (int r = body0; r < d.n_rows; ++r) {
    const auto t = text(d, r, 0);
    labelled += t && !t->empty();
  }
  const bool led = body >= 3 && static_cast<double>(labelled) >= 0.6 * static_cast<double>(body);
  int r = body0 + 1;
  while (r < d.n_rows) {
    std::vector<std::optional<std::string>> row, above;
    for (int c = 0; c < d.n_cols; ++c) row.push_back(text(d, r, c)), above.push_back(text(d, r - 1, c));
    const bool any_none = std::any_of(row.begin(), row.end(), [](const auto &t) { return !t; }) ||
                          std::any_of(above.begin(), above.end(), [](const auto &t) { return !t; });
    if (any_none) {
      // the same wrap written with spans: the record's other cells span both rows and only the
      // wrapped column has a cell of its own in row r
      std::vector<std::pair<int, int>> own;
      int spanned = 0;
      bool ok = true;
      for (int c = 0; c < d.n_cols && ok; ++c) {
        const int k = at(d, r, c);
        if (k >= 0 && d.cells[k].r0 < r) {
          ++spanned;
        } else if (k >= 0 && d.cells[k].rs == 1 && d.cells[k].cs == 1) {
          const std::string_view t = trim(d.cells[k].text);
          if (t.empty()) continue;
          const int j = single(d, r - 1, c);
          if (j < 0) {
            ok = false;
            continue;
          }
          const std::string_view u = trim(d.cells[j].text);
          const bool fits = !u.empty() && (ends_with_wrap(u) || !is_value(std::string(t)));
          if (fits) own.push_back({j, k});
          ok = fits;
        } else {
          ok = false;
        }
      }
      // a wrap continues one or two cells of a record whose other cells span it; a row with more
      // cells of its own than spanned slots is a new record under row-group labels
      int own_filled = 0;
      for (int c = 0; c < d.n_cols; ++c) {
        const int k = at(d, r, c);
        own_filled += k >= 0 && d.cells[k].r0 == r && !blank(d.cells[k].text);
      }
      if (ok && spanned > 0 && !own.empty() && own_filled <= 2 && own_filled < spanned) {
        for (const auto &[j, k] : own) {
          const std::string t(trim(d.cells[k].text));
          const std::string up(trim_end(d.cells[j].text));
          d.cells[j].text = join_wrap(up, t);
          if (d.cells[j].bbox && d.cells[k].bbox) d.cells[j].bbox = unite(*d.cells[j].bbox, *d.cells[k].bbox);
        }
        remove_row(d, r);
        continue;
      }
      ++r;
      continue;
    }
    std::vector<std::string> rt, at_;
    for (auto &t : row) rt.push_back(std::move(*t));
    for (auto &t : above) at_.push_back(std::move(*t));
    std::vector<int> filled;
    for (int c = 0; c < d.n_cols; ++c)
      if (!rt[c].empty()) filled.push_back(c);
    const size_t filled_above = std::count_if(at_.begin(), at_.end(), [](const std::string &t) { return !t.empty(); });
    // each filled cell continues the cell above it on the page: the one its box overlaps most
    // horizontally (a renderer may have put a lone continuation line in the first slot), else
    // the one in its own column
    auto target = [&](int c) -> int {
      const int kb = single(d, r, c);
      if (kb < 0 || !d.cells[kb].bbox) return c;
      const Rect b = *d.cells[kb].bbox;
      int best = -1;
      float best_o = 0;
      for (int cc = 0; cc < d.n_cols; ++cc) {
        const int ka = single(d, r - 1, cc);
        if (ka < 0 || !d.cells[ka].bbox) continue;
        const Rect a = *d.cells[ka].bbox;
        const float o = std::min(b.x + b.width, a.x + a.width) - std::max(b.x, a.x);
        if (!(o > 0.5f * std::min(b.width, a.width))) continue;
        if (best < 0 || o >= best_o) best = cc, best_o = o;  // Iterator::max_by keeps the last maximum
      }
      return best < 0 ? c : best;
    };
    std::vector<int> targets;
    for (int c : filled) targets.push_back(target(c));
    std::vector<int> sorted = targets;
    std::sort(sorted.begin(), sorted.end());
    const bool distinct = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
    bool continues = !filled.empty() && distinct && filled.size() < filled_above;
    for (size_t i = 0; i < filled.size() && continues; ++i) {
      const std::string &t = rt[filled[i]], &u = at_[targets[i]];
      const bool marker = starts_bullet(t) || numbered_marker(t);
      continues = !u.empty() && !marker &&
                  (ends_with_wrap(u) || (!has_digit(t) && led && targets[i] != 0 && !at_[0].empty()));
    }
    if (!continues) {
      ++r;
      continue;
    }
    for (size_t i = 0; i < filled.size(); ++i) {
      const int a = single(d, r - 1, targets[i]), b = single(d, r, filled[i]);
      if (a < 0 || b < 0) continue;
      const std::string t(trim(d.cells[b].text));
      const std::string up(trim_end(d.cells[a].text));
      d.cells[a].text = join_wrap(up, t);
      if (d.cells[a].bbox && d.cells[b].bbox) d.cells[a].bbox = unite(*d.cells[a].bbox, *d.cells[b].bbox);
    }
    remove_row(d, r);
  }
}

// In a multi-row header, a cell followed by empty slots that the next header row fills spans those
// columns when its box is centred over them (a group header over its sub-headers).
void span_group_headers(Dense &d) {
  if (d.header_rows < 2) return;
  for (int h = 0; h < d.header_rows - 1; ++h) {
    int c = 0;
    while (c < d.n_cols) {
      const int k = single(d, h, c);
      if (k < 0 || blank(d.cells[k].text)) {
        ++c;
        continue;
      }
      int end = c;
      while (end + 1 < d.n_cols) {
        const int j = single(d, h, end + 1);
        if (j < 0 || !blank(d.cells[j].text)) break;
        const int s = at(d, h + 1, end + 1);
        if (s < 0 || blank(d.cells[s].text)) break;
        ++end;
      }
      if (end == c) {
        ++c;
        continue;
      }
      // centred over the group: closer to the group's centre than to its first sub-column's
      auto centre = [](const Rect &b) { return b.x + b.width / 2.0f; };
      std::vector<Rect> subs;
      for (int cc = c; cc <= end; ++cc) {
        const int j = at(d, h + 1, cc);
        if (j >= 0 && d.cells[j].bbox) subs.push_back(*d.cells[j].bbox);
      }
      bool ok = false;
      if (d.cells[k].bbox && !subs.empty()) {
        Rect g = subs[0];
        for (size_t i = 1; i < subs.size(); ++i) g = unite(g, subs[i]);
        const Rect hb = *d.cells[k].bbox;
        ok = std::abs(centre(hb) - centre(g)) < std::abs(centre(hb) - centre(subs[0]));
      }
      if (!ok) {
        c = end + 1;
        continue;
      }
      std::erase_if(d.cells, [&](const DCell &x) { return x.r0 == h && x.rs == 1 && x.cs == 1 && x.c0 > c && x.c0 <= end; });
      d.cells[single(d, h, c)].cs = end - c + 1;
      c = end + 1;
    }
  }
}

void run(Dense &d) {
  strip_leaders(d);
  const std::vector<int> emptied = move_currency(d);
  drop_empty_columns(d, emptied);
  fold_continuation_rows(d);
  span_group_headers(d);
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

}  // namespace

// A run of 4+ leader characters (dots / underscores / middle dots / ellipses), spaces allowed between.
std::string strip_leader_runs(const std::string &t) {
  const std::u32string chars = tn::utf32(t);
  auto leader = [](char32_t ch) { return ch == U'.' || ch == U'_' || ch == U'·' || ch == U'…'; };
  std::u32string out;
  size_t i = 0;
  while (i < chars.size()) {
    if (leader(chars[i])) {
      size_t j = i, n = 0;
      while (j < chars.size() && (leader(chars[j]) || chars[j] == U' ')) {
        n += leader(chars[j]);
        ++j;
      }
      if (n >= 4) {
        out.push_back(U' ');
        i = j;
        continue;
      }
    }
    out.push_back(chars[i]);
    ++i;
  }
  std::string s;
  for (const std::string &w : tn::split_whitespace(tn::utf8(out))) s += (s.empty() ? "" : " ") + w;
  return s;
}

// A plain number / amount / percent / date-like value (not wrapped label text).
bool is_value(const std::string &t) {
  size_t digits = 0, letters = 0;
  for (char32_t ch : tn::utf32(t)) digits += tn::is_ascii_digit(ch), letters += tn::is_alphabetic(ch);
  return digits > 0 && digits >= letters;
}

void fix_merged(SpanTable &t) {
  std::optional<Dense> d = from_span_rows(t.rows, t.header_rows, t.th);
  if (!d) return;
  run(*d);
  t = to_span_rows(*d);
}

void fix_plain(SpanTable &t) {
  std::optional<Dense> d = from_span_rows(t.rows, t.header_rows > 0 ? 1 : 0);
  if (!d) return;
  run(*d);
  // a plain table stays plain: the rules here never create spans in a one-row header
  SpanTable out = to_span_rows(*d);
  for (const auto &r : out.rows)
    for (const SpanCell &c : r)
      if (c.colspan > 1 || c.rowspan > 1) return;
  out.header_rows = out.header_rows > 0 && !out.rows.empty() ? 1 : 0;
  t = std::move(out);
}

SpanTable span_rows(const tn::Grid &g) {
  std::map<std::pair<int, int>, size_t> at;  // top-left slot -> cell (the last one wins)
  std::set<std::pair<int, int>> taken;
  for (size_t k = 0; k < g.cells.size(); ++k) {
    const tn::GridCell &c = g.cells[k];
    at[{c.r0, c.c0}] = k;
    for (int r = c.r0; r <= c.r1; ++r)
      for (int q = c.c0; q <= c.c1; ++q) taken.insert({r, q});
  }
  SpanTable t;
  for (int r = 0; r < g.n_rows; ++r) {
    std::vector<SpanCell> row;
    int q = 0;
    while (q < g.n_cols) {
      const auto it = at.find({r, q});
      if (it != at.end()) {
        const tn::GridCell &c = g.cells[it->second];
        const auto &b = c.bx;
        row.push_back({c.text, c.c1 - q + 1, c.r1 - r + 1,
                       Rect{static_cast<float>(b[0]), static_cast<float>(b[1]), static_cast<float>(b[2] - b[0]),
                            static_cast<float>(b[3] - b[1])}});
        q = c.c1 + 1;
      } else if (taken.count({r, q})) {
        ++q;
      } else {
        row.push_back({});
        ++q;
      }
    }
    t.rows.push_back(std::move(row));
  }
  while (t.header_rows < g.n_rows &&
         std::find(g.head_rows.begin(), g.head_rows.end(), t.header_rows) != g.head_rows.end())
    ++t.header_rows;
  // head rows below the leading ones keep their <th> (as tn::Grid::to_html writes them)
  for (int r = t.header_rows; r < g.n_rows; ++r)
    if (std::find(g.head_rows.begin(), g.head_rows.end(), r) != g.head_rows.end()) {
      t.th.assign(g.n_rows, 0);
      for (int h : g.head_rows)
        if (h >= 0 && h < g.n_rows) t.th[h] = 1;
      break;
    }
  return t;
}

std::string to_html(const SpanTable &t) {
  std::string out = "<table>";
  for (size_t r = 0; r < t.rows.size(); ++r) {
    const bool head = static_cast<int>(r) < t.header_rows || (r < t.th.size() && t.th[r]);
    const std::string tag = head ? "th" : "td";
    std::string tds;
    for (const SpanCell &c : t.rows[r]) {
      std::string attr;
      if (c.rowspan > 1) attr += " rowspan=\"" + std::to_string(c.rowspan) + "\"";
      if (c.colspan > 1) attr += " colspan=\"" + std::to_string(c.colspan) + "\"";
      tds += "<" + tag + attr + ">" + escape(c.text) + "</" + tag + ">";
    }
    out += "\n<tr>" + tds + "</tr>";
  }
  out += "\n</table>";
  return out;
}

bool is_plain(const SpanTable &t) {
  if (t.header_rows > 1) return false;
  for (size_t r = 1; r < t.th.size(); ++r)
    if (t.th[r]) return false;
  for (const auto &r : t.rows)
    for (const SpanCell &c : r)
      if (c.colspan > 1 || c.rowspan > 1) return false;
  const size_t w = t.rows.empty() ? 0 : t.rows[0].size();
  return std::all_of(t.rows.begin(), t.rows.end(), [&](const auto &r) { return r.size() == w; });
}

std::vector<BandLine> band_lines(const std::vector<BandSpan> &spans, const std::vector<char> &skip) {
  std::vector<int> idx;
  for (size_t i = 0; i < spans.size(); ++i)
    if ((skip.empty() || !skip[i]) && !blank(spans[i].text) && spans[i].height > 0) idx.push_back(static_cast<int>(i));
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    const float ca = spans[a].y + spans[a].height / 2, cb = spans[b].y + spans[b].height / 2;
    return ca != cb ? ca < cb : spans[a].x < spans[b].x;
  });
  struct Group {
    float y0, y1;
    std::vector<int> m;
  };
  std::vector<Group> groups;
  for (int i : idx) {
    const BandSpan &s = spans[i];
    if (!groups.empty()) {
      Group &g = groups.back();
      const float ov = std::min(g.y1, s.y + s.height) - std::max(g.y0, s.y);
      if (ov >= 0.5f * std::min(s.height, g.y1 - g.y0)) {
        g.y0 = std::min(g.y0, s.y);
        g.y1 = std::max(g.y1, s.y + s.height);
        g.m.push_back(i);
        continue;
      }
    }
    groups.push_back({s.y, s.y + s.height, {i}});
  }
  std::vector<BandLine> out;
  for (Group &g : groups) {
    std::stable_sort(g.m.begin(), g.m.end(), [&](int a, int b) { return spans[a].x < spans[b].x; });
    BandLine l;
    l.members = g.m;
    for (int i : g.m) {
      const BandSpan &s = spans[i];
      const std::string t(trim(s.text));
      if (!l.spans.empty()) {
        BandSpan &p = l.spans.back();
        const float h = std::min(p.height, s.height), gap = s.x - (p.x + p.width);
        if (gap < 0.5f * h) {  // a word gap: same phrase
          if (gap > 0.12f * h) p.text += ' ';
          p.text += t;
          const float x1 = std::max(p.x + p.width, s.x + s.width), y1 = std::max(p.y + p.height, s.y + s.height);
          p.x = std::min(p.x, s.x);
          p.y = std::min(p.y, s.y);
          p.width = x1 - p.x;
          p.height = y1 - p.y;
          continue;
        }
      }
      l.spans.push_back({t, s.x, s.y, s.width, s.height});
    }
    Rect box{l.spans[0].x, l.spans[0].y, l.spans[0].width, l.spans[0].height};
    for (const BandSpan &s : l.spans) {
      box = unite(box, Rect{s.x, s.y, s.width, s.height});
      l.text += (l.text.empty() ? "" : " ") + s.text;
    }
    l.box = box;
    out.push_back(std::move(l));
  }
  return out;
}

std::vector<int> absorb_header_band(tn::Grid &g, const std::vector<BandLine> &lines, const std::vector<char> &taken,
                                    bool full_top) {
  if (g.n_cols < 2 || g.n_rows < 2) return {};
  auto filled = [&](int r, int c) {
    return std::any_of(g.cells.begin(), g.cells.end(), [&](const tn::GridCell &x) {
      return x.r0 <= r && r <= x.r1 && x.c0 <= c && c <= x.c1 && !blank(x.text);
    });
  };
  std::vector<int> empty0;
  for (int c = 0; c < g.n_cols; ++c)
    if (!filled(0, c)) empty0.push_back(c);
  // hp: with every top slot filled (and the caller allowing it), any column will do, but a line
  // holding a value is a data row left out of the table, not its header
  const bool headless = full_top && empty0.empty();
  if (headless)
    for (int c = 0; c < g.n_cols; ++c) empty0.push_back(c);
  if (empty0.empty()) return {};
  // column extents from the single-column cells
  std::vector<std::optional<std::pair<double, double>>> rng(g.n_cols);
  std::vector<double> hs;
  for (const tn::GridCell &x : g.cells) {
    if (blank(x.text)) continue;
    hs.push_back(x.bx[3] - x.bx[1]);
    if (x.c0 == x.c1) {
      auto &e = rng[x.c0];
      if (!e) e = std::pair{x.bx[0], x.bx[2]};
      e->first = std::min(e->first, x.bx[0]);
      e->second = std::max(e->second, x.bx[2]);
    }
  }
  if (hs.empty() || std::any_of(rng.begin(), rng.end(), [](const auto &r) { return !r; })) return {};
  std::sort(hs.begin(), hs.end());
  const double hh = std::max(hs[hs.size() / 2], 1.0);
  double top = std::numeric_limits<double>::infinity();
  for (const tn::GridCell &x : g.cells) top = std::fmin(top, x.bx[1]);
  const double tx0 = rng[0]->first, tx1 = rng[g.n_cols - 1]->second;
  auto column = [&](double cx) -> int {
    int best = -1;
    double bd = std::numeric_limits<double>::infinity();
    for (int c = 0; c < g.n_cols; ++c) {
      const auto [a, b] = *rng[c];
      const double dist = cx < a ? a - cx : (cx > b ? cx - b : 0.0);
      if (dist < bd) bd = dist, best = c;
    }
    return bd <= 2.0 * hh ? best : -1;
  };
  // candidate lines above the table, closest first
  std::vector<int> cand;
  for (size_t i = 0; i < lines.size(); ++i) {
    const Rect &b = lines[i].box;
    const double bot = static_cast<double>(b.y + b.height), cx = static_cast<double>(b.x + b.width / 2.0f);
    if (!(i < taken.size() && taken[i]) && lines[i].spans.size() >= 2 && bot <= top + 0.3 * hh &&
        top - bot <= 2.5 * hh && tx0 - hh <= cx && cx <= tx1 + hh)
      cand.push_back(static_cast<int>(i));
  }
  std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return lines[b].box.y < lines[a].box.y; });
  struct BandCell {
    int c;
    std::string text;
    std::array<double, 4> bx;
  };
  std::vector<std::pair<int, std::vector<BandCell>>> rows;
  double edge = top;
  for (int i : cand) {
    const BandLine &l = lines[i];
    const double bot = static_cast<double>(l.box.y + l.box.height);
    if (edge - bot > 2.0 * hh || rows.size() >= 2) break;
    const std::string_view text = trim(l.text);
    if (tn::split_whitespace(text).size() > 16 || (text.ends_with('.') && text.size() > 40)) break;
    std::vector<const BandSpan *> spans;
    for (const BandSpan &s : l.spans)
      if (!blank(s.text)) spans.push_back(&s);
    std::stable_sort(spans.begin(), spans.end(), [](const BandSpan *a, const BandSpan *b) { return a->x < b->x; });
    bool wide_gap = false;
    for (size_t k = 1; k < spans.size(); ++k)
      wide_gap |= static_cast<double>(spans[k]->x - (spans[k - 1]->x + spans[k - 1]->width)) >= 1.5 * hh;
    std::vector<BandCell> cells;
    bool ok = wide_gap;
    for (const BandSpan *s : spans) {
      const int c = column(static_cast<double>(s->x + s->width / 2.0f));
      if (c < 0) {
        ok = false;
        break;
      }
      const std::array<double, 4> bx{s->x, s->y, static_cast<double>(s->x + s->width),
                                     static_cast<double>(s->y + s->height)};
      auto e = std::find_if(cells.begin(), cells.end(), [&](const BandCell &x) { return x.c == c; });
      if (e != cells.end()) {
        e->text = e->text + " " + std::string(trim(s->text));
        e->bx = {std::min(e->bx[0], bx[0]), std::min(e->bx[1], bx[1]), std::max(e->bx[2], bx[2]),
                 std::max(e->bx[3], bx[3])};
      } else {
        cells.push_back({c, std::string(trim(s->text)), bx});
      }
    }
    const bool over_empty = std::any_of(cells.begin(), cells.end(), [&](const BandCell &x) {
      return std::find(empty0.begin(), empty0.end(), x.c) != empty0.end();
    });
    if (!ok || cells.size() < 2 || !over_empty) break;
    if (headless && std::any_of(cells.begin(), cells.end(), [](const BandCell &x) { return is_value(x.text); })) break;
    rows.push_back({i, std::move(cells)});
    edge = static_cast<double>(l.box.y);
  }
  if (rows.empty()) return {};
  // top-most line first; every existing row moves down
  std::reverse(rows.begin(), rows.end());
  const int k = static_cast<int>(rows.size());
  for (tn::GridCell &x : g.cells) x.r0 += k, x.r1 += k;
  for (int &h : g.head_rows) h += k;
  for (int r = 0; r < k; ++r) {
    g.head_rows.push_back(r);
    for (int c = 0; c < g.n_cols; ++c) {
      const auto &cells = rows[r].second;
      auto e = std::find_if(cells.begin(), cells.end(), [&](const BandCell &x) { return x.c == c; });
      tn::GridCell gc;
      gc.r0 = gc.r1 = r;
      gc.c0 = gc.c1 = c;
      if (e != cells.end()) gc.text = e->text, gc.bx = e->bx;
      g.cells.push_back(std::move(gc));
    }
  }
  // the band's rows lead the header; rows that were header stay header
  std::sort(g.head_rows.begin(), g.head_rows.end());
  g.head_rows.erase(std::unique(g.head_rows.begin(), g.head_rows.end()), g.head_rows.end());
  g.n_rows += k;
  std::vector<int> out;
  for (const auto &r : rows) out.push_back(r.first);
  return out;
}

std::string model_table_html(tn::Grid g, const std::vector<BandSpan> &spans, std::vector<char> &taken,
                             const FixOptions &opt) {
  if (opt.band) {
    const std::vector<BandLine> lines = band_lines(spans, taken);
    // hp: a grid without any header row most often lost it (the header row is in the Table
    // region, above the cells LayoutNet assembled); there the band may cover a full top row.
    for (int i : absorb_header_band(g, lines, {}, opt.headless && g.head_rows.empty()))
      for (int m : lines[i].members) taken[m] = 1;
  }
  SpanTable t = span_rows(g);
  if (opt.clean_up) fix_merged(t);
  return to_html(t);
}

}  // namespace hp::tfix
