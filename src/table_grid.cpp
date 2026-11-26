#include "table_grid.h"

#include <algorithm>
#include <cmath>
#include <optional>

#include "table_fix.h"

namespace hp {
namespace {

struct Row {
  float y0, y1;
  std::vector<PageLine> spans;  // left to right after finalisation
};

float height(const PageLine &s) { return s.y1 - s.y0; }

float median(std::vector<float> v) {
  if (v.empty()) return 0;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

std::string trim(const std::string &s) {
  const auto a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

// Cell text: line breaks become spaces.
std::string cell_text(const std::string &s) {
  std::string out;
  for (char c : s) out += c == '\n' || c == '\r' ? ' ' : c;
  return trim(out);
}

// Markdown pipe-table cell: '|' escaped.
std::string escape_cell(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '|') out += "\\|";
    else out += c;
  }
  return out;
}

tfix::Rect rect_of(const PageLine &s) { return {s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0}; }

tfix::Rect unite(const tfix::Rect &a, const tfix::Rect &b) {
  const float x0 = std::min(a.x, b.x), y0 = std::min(a.y, b.y);
  const float x1 = std::max(a.x + a.width, b.x + b.width), y1 = std::max(a.y + a.height, b.y + b.height);
  return {x0, y0, x1 - x0, y1 - y0};
}

// Words of one cell: spans closer than a word gap become one cell candidate.
std::vector<PageLine> join_words(std::vector<PageLine> spans, float h) {
  std::sort(spans.begin(), spans.end(), [](const PageLine &a, const PageLine &b) { return a.x0 < b.x0; });
  std::vector<PageLine> out;
  for (auto &s : spans) {
    if (!out.empty() && s.x0 - out.back().x1 < 0.5f * h) {  // < the smallest column gap
      PageLine &c = out.back();
      if (s.x0 - c.x1 > 0.12f * h) c.text += ' ';
      c.text += s.text;
      c.x1 = std::max(c.x1, s.x1);
      c.y0 = std::min(c.y0, s.y0);
      c.y1 = std::max(c.y1, s.y1);
      continue;
    }
    out.push_back(std::move(s));
  }
  return out;
}

// Cells of one grid row: text and box per column.
struct Cells {
  float y0, y1;
  std::vector<std::string> text;
  std::vector<std::optional<tfix::Rect>> box;
  size_t filled = 0;
};

// Append a wrapped line to a cell: no space after a trailing '-' or '/' ("Co-" + "Pay").
void append_line(std::string &up, const std::string &t) {
  if (up.empty()) up = t;
  else if (up.back() == '-' || up.back() == '/') up += t;
  else up += ' ' + t;
}

}  // namespace

bool stack_header_lines(std::vector<std::vector<std::string>> &text, std::vector<std::vector<std::optional<tfix::Rect>>> *box,
                        const std::vector<std::pair<float, float>> &band, float h) {
  size_t b = 0;
  while (b < text.size() &&
         std::none_of(text[b].begin(), text[b].end(), [](const std::string &t) { return tfix::is_value(t); }))
    ++b;
  // a line repeating a body cell's text in the same column is a data row without a number (a
  // listing's first record: "Common Stock | KO | New York Stock Exchange"), not header text
  for (size_t i = 1; i < b; ++i) {
    bool repeats = false;
    for (size_t k = 0; k < text[i].size() && !repeats; ++k)
      for (size_t j = b; j < text.size() && !repeats; ++j)
        repeats = !text[i][k].empty() && text[i][k] == text[j][k];
    if (repeats) {
      b = i;
      break;
    }
  }
  if (b < 2 || b > kMaxHeaderLines || b >= text.size()) return false;
  for (size_t i = 1; i < b; ++i)
    if (band[i].first - band[i - 1].second >= 0.6f * h) return false;
  for (size_t i = 1; i < b; ++i)
    for (size_t k = 0; k < text[0].size(); ++k) {
      if (text[i][k].empty()) continue;
      append_line(text[0][k], text[i][k]);
      if (box) {
        auto &hb = (*box)[0][k];
        const auto &ib = (*box)[i][k];
        if (ib) hb = hb ? unite(*hb, *ib) : ib;
      }
    }
  text.erase(text.begin() + 1, text.begin() + static_cast<std::ptrdiff_t>(b));
  if (box) box->erase(box->begin() + 1, box->begin() + static_cast<std::ptrdiff_t>(b));
  return true;
}

std::string table_markdown(std::vector<PageLine> spans, const GridOptions &opt) {
  spans.erase(std::remove_if(spans.begin(), spans.end(),
                             [](const PageLine &s) { return trim(s.text).empty() || height(s) <= 0; }),
              spans.end());
  if (spans.size() < 4) return "";
  std::vector<float> hs;
  for (const auto &s : spans) hs.push_back(height(s));
  const float h = median(hs);

  // ── rows ──
  std::sort(spans.begin(), spans.end(),
            [](const PageLine &a, const PageLine &b) { return a.y0 + a.y1 < b.y0 + b.y1; });
  std::vector<Row> rows;
  for (auto &s : spans) {
    if (!rows.empty()) {
      Row &r = rows.back();
      const float ov = std::min(r.y1, s.y1) - std::max(r.y0, s.y0);
      if (ov >= 0.5f * std::min(height(s), r.y1 - r.y0)) {
        r.y0 = std::min(r.y0, s.y0);
        r.y1 = std::max(r.y1, s.y1);
        r.spans.push_back(std::move(s));
        continue;
      }
    }
    rows.push_back({s.y0, s.y1, {std::move(s)}});
  }
  for (auto &r : rows) r.spans = join_words(std::move(r.spans), h);
  if (rows.size() < 2) return "";

  // ── columns: x ranges covered by no multi-cell row ──
  float x_lo = 1e30f, x_hi = -1e30f;
  for (const auto &r : rows)
    for (const auto &s : r.spans) x_lo = std::min(x_lo, s.x0), x_hi = std::max(x_hi, s.x1);
  const int W = static_cast<int>(std::ceil(x_hi - x_lo)) + 1;
  std::vector<int> cover(static_cast<size_t>(W), 0);
  int voters = 0;
  for (const auto &r : rows) {
    if (r.spans.size() < 2) continue;  // a lone span (title, wrapped text) may straddle columns
    ++voters;
    for (const auto &s : r.spans)
      for (int x = static_cast<int>(s.x0 - x_lo); x <= static_cast<int>(s.x1 - x_lo) && x < W; ++x) ++cover[x];
  }
  if (voters < 2) return "";
  std::vector<float> cuts;  // column boundaries (page x)
  for (int x = 0; x < W;) {
    if (cover[x] > 0) { ++x; continue; }
    int e = x;
    while (e < W && cover[e] == 0) ++e;
    if (x > 0 && e < W && e - x >= 0.5f * h) cuts.push_back(x_lo + 0.5f * (x + e));
    x = e;
  }
  const size_t ncols = cuts.size() + 1;
  if (ncols < 2) return "";
  auto col_of = [&](const PageLine &s) {
    const float cx = 0.5f * (s.x0 + s.x1);
    return static_cast<size_t>(std::upper_bound(cuts.begin(), cuts.end(), cx) - cuts.begin());
  };

  // ── cells, with wrapped-cell rows folded into the row above ──
  std::vector<Cells> grid;
  for (const auto &r : rows) {
    Cells c{r.y0, r.y1, std::vector<std::string>(ncols), std::vector<std::optional<tfix::Rect>>(ncols), 0};
    for (const auto &s : r.spans) {
      const size_t k = col_of(s);
      std::string &t = c.text[k];
      if (!t.empty()) t += ' ';
      t += cell_text(s.text);
      c.box[k] = c.box[k] ? unite(*c.box[k], rect_of(s)) : rect_of(s);
    }
    for (const auto &t : c.text) c.filled += !t.empty();
    if (!grid.empty()) {
      Cells &p = grid.back();
      bool subset = c.filled < p.filled;
      for (size_t k = 0; k < ncols && subset; ++k) subset = c.text[k].empty() || !p.text[k].empty();
      if (subset && c.y0 - p.y1 < 0.6f * h) {
        for (size_t k = 0; k < ncols; ++k)
          if (!c.text[k].empty()) {
            p.text[k] += ' ' + c.text[k];
            p.box[k] = unite(*p.box[k], *c.box[k]);
          }
        p.y1 = c.y1;
        continue;
      }
    }
    grid.push_back(std::move(c));
  }
  // Drop columns no row uses (a gap inside one column's whitespace).
  std::vector<size_t> used;
  for (size_t k = 0; k < ncols; ++k)
    if (std::any_of(grid.begin(), grid.end(), [k](const Cells &c) { return !c.text[k].empty(); })) used.push_back(k);
  if (used.size() < 2) return "";
  size_t multi = 0;
  for (const auto &c : grid) multi += c.filled >= 2;
  if (grid.size() < 2 || 2 * multi < grid.size()) return "";
  std::vector<std::vector<std::string>> text;
  std::vector<std::vector<std::optional<tfix::Rect>>> box;
  std::vector<std::pair<float, float>> band;
  for (auto &c : grid) {
    std::vector<std::string> t;
    std::vector<std::optional<tfix::Rect>> b;
    for (size_t k : used) t.push_back(std::move(c.text[k])), b.push_back(c.box[k]);
    text.push_back(std::move(t));
    box.push_back(std::move(b));
    band.push_back({c.y0, c.y1});
  }
  return write_grid(std::move(text), std::move(box), band, h, opt);
}

std::string write_grid(std::vector<std::vector<std::string>> text, std::vector<std::vector<std::optional<tfix::Rect>>> box,
                       const std::vector<std::pair<float, float>> &band, float h, const GridOptions &opt) {
  // ── a header wrapped over several rows becomes one row, then the table clean-up (table_fix.h)
  // on a plain table whose first row is the header, as liteparse fixes its heuristic tables ──
  if (opt.stack_header) stack_header_lines(text, &box, band, h);
  tfix::SpanTable t;
  t.header_rows = 1;
  for (size_t r = 0; r < text.size(); ++r) {
    std::vector<tfix::SpanCell> row;
    for (size_t k = 0; k < text[r].size(); ++k) row.push_back({std::move(text[r][k]), 1, 1, box[r][k]});
    t.rows.push_back(std::move(row));
  }
  if (opt.clean_up) tfix::fix_plain(t);
  if (!tfix::is_plain(t) || t.rows.empty()) return tfix::to_html(t);

  // ── Markdown ──
  auto line = [&](const std::vector<tfix::SpanCell> &cells) {
    std::string s = "|";
    for (const auto &c : cells) s += ' ' + escape_cell(c.text) + " |";
    return s + '\n';
  };
  std::string md = line(t.rows[0]);
  md += line(std::vector<tfix::SpanCell>(t.rows[0].size(), tfix::SpanCell{"---", 1, 1, std::nullopt}));
  for (size_t i = 1; i < t.rows.size(); ++i) md += line(t.rows[i]);
  md.pop_back();
  return md;
}

}  // namespace hp
