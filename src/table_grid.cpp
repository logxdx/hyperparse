#include "table_grid.h"

#include <algorithm>
#include <cmath>

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

std::string escape_cell(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '|') out += "\\|";
    else if (c == '\n' || c == '\r') out += ' ';
    else out += c;
  }
  return trim(out);
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

}  // namespace

std::string table_markdown(std::vector<PageLine> spans) {
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
  struct Cells {
    float y0, y1;
    std::vector<std::string> text;
    size_t filled = 0;
  };
  std::vector<Cells> grid;
  for (const auto &r : rows) {
    Cells c{r.y0, r.y1, std::vector<std::string>(ncols), 0};
    for (const auto &s : r.spans) {
      std::string &t = c.text[col_of(s)];
      if (!t.empty()) t += ' ';
      t += escape_cell(s.text);
    }
    for (const auto &t : c.text) c.filled += !t.empty();
    if (!grid.empty()) {
      Cells &p = grid.back();
      bool subset = c.filled < p.filled;
      for (size_t k = 0; k < ncols && subset; ++k) subset = c.text[k].empty() || !p.text[k].empty();
      if (subset && c.y0 - p.y1 < 0.6f * h) {
        for (size_t k = 0; k < ncols; ++k)
          if (!c.text[k].empty()) p.text[k] += ' ' + c.text[k];
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
  for (auto &c : grid) {
    std::vector<std::string> t;
    for (size_t k : used) t.push_back(std::move(c.text[k]));
    c.text = std::move(t);
  }
  const size_t kept = used.size();
  size_t multi = 0;
  for (const auto &c : grid) multi += c.filled >= 2;
  if (grid.size() < 2 || 2 * multi < grid.size()) return "";

  // ── Markdown ──
  auto line = [&](const std::vector<std::string> &cells) {
    std::string s = "|";
    for (const auto &t : cells) s += ' ' + t + " |";
    return s + '\n';
  };
  std::string md = line(grid[0].text);
  md += line(std::vector<std::string>(kept, "---"));
  for (size_t i = 1; i < grid.size(); ++i) md += line(grid[i].text);
  md.pop_back();
  return md;
}

}  // namespace hp
