#include "ordernet/page.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <set>
#include <string>
#include <string_view>

#include "style_md.h"
#include "tablenet/text.h"

namespace hp::on {
namespace {

// Centre of a line inside a region (edges inclusive).
bool inside(const PageLine &l, const LayoutBox &r) {
  const float cx = (l.x0 + l.x1) / 2, cy = (l.y0 + l.y1) / 2;
  return r.x0 <= cx && cx <= r.x1 && r.y0 <= cy && cy <= r.y1;
}

bool is_class(const LayoutBox &b, std::string_view name) { return kDfineLabels[b.cls] == name; }

// A line of running text: >= 25 characters, >= 4 spaces, and lower-case letters for most of its
// letters (table cells are short, numeric or capitalised).
bool is_prose_line(const std::string &text) {
  const std::u32string t = tn::utf32(text);
  if (t.size() < 25) return false;
  int spaces = 0, letters = 0, lower = 0;
  for (char32_t c : t) {
    spaces += c == U' ';
    if (tn::is_alphabetic(c)) {
      ++letters;
      lower += tn::is_lowercase(c);
    }
  }
  return spaces >= 4 && letters > 0 && lower * 10 >= letters * 6;
}

std::string strip_ascii_ws(std::string s) {
  const auto a = s.find_first_not_of(" \t\r\n\f\v");
  if (a == std::string::npos) return {};
  return s.substr(a, s.find_last_not_of(" \t\r\n\f\v") - a + 1);
}

// Lower-cased words (runs of ASCII alphanumerics and non-ASCII bytes) of `s`; with html, text
// inside <...> tags is skipped.
std::set<std::string> words_of(const std::string &s, bool html) {
  std::set<std::string> out;
  std::string cur;
  bool tag = false;
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    if (html && c == '<') tag = true;
    if (!tag && (std::isalnum(c) || c >= 0x80)) {
      cur += static_cast<char>(std::tolower(c));
      continue;
    }
    if (html && c == '>') tag = false;
    if (!cur.empty()) out.insert(std::move(cur)), cur.clear();
  }
  if (!cur.empty()) out.insert(std::move(cur));
  return out;
}

// Lines of `idx` top to bottom, left to right.
void sort_geometric(std::vector<int> &idx, const std::vector<PageLine> &lines) {
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    return lines[a].y0 != lines[b].y0 ? lines[a].y0 < lines[b].y0 : lines[a].x0 < lines[b].x0;
  });
}

}  // namespace

PageOrder order_page(OrderNet &net, cudaStream_t stream, int w, int h, const std::vector<LayoutBox> &layout,
                     const std::vector<PageLine> &lines, const std::vector<PageFormula> &formulas,
                     const std::vector<PageTable> &tables) {
  PageOrder out;
  const int n_lines = static_cast<int>(lines.size());
  // What each page line becomes: kept (its index in `in`), furniture, or part of a region.
  std::vector<char> dropped(n_lines, 0);
  std::vector<int> header, footer;
  for (const LayoutBox &r : layout) {
    const bool head = is_class(r, "Page-header"), foot = is_class(r, "Page-footer");
    if (!head && !foot) continue;
    for (int i = 0; i < n_lines; ++i)
      if (!dropped[i] && inside(lines[i], r)) {
        dropped[i] = 1;
        (head ? header : footer).push_back(i);
      }
  }
  // Region lines (formula / table): one input line each, standing for the region's own lines.
  struct Region {
    Line line;
    std::vector<int> members;
    std::string payload;  // table HTML / Markdown that replaces the placeholder
    bool table = false;
  };
  std::vector<Region> regions;
  auto take_region = [&](const LayoutBox &r) {
    std::vector<int> m;
    for (int i = 0; i < n_lines; ++i)
      if (dropped[i] != 2 && inside(lines[i], r)) {
        if (!dropped[i]) m.push_back(i);
        dropped[i] = 2;
      }
    sort_geometric(m, lines);
    return m;
  };
  std::vector<std::pair<std::array<float, 4>, std::string>> formula_lines;
  for (const PageFormula &f : formulas) {
    if (f.layout_index < 0 || f.layout_index >= static_cast<int>(layout.size())) continue;
    const LayoutBox &r = layout[f.layout_index];
    std::vector<int> m = take_region(r);
    std::string latex = f.latex;
    std::erase(latex, '$');
    latex = strip_ascii_ws(latex);
    if (latex.empty()) continue;  // the region's lines stay dropped, as in the evaluated setup
    regions.push_back({{{r.x0, r.y0, r.x1, r.y1}, "$" + latex + "$"}, std::move(m), {}, false});
  }
  for (size_t k = 0; k < tables.size(); ++k) {
    const PageTable &t = tables[k];
    if (t.layout_index < 0 || t.layout_index >= static_cast<int>(layout.size())) continue;
    const LayoutBox &r = layout[t.layout_index];
    std::vector<int> m = take_region(r);
    const std::string &payload = t.html.empty() ? t.markdown : t.html;
    // A line of the region the table does not contain (LayoutNet left out a header row, or put
    // together only part of the box) stays a page line instead of being lost with the region:
    // under half of its words appear in the table.
    const std::set<std::string> in_table = words_of(payload, /*html=*/true);
    std::erase_if(m, [&](int i) {
      const std::set<std::string> w = words_of(lines[i].text, false);
      if (w.empty()) return false;
      size_t hit = 0;
      for (const auto &x : w) hit += in_table.count(x);
      if (hit * 2 >= w.size()) return false;
      dropped[i] = 0;
      return true;
    });
    regions.push_back({{{r.x0, r.y0, r.x1, r.y1}, "TBLHOLD" + std::to_string(k) + "X"}, std::move(m), payload, true});
  }

  // OrderNet input: kept lines in page order, then the region lines.
  std::vector<Line> in;
  std::vector<int> src;  // in[j] -> page line index, or -(1 + region index)
  for (int i = 0; i < n_lines; ++i)
    if (!dropped[i]) {
      in.push_back({{lines[i].x0, lines[i].y0, lines[i].x1, lines[i].y1}, lines[i].text});
      src.push_back(i);
    }
  for (size_t k = 0; k < regions.size(); ++k) {
    in.push_back(regions[k].line);
    src.push_back(-1 - static_cast<int>(k));
  }
  const int n = static_cast<int>(in.size());
  if (n < 2 || n > kMaxLines) return out;

  Heads heads = net.forward(line_features(in, static_cast<float>(w), static_cast<float>(h)), n, stream);
  // A region line (formula, table) is a block of its own and plain Text: grouped with other lines,
  // a Table block would rebuild a grid around the placeholder (and nest the real table in a cell),
  // and a furniture class would drop it. Only the block structure changes; the order head stays.
  const int n_kept = n - static_cast<int>(regions.size());
  for (int j = n_kept; j < n; ++j) {
    for (int i = 0; i < n; ++i)
      if (i != j) heads.same[static_cast<size_t>(i) * n + j] = heads.same[static_cast<size_t>(j) * n + i] = -50.f;
    for (int c = 0; c < kClasses; ++c) heads.cls[static_cast<size_t>(j) * kClasses + c] = c == 0 ? 50.f : -50.f;
  }
  // OrderNet's grid classes (Table, Table-Of-Contents, Form) on the remaining lines: a block of them
  // is read row by row and rebuilt as a geometric table. That finds tables D-FINE missed or cut
  // (olmOCR table tests 68 -> 72.8, OmniDocBench TEDS 0.640 -> 0.658), but dense multi-column prose
  // read as a table interleaves its columns (long_tiny_text 79 -> 67). A grid block whose lines are
  // mostly prose (>= 25 characters, >= 4 spaces, mostly lower-case letters) reads as text instead.
  {
    const std::vector<int> first = decode_blocks_order(heads, in);
    bool changed = false;
    for (const std::vector<int> &blk : split_blocks(first, heads)) {
      std::array<float, kClasses> sum{};
      for (int i : blk)
        for (int c = 0; c < kClasses; ++c) sum[c] += heads.cls[static_cast<size_t>(i) * kClasses + c];
      const int cls = static_cast<int>(std::max_element(sum.begin(), sum.end()) - sum.begin());
      if (cls != 5 && cls != 8 && cls != 9) continue;
      int prose = 0;
      for (int i : blk) prose += is_prose_line(in[i].text);
      if (prose * 2 < static_cast<int>(blk.size())) continue;
      for (int i : blk)
        for (int c : {5, 8, 9}) heads.cls[static_cast<size_t>(i) * kClasses + c] = -50.f;
      changed = true;
    }
    (void)changed;
  }
  const std::vector<int> order = decode_blocks_order(heads, in);
  // Styled Markdown for the page's own lines (OrderNet and the grid rebuild saw plain text).
  std::vector<std::string> styled(in.size());
  for (int j = 0; j < n; ++j) styled[j] = src[j] >= 0 ? styled_markdown(lines[src[j]]) : in[j].text;
  std::string md = page_markdown(in, order, heads, /*keep_furniture=*/false, &styled);
  merge_adjacent_emphasis(md);
  for (const Region &r : regions) {
    if (!r.table) continue;
    const std::string &ph = r.line.text, with = "\n\n" + r.payload + "\n\n";
    for (size_t p = md.find(ph); p != std::string::npos; p = md.find(ph, p + with.size())) md.replace(p, ph.size(), with);
  }
  out.markdown = std::move(md);

  // Reading order over the page's lines: headers, the ordered body (a region line expands to
  // its lines), footers.
  sort_geometric(header, lines);
  sort_geometric(footer, lines);
  out.order = header;
  for (int j : order) {
    if (src[j] >= 0) out.order.push_back(src[j]);
    else {
      const auto &m = regions[-1 - src[j]].members;
      out.order.insert(out.order.end(), m.begin(), m.end());
    }
  }
  out.order.insert(out.order.end(), footer.begin(), footer.end());
  out.ok = true;
  return out;
}

}  // namespace hp::on
