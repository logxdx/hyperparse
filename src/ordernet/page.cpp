#include "ordernet/page.h"

#include <algorithm>
#include <string_view>

namespace hp::on {
namespace {

// Centre of a line inside a region (edges inclusive).
bool inside(const PageLine &l, const LayoutBox &r) {
  const float cx = (l.x0 + l.x1) / 2, cy = (l.y0 + l.y1) / 2;
  return r.x0 <= cx && cx <= r.x1 && r.y0 <= cy && cy <= r.y1;
}

bool is_class(const LayoutBox &b, std::string_view name) { return kDfineLabels[b.cls] == name; }

std::string strip_ascii_ws(std::string s) {
  const auto a = s.find_first_not_of(" \t\r\n\f\v");
  if (a == std::string::npos) return {};
  return s.substr(a, s.find_last_not_of(" \t\r\n\f\v") - a + 1);
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
    regions.push_back({{{r.x0, r.y0, r.x1, r.y1}, "TBLHOLD" + std::to_string(k) + "X"}, std::move(m),
                       t.html.empty() ? t.markdown : t.html, true});
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

  const Heads heads = net.forward(line_features(in, static_cast<float>(w), static_cast<float>(h)), n, stream);
  const std::vector<int> order = decode_blocks_order(heads, in);
  std::string md = page_markdown(in, order, heads, /*keep_furniture=*/false);
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
