#include "assemble.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <regex>

#include "core/layout/layout_types.h"
#include "core/markdown/markdown_export.h"
#include "table_grid.h"
#include "ordernet/page.h"
#include "core/pipeline/pipeline_result.h"
#include "core/pipeline/reading_order_util.h"

namespace hp {
namespace {

int pp_class_id(std::string_view name) {
  const auto &L = hp::core::layout::kLayoutLabels;
  for (size_t i = 0; i < L.size(); ++i)
    if (L[i] == name) return static_cast<int>(i);
  return 22;  // text
}

hp::core::Box quad(float x0, float y0, float x1, float y1) {
  const int a = (int)std::lround(x0), b = (int)std::lround(y0), c = (int)std::lround(x1), d = (int)std::lround(y1);
  return hp::core::Box{{{{a, b}, {c, b}, {c, d}, {a, d}}}};
}

bool center_in(const PageLine &l, const LayoutBox &b) {
  const float cx = 0.5f * (l.x0 + l.x1), cy = 0.5f * (l.y0 + l.y1);
  return cx >= b.x0 && cx <= b.x1 && cy >= b.y0 && cy <= b.y1;
}

}  // namespace

std::vector<PageLine> merge_into_lines(const std::vector<PageLine> &in) {
  // Slot inline OCR pieces into the native sequence.
  std::vector<PageLine> items;
  std::vector<const PageLine *> inl;
  for (const auto &it : in) (it.inline_ocr ? (void)inl.push_back(&it) : items.push_back(it));
  auto vov = [](const PageLine &a, const PageLine &b) {
    const float h = std::min(a.y1 - a.y0, b.y1 - b.y0);
    return h > 0 && std::min(a.y1, b.y1) - std::max(a.y0, b.y0) >= 0.5f * h;
  };
  for (const PageLine *p : inl) {
    const float h = p->y1 - p->y0;
    int best = -1;
    for (int j = 0; j < (int)items.size(); ++j)
      if (!items[j].ocr && vov(items[j], *p) && items[j].x1 <= p->x0 + 0.5f * h &&
          (best < 0 || items[j].x1 > items[best].x1))
        best = j;
    if (best < 0) {  // nothing to the left: before the nearest native item to the right
      for (int j = 0; j < (int)items.size(); ++j)
        if (!items[j].ocr && vov(items[j], *p) && items[j].x0 >= p->x1 - 0.5f * h &&
            (best < 0 || items[j].x0 < items[best].x0))
          best = j;
      if (best >= 0) { items.insert(items.begin() + best, *p); continue; }
      items.push_back(*p);
      continue;
    }
    items.insert(items.begin() + best + 1, *p);
  }
  std::vector<PageLine> out;
  for (const auto &it : items) {
    if (!out.empty()) {
      PageLine &l = out.back();
      const float h = std::min(l.y1 - l.y0, it.y1 - it.y0);
      const float ov = std::min(l.y1, it.y1) - std::max(l.y0, it.y0);
      const float gap = it.x0 - l.x1;
      const bool same_kind = (l.ocr && !l.inline_ocr) == (it.ocr && !it.inline_ocr);
      if (same_kind && h > 0 && ov >= 0.5f * h && gap > -0.5f * h && gap < 2.5f * h) {
        if (gap > 0.12f * h) l.text += ' ';
        l.text += it.text;
        l.x0 = std::min(l.x0, it.x0); l.y0 = std::min(l.y0, it.y0);
        l.x1 = std::max(l.x1, it.x1); l.y1 = std::max(l.y1, it.y1);
        l.conf = std::min(l.conf, it.conf);
        continue;
      }
    }
    out.push_back(it);
  }
  return out;
}

AssembledPage assemble_page(int page_index, int w, int h, const std::vector<LayoutBox> &layout_in,
                            const std::vector<PageLine> &items, std::vector<PageLine> lines,
                            const std::vector<PageFormula> &formulas, const std::vector<ModelTable> &model_tables,
                            on::OrderNet *ordernet, void *stream) {
  AssembledPage page;
  // ── layout in the PP vocabulary ──
  struct L { LayoutBox b; int cls; int src; };
  std::vector<L> lay;
  for (size_t i = 0; i < layout_in.size(); ++i)
    lay.push_back({layout_in[i], pp_class_id(kDfineToPP[layout_in[i].cls]), static_cast<int>(i)});

  // doc_title: on the first page, the Title whose lines are tallest (and clearly
  // taller than other titles) in the top half of the page.
  if (page_index == 0) {
    int best = -1;
    float best_h = 0, second_h = 0;
    for (size_t i = 0; i < lay.size(); ++i) {
      if (lay[i].cls != pp_class_id("paragraph_title") || lay[i].b.y0 > 0.5f * h) continue;
      float lh = 0;
      int n = 0;
      for (const auto &ln : lines)
        if (center_in(ln, lay[i].b)) { lh += ln.y1 - ln.y0; ++n; }
      if (!n) continue;
      lh /= n;
      if (lh > best_h) { second_h = best_h; best_h = lh; best = (int)i; }
      else second_h = std::max(second_h, lh);
    }
    if (best >= 0 && best_h > 1.15f * second_h) lay[best].cls = pp_class_id("doc_title");
  }

  // formula_number: an "(n)" line at the right edge of a display formula band.
  static const std::regex kNum(R"(^\s*\(\s*[0-9]+[a-z]?(\.[0-9]+)?\s*\)\s*$)");
  const int formula_cls = pp_class_id("display_formula");
  for (const auto &ln : lines) {
    if (!std::regex_match(ln.text, kNum)) continue;
    const float cy = 0.5f * (ln.y0 + ln.y1);
    for (const auto &f : lay)
      // Only numbers OUTSIDE the formula box: inside it the formula model reads "(n)" itself.
      if (f.cls == formula_cls && cy >= f.b.y0 && cy <= f.b.y1 && ln.x0 >= f.b.x1 - 2.f) {
        lay.push_back({LayoutBox{0, 1.f, ln.x0, ln.y0, ln.x1, ln.y1}, pp_class_id("formula_number"), -1});
        break;
      }
  }

  // Smallest region first: a line's layout_id is the FIRST region containing its centre.
  std::stable_sort(lay.begin(), lay.end(), [](const L &a, const L &b) {
    return (a.b.x1 - a.b.x0) * (a.b.y1 - a.b.y0) < (b.b.x1 - b.b.x0) * (b.b.y1 - b.b.y0);
  });

  hp::core::pipeline::OcrPipelineResult res;
  res.image_width = w;
  res.image_height = h;
  std::vector<int> src_to_new(layout_in.size(), -1);
  for (size_t i = 0; i < lay.size(); ++i) {
    hp::core::layout::LayoutBox b;
    b.class_id = lay[i].cls;
    b.score = lay[i].b.score;
    b.box = quad(lay[i].b.x0, lay[i].b.y0, lay[i].b.x1, lay[i].b.y1);
    res.layout.push_back(b);
    if (lay[i].src >= 0) src_to_new[lay[i].src] = static_cast<int>(i);
  }
  for (const auto &ln : lines) {
    hp::core::OCRResultItem it;
    it.text = ln.text;
    it.confidence = ln.conf;
    it.box = quad(ln.x0, ln.y0, ln.x1, ln.y1);
    it.source = ln.ocr ? "" : "pdf";
    res.results.push_back(std::move(it));
  }
  for (const auto &f : formulas) {
    if (f.layout_index < 0 || f.layout_index >= (int)src_to_new.size() || src_to_new[f.layout_index] < 0) continue;
    hp::core::router::FormulaResult r;
    r.layout_id = src_to_new[f.layout_index];
    r.latex = f.latex;
    r.score = f.score;
    r.box = res.layout[r.layout_id].box;
    res.formulas.push_back(std::move(r));
  }
  // Tables. LayoutNet's tables go to the Table region each overlaps most (at least half of the
  // smaller box; D-FINE decides where tables are, the model their structure), in reading order
  // when a region holds several. Other Table regions get a grid from their spans (table_grid.h).
  std::vector<std::vector<const ModelTable *>> by_region(layout_in.size());
  for (const ModelTable &mt : model_tables) {
    int best = -1;
    float best_ov = 0;
    for (size_t i = 0; i < layout_in.size(); ++i) {
      if (kDfineLabels[layout_in[i].cls] != "Table" || src_to_new[i] < 0) continue;
      const LayoutBox &b = layout_in[i];
      const float ix = std::min(mt.x1, b.x1) - std::max(mt.x0, b.x0), iy = std::min(mt.y1, b.y1) - std::max(mt.y0, b.y0);
      if (ix <= 0 || iy <= 0) continue;
      const float small = std::min((mt.x1 - mt.x0) * (mt.y1 - mt.y0), (b.x1 - b.x0) * (b.y1 - b.y0));
      const float ov = ix * iy / std::max(small, 1.f);
      if (ov > best_ov) best_ov = ov, best = static_cast<int>(i);
    }
    if (best >= 0 && best_ov >= 0.5f) by_region[best].push_back(&mt);
  }
  for (size_t i = 0; i < layout_in.size(); ++i) {
    if (kDfineLabels[layout_in[i].cls] != "Table" || src_to_new[i] < 0) continue;
    hp::core::router::TableResult t;
    t.layout_id = src_to_new[i];
    t.box = res.layout[t.layout_id].box;
    PageTable pt{static_cast<int>(i), {}, {}};
    if (!by_region[i].empty()) {
      auto &v = by_region[i];
      std::stable_sort(v.begin(), v.end(), [](const ModelTable *a, const ModelTable *b) { return a->y0 < b->y0; });
      for (const ModelTable *mt : v) pt.html += (pt.html.empty() ? "" : "\n") + mt->html;
      t.html = pt.html;
    } else {
      std::vector<PageLine> spans;
      for (const auto &it : items)
        if (center_in(it, layout_in[i])) spans.push_back(it);
      pt.markdown = table_markdown(std::move(spans));
      if (pt.markdown.empty()) continue;
      t.markdown = pt.markdown;
    }
    res.tables.push_back(std::move(t));
    page.tables.push_back(std::move(pt));
  }
  // OrderNet: reading order and blocks over the page's lines (ordernet/page.h). Without it, or on a
  // page it cannot take (under 2 or over 2048 lines), XY-cut over the regions and the region renderer.
  if (ordernet) {
    on::PageOrder po = on::order_page(*ordernet, static_cast<cudaStream_t>(stream), w, h, layout_in, lines, formulas,
                                      page.tables);
    if (po.ok) {
      page.markdown = std::move(po.markdown);
      page.order = std::move(po.order);
      page.lines = std::move(lines);
      page.ordered_by_model = true;
      return page;
    }
  }
  hp::core::pipeline::maybe_assign_reading_order(true, res.results, res.layout, res.reading_order);
  page.markdown = hp::core::markdown::render_markdown(res);
  page.order = res.reading_order;
  page.lines = std::move(lines);
  return page;
}

}  // namespace hp
