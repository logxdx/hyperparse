#pragma once
// Page assembly: layout boxes + text lines (native and OCR) + formula LaTeX ->
// reading order + Markdown, via the XY-cut reading order (core/layout) and
// Markdown renderer. hp adds what D-FINE's 12-class layout lacks for papers:
// native word items merged into lines, doc_title promotion (largest Title on the
// first page) and formula_number boxes synthesised from "(n)" lines beside
// display formulas (core/markdown folds those into \tag{n}).

#include <cstdint>
#include <string>
#include <vector>

#include "dfine_layout.h"

namespace hp::on {
class OrderNet;
}

namespace hp {

// A byte range of a line's text set in one inline style (StyleBit set, text_layer.h).
struct StyleRun {
  uint32_t b, e;
  uint8_t style;
};

struct PageLine {
  std::string text;
  float x0, y0, x1, y1;  // page px
  float conf = 1.f;
  bool ocr = false;
  bool inline_ocr = false;  // read from pixels at a text-layer box: belongs inside a native line
  std::vector<StyleRun> styles = {};  // native text only; the Markdown renders them (style_md.h)
};

struct PageFormula {
  int layout_index;  // index into the layout vector passed to assemble_page
  std::string latex;
  float score = 1.f;
};

struct PageTable {
  int layout_index;      // index into the layout vector passed to assemble_page
  std::string markdown;  // pipe table (table_grid.h), when no model table
  std::string html;      // LayoutNet's table(s) in this region (colspan / rowspan, <th> header rows)
};

// A table LayoutNet found on the page: its box (page px) and HTML.
struct ModelTable {
  float x0, y0, x1, y1;
  std::string html;
};

struct AssembledPage {
  std::string markdown;
  std::vector<PageTable> tables;
  std::vector<int> order;  // reading order over `lines` (after line merging)
  std::vector<PageLine> lines;
  bool ordered_by_model = false;  // OrderNet decided order and blocks
};

// Merge word/phrase items (extraction order) into visual lines. inline_ocr items are
// first slotted in after their left neighbour on the same line.
std::vector<PageLine> merge_into_lines(const std::vector<PageLine> &items);

// `items` are the spans before merge_into_lines (table cells come from them);
// `lines` are the merged lines.
// `model_tables`: LayoutNet's tables; each goes to the Table region it overlaps most (at least
// half of the smaller box). A Table region without one falls back to table_grid.
AssembledPage assemble_page(int page_index, int w, int h, const std::vector<LayoutBox> &layout,
                            const std::vector<PageLine> &items, std::vector<PageLine> lines,
                            const std::vector<PageFormula> &formulas,
                            const std::vector<ModelTable> &model_tables = {},
                            on::OrderNet *ordernet = nullptr, void *stream = nullptr);

}  // namespace hp
