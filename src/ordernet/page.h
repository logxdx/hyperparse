#pragma once
// A page's Markdown and reading order from OrderNet (the "v2c" assembly):
//   - D-FINE decides what is page furniture: lines in Page-header / Page-footer regions are left
//     out of the body (and of OrderNet's input);
//   - each formula region becomes one "$<latex>$" line and each table region one placeholder
//     line (the region's own lines are dropped), so OrderNet places them in the flow;
//   - OrderNet orders and groups the rest: blocks, headings, lists, display math, paragraphs;
//   - placeholders are replaced by the regions' tables (LayoutNet HTML or table_grid Markdown).
// On olmOCR-bench this scored 54.1 vs 52.1 for XY-cut (order tests 43.5 -> 52.2, multi-column
// 60 -> 72); on OmniDocBench reading-order Kendall tau 0.88 -> 0.95.

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "assemble.h"
#include "dfine_layout.h"
#include "ordernet/net.h"

namespace hp::on {

struct PageOrder {
  bool ok = false;           // false: too few / too many lines; keep the geometric path
  std::string markdown;
  std::vector<int> order;    // reading order over `lines` (furniture: headers first, footers last)
};

PageOrder order_page(OrderNet &net, cudaStream_t stream, int w, int h, const std::vector<LayoutBox> &layout,
                     const std::vector<PageLine> &lines, const std::vector<PageFormula> &formulas,
                     const std::vector<PageTable> &tables);

}  // namespace hp::on
