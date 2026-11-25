#pragma once
// A page's Markdown and reading order from OrderNet (the "v2c" assembly):
//   - D-FINE decides what is page furniture: lines in Page-header / Page-footer regions are left
//     out of the body (and of OrderNet's input);
//   - each formula region becomes one "$<latex>$" line and each table region one placeholder
//     line (the region's own lines are dropped), so OrderNet places them in the flow;
//   - OrderNet orders and groups the rest: blocks, headings, lists, display math, paragraphs;
//   - placeholders are replaced by the regions' tables (LayoutNet HTML or table_grid Markdown);
//   - region lines are blocks of their own, classed Text; OrderNet's grid classes (Table, Form,
//     Table-Of-Contents) are kept for tables D-FINE missed, except on blocks of prose lines.
// End to end (FP16 engines, RTX 4050): olmOCR-bench 55.6 vs 52.3 for XY-cut (order tests 43.5 -> 55.7,
// multi-column 60 -> 77, tables 68 -> 72.8); OmniDocBench reading-order Kendall tau 0.87 -> 0.945,
// table TEDS 0.640 -> 0.658; 1-3 ms per page on the assembly thread.

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
