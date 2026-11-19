#pragma once
// Table structure from geometry: the text spans inside a Table layout region
// (text-layer items and OCR boxes, before line merging) laid out as a grid and
// written as a Markdown pipe table. No table-structure model is involved.
//
//   rows     spans whose vertical extents overlap by >= half the smaller height;
//            a row that only continues cells of the row above, at line spacing,
//            (gap < 0.6 line heights) is a wrapped cell and joins that row
//   columns  gaps in the x-projection that every multi-cell row agrees on
//   cells    spans in the same row and column, joined left to right
// Anything that does not come out as >= 2 rows x >= 2 columns, with most rows
// holding several cells, is not reported as a table.

#include <string>
#include <vector>

#include "assemble.h"

namespace hp {

// Markdown pipe table (first row as the header), or "" when the spans do not
// form a grid.
[[nodiscard]] std::string table_markdown(std::vector<PageLine> spans);

}  // namespace hp
