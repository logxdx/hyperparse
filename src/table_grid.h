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
//   header   a header wrapped over several rows (stack_header_lines) becomes one row
//   clean-up table_fix.h on a plain table whose first row is the header
// Anything that does not come out as >= 2 rows x >= 2 columns, with most rows
// holding several cells, is not reported as a table.

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "assemble.h"
#include "table_fix.h"

namespace hp {

// Which passes run (all by default; the switches exist for ablations, tests/table_fix_test --cli).
struct GridOptions {
  bool stack_header = true;  // stack_header_lines
  bool clean_up = true;      // tfix::fix_plain
};

// Markdown pipe table (first row as the header), or "" when the spans do not
// form a grid. (HTML only if the clean-up ever leaves spans or several header rows,
// which it does not for a plain one-header table.)
[[nodiscard]] std::string table_markdown(std::vector<PageLine> spans, const GridOptions &opt = {});

// The writing half of table_markdown: rows x columns of cell text and boxes (page px), each row's
// vertical band, the median span height `h` -> stack_header_lines, tfix::fix_plain, Markdown.
[[nodiscard]] std::string write_grid(std::vector<std::vector<std::string>> text,
                                     std::vector<std::vector<std::optional<tfix::Rect>>> box,
                                     const std::vector<std::pair<float, float>> &band, float h,
                                     const GridOptions &opt = {});

// A header written one line per row (vertically centred labels, wrapped column titles): the
// leading rows before the first row that holds a value (tfix::is_value) or repeats a body cell's
// text in the same column (a data row without a number), when there are 2..kMaxHeaderLines of
// them and a body follows, each starting within 0.6 line heights `h` of the previous one's
// bottom (`band`: each row's y0, y1). Each column's lines join top to bottom
// into row 0 ("Co-" + "Pay" -> "Co-Pay"); the folded rows are removed (and their boxes merged
// when `box` is given). Pipe tables have exactly one header row, and the scorers key every value
// by its column's header text. Returns whether it stacked anything.
constexpr size_t kMaxHeaderLines = 8;
bool stack_header_lines(std::vector<std::vector<std::string>> &text,
                        std::vector<std::vector<std::optional<tfix::Rect>>> *box,
                        const std::vector<std::pair<float, float>> &band, float h);

}  // namespace hp
