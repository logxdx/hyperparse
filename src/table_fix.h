#pragma once
// Structural clean-up of emitted tables: a port of liteparse markdown_layout/table_fix.rs, applied
// to every table (LayoutNet grid or geometry grid) before it is written. Each rule fixes a pattern
// that costs whole columns or rows in table scoring even when every value was read correctly:
//
//   - currency symbols in their own cell (`$ | 14,046`) or stuck to the previous value
//     (`8.8 % $ | 13,917`) move onto the number they belong to; columns left empty are removed;
//   - a wrapped cell's continuation line emitted as its own row is folded into the row above;
//   - a group header in a multi-row header ("Fiscal 2024" over "Net Sales | % of Net Sales") spans
//     the sub-columns it is centred over instead of the first one only;
//   - dot / underscore leaders are stripped from cell text;
//   - (absorb_header_band) a header band printed just above a LayoutNet table, with the table's
//     top row partly empty, joins the table as header rows.
//
// Thresholds, rule order and float widths follow the Rust (f32 cell boxes, f64 grid boxes).

#include <optional>
#include <string>
#include <vector>

#include "tablenet/decode.h"

namespace hp::tfix {

// Rust types::Rect (f32, top-left origin).
struct Rect {
  float x = 0, y = 0, width = 0, height = 0;
};

// blocks.rs SpanCell: a cell with its spans; cells covered by a neighbour's span are absent.
struct SpanCell {
  std::string text;
  int colspan = 1, rowspan = 1;
  std::optional<Rect> bbox;
};

// blocks.rs Block::MergedTable: rows of span cells, the first `header_rows` rows are the header.
// hp: `th` flags rows (any, per row; empty = none) also written as <th>: LayoutNet marks header
// rows below the top too (a sub-header mid-table), and hp's HTML has always kept them; the rules
// see only the leading `header_rows`, as liteparse's.
struct SpanTable {
  std::vector<std::vector<SpanCell>> rows;
  int header_rows = 0;
  std::vector<char> th = {};
};

// fix_block on a Block::MergedTable.
void fix_merged(SpanTable &t);

// fix_block on a Block::Table: `t` is a plain grid (every cell 1x1) whose first row is the header
// when t.header_rows is 1 (0: no header). A plain table stays plain: when the rules would create a
// span, `t` is left as it was.
void fix_plain(SpanTable &t);

// decode.rs Grid::to_span_rows: span rows and the number of leading header rows (plus `th` for
// head rows below them).
[[nodiscard]] SpanTable span_rows(const tn::Grid &g);

// HTML in the LayoutNet writer's format (tn::Grid::to_html): one <tr> per line, <th> for the
// header rows, rowspan before colspan, Python html.escape(quote=True) text.
[[nodiscard]] std::string to_html(const SpanTable &t);

// True when the table has no spans, equal-length rows and at most one header row (GFM can hold it).
[[nodiscard]] bool is_plain(const SpanTable &t);

// A text span of the page (a text-layer item or an OCR box), page points, top-left origin.
struct BandSpan {
  std::string text;
  float x = 0, y = 0, width = 0, height = 0;
};

// A text line as absorb_header_band reads it (liteparse ProjectedLine: text, spans, page box).
struct BandLine {
  std::string text;
  std::vector<BandSpan> spans;
  Rect box;
  std::vector<int> members;  // indices of the spans it was built from
};

// Lines from loose spans: spans whose vertical extents overlap by >= half the smaller height form a
// line (top to bottom by centre); within a line, spans closer than half a line height join into one
// span (a word gap), so a span is a phrase as in a PDF text item. Spans with `skip[i]` set are left out.
[[nodiscard]] std::vector<BandLine> band_lines(const std::vector<BandSpan> &spans,
                                               const std::vector<char> &skip = {});

// table_fix.rs absorb_header_band: when the grid's top row has empty slots, the lines just above
// it whose spans fall over its columns (two or more columns, separated by column-sized gaps, not
// prose) become new header rows. `taken[i]`: line i belongs to something else. Returns the indices
// of the absorbed lines, top-most first. `full_top` (hp): also when every slot of the top row is
// filled (LayoutNet left the whole header row out), then over any columns and only lines without
// a value (is_value) in any cell; the caller decides when that is safe. With an empty top slot
// the rule is liteparse's whatever `full_top` says.
std::vector<int> absorb_header_band(tn::Grid &g, const std::vector<BandLine> &lines, const std::vector<char> &taken,
                                    bool full_top = false);

// Which passes run (all by default; the switches exist for ablations, tests/table_fix_test --cli).
struct FixOptions {
  bool band = true;       // absorb_header_band
  bool clean_up = true;   // fix_merged / fix_plain
  bool headless = true;   // band with full_top on a grid without any header row (see model_table_html)
};

// A LayoutNet table as written: the header band among `spans` (the text of the table's region,
// page points) joins the grid, then fix_merged, then HTML. `taken` (one flag per span) marks spans
// owned elsewhere; the spans of absorbed lines are marked on return.
[[nodiscard]] std::string model_table_html(tn::Grid g, const std::vector<BandSpan> &spans, std::vector<char> &taken,
                                           const FixOptions &opt = {});

// table_fix.rs helpers, exposed for the tests.
[[nodiscard]] std::string strip_leader_runs(const std::string &t);
[[nodiscard]] bool is_value(const std::string &t);

}  // namespace hp::tfix
