#pragma once

// Recursive XY-cut reading-order algorithm.
//
// Port of the canonical PaddleX reference (paddlex/inference/pipelines/
// layout_parsing/xycut_enhanced/utils.py — recursive_xy_cut and helpers
// projection_by_bboxes / split_projection_profile). The algorithm:
//
//   1. Sort layout boxes by x_min, project onto X-axis.
//   2. Split the X projection into vertical column segments.
//   3. For each column, sort by y_min, project onto Y-axis.
//   4. Split the Y projection into horizontal row segments.
//   5. If a row spans multiple sub-columns, recurse; else emit indices.
//
// Output is a permutation of the input indices in correct reading order
// (left-to-right within a line, top-to-bottom across lines, for any
// number of columns).
//
// Pure CPU, no exceptions in the hot path: degenerate input (empty
// vector, zero-area boxes, all overlapping) is handled by emitting a
// best-effort sequence rather than throwing.

#include <array>
#include <string_view>
#include <vector>

#include "core/common/geometry/box.h"
#include "core/common/types.h"
#include "core/layout/layout_types.h"

namespace hp::core::layout {

// Page-level mean text-line width/height across body-text cells, as consumed
// by weighted_distance_insert (doc_title disperse term) and
// detect_child_blocks (cross-cell proximity threshold). Both are 0 when no
// body cell carried any clustered text line.
//
// "Body text" is class 22 (`text`) AND class 23 (`vertical_text`). Extracted
// from assign_reading_order_for_results so this can be asserted directly:
// OmniDocBench contains no vertical page that also carries a title or figure,
// so the class-23 half of this is unreachable from the corpus and a unit test
// is the only way to cover it.
struct PageTextLineMetrics {
  int width = 0;
  int height = 0;
};
[[nodiscard]] PageTextLineMetrics
page_text_line_metrics(const std::vector<LayoutBox> &layout) noexcept;

// Parse a READING_ORDER_INLINE spec into a bitmask of OrderLabel values that
// should stay inside the geometric XY-cut instead of being re-inserted by
// match_unsorted_blocks. Bit N corresponds to OrderLabel value N.
//
// Tokens (comma-, space- or semicolon-separated, case-insensitive):
//   title      kDocTitle + kParagraphTitle
//   vision     kVision + kVisionTitle
//   xref       kCrossReference
//   unordered  kUnordered
//   all        all of the above  (the shipped default)
//   none/off   clear the mask — only kBody is cut geometrically
// Unknown tokens are ignored; an empty spec yields 0. `none` anywhere in the
// spec wins over every other token. kBody is always cut and never appears
// here.
//
// Exposed for tests: the live value is read from the environment once per
// process, so the parser is the only part worth asserting on.
[[nodiscard]] unsigned parse_inline_label_spec(std::string_view spec) noexcept;

// Half-open segment of a 1D projection profile: [start, end).
struct ProjectionSegment {
  int start = 0;
  int end = 0;
};

// 1D projection histogram of N axis-aligned rects along `axis`
// (0 = X projection, summing over horizontal extents;
//  1 = Y projection, summing over vertical extents).
//
// Each rect is given as [x0, y0, x1, y1] in original-image coords.
// The histogram length spans [0, max_extent_along_axis); rects with
// negative coordinates are treated as |coord| (matching the PaddleX
// reference which mirrors right-to-left layouts onto the positive axis).
[[nodiscard]] std::vector<int>
projection_by_bboxes(const std::vector<std::array<int, 4>> &rects, int axis);

// Split a projection profile into contiguous segments where the value
// strictly exceeds `min_value`. A gap of more than `min_gap` zero-bins
// between two non-zero runs starts a new segment.
[[nodiscard]] std::vector<ProjectionSegment>
split_projection_profile(const std::vector<int> &projection, int min_value,
                         int min_gap);

// Recursive XY-cut on the rects identified by `indices` (positions into
// the caller's rects vector). Appends the reading-order permutation to
// `res`. `min_gap` controls the minimum vertical gap (in pixels) between
// row segments after the column split.
//
// `rects` is taken by const-ref so callers can compute the AABB list
// once and reuse it across recursive frames.
void recursive_xy_cut(const std::vector<std::array<int, 4>> &rects,
                      const std::vector<int> &indices,
                      std::vector<int> &res, int min_gap = 1);

// Top-level entry point. Computes axis-aligned rects from the layout
// boxes' 4-corner quads and runs recursive_xy_cut. Returns the
// reading-order permutation of the input indices: result[k] is the
// index (into `layout`) of the k-th item to read.
//
// Empty input → empty output. A single box → single-element output.
// Boxes with zero area are still included (they'd otherwise vanish from
// the response without explanation).
[[nodiscard]] std::vector<int>
assign_reading_order(const std::vector<LayoutBox> &layout, int min_gap = 1);

// Reading-order permutation expressed as indices into `results` (the JSON
// contract emits this directly under "reading_order"). Computes the
// layout-level XY-cut order, then groups results by `layout_id` and emits
// them in that group order. Within a single layout region (and for any
// results whose layout_id is -1, e.g. when layout matching missed) ties
// are broken by box-center y, then x — so results inside one region come
// out top-to-bottom, left-to-right.
//
// Caller MUST run assign_layout_ids() before calling this so each result
// has a valid layout_id; otherwise everything degrades to the y/x
// fallback. Empty layout → indices [0..results.size()) sorted purely by
// y/x. Empty results → empty output.
//
// `layout` is taken by non-const ref because the function runs
// cluster_text_lines internally as a pre-pass — populating each
// LayoutBox's direction / num_of_lines / text_line_height /
// text_line_width / seg_*_coordinate fields. None of those fields are
// serialised so the JSON output is unaffected, but downstream code
// that re-uses `layout` will see the populated values.
[[nodiscard]] std::vector<int>
assign_reading_order_for_results(const std::vector<OCRResultItem> &results,
                                 std::vector<LayoutBox> &layout,
                                 int min_gap = 1);

} // namespace hp::core::layout
