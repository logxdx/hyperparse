#pragma once

#include <array>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/docassembly/parsing_block.h"
#include "core/layout/layout_types.h"

namespace hp::core::docassembly {

// Group one page's recognized items into per-region ParsingBlocks.
//
// This is the missing producer: `merge_tables_across_pages`,
// `assign_title_levels` and `truncate_repetitive_content` all operate on
// ParsingBlock, but nothing built one, so the whole document-assembly stage
// was unreachable from the request path.
//
// Grouping mirrors the serializer's `append_blocks_array`: walk
// `reading_order`, bucket each result by its `layout_id`, and emit one block
// per non-empty layout region in first-encounter (i.e. reading) order.
//
// `table_html` supplies recognized table structure keyed by layout id. A
// `table` region with an entry takes the HTML as its content, matching
// ParsingBlock's contract that `content` is "plain text for text-bound
// labels, an HTML table for `table`, LaTeX for formulas" — which is also
// what `merge_tables_across_pages` expects to splice.
//
// NOTE: the content line-joining below is a deliberate duplicate of the
// serializer's. Hoisting the serializer's version would be the tidier fix,
// but it is written as in-place `j += ...` string building against an output
// buffer and rewriting it risks perturbing an existing, test-covered
// response shape. Kept separate until that refactor is done on purpose.
[[nodiscard]] std::vector<ParsingBlock> build_page_blocks(
    const std::vector<OCRResultItem> &results,
    const std::vector<layout::LayoutBox> &layout,
    const std::vector<int> &reading_order, int page_index,
    const std::vector<std::pair<int, std::string>> &table_html = {});

// Does the layout detector expect readable text inside this class?
//
// Coverage is only meaningful where text is expected: a figure or a chart
// legitimately contains no detected text, and scoring it 0.0 would drag every
// page-level aggregate down for a region that is behaving correctly. Those
// classes report coverage -1 (not applicable) instead.
[[nodiscard]] bool class_expects_text(int class_id) noexcept;

// Fraction of `region` covered by the UNION of `boxes`, in [0, 1].
//
// Union, not the sum of areas. Detection boxes overlap — adjacent lines share
// a pixel row, and a re-detected fragment can sit inside its neighbour — and a
// sum would exceed the region area and clamp to 1.0. That fails in the
// dangerous direction: it makes an unread region look fully covered, which is
// precisely the signal this is meant to raise.
//
// Computed by coordinate compression (sort the distinct x and y edges, then
// mark the cells each box spans). O(n^2) cells for n boxes, which is a few
// thousand for a realistic region and runs once per block.
[[nodiscard]] float region_coverage(
    const std::array<int, 4> &region,
    const std::vector<std::array<int, 4>> &boxes) noexcept;

// Area-weighted mean coverage over the text-bearing blocks of a page, in
// [0, 1]; -1 when the page has no text-bearing block at all.
//
// Area-weighted rather than a flat mean so a page is judged by how much of its
// *text* was read, not by how many regions happened to be small. Computed from
// per-block coverage rather than as one page-wide union, which keeps the cost
// linear in blocks and avoids letting one enormous region dominate the
// geometry.
[[nodiscard]] float page_coverage(const std::vector<ParsingBlock> &blocks) noexcept;

// Aggregate confidence for a block built from `n` items.
//
// Length-weighted mean, NOT the minimum used by `merge_confidence`. The two
// operations are asking different questions and deserve different answers:
//
//   building — many small items make up one region, so the region's quality
//              is their average; a single badly-read word must not condemn a
//              200-word paragraph.
//   merging  — two whole regions become one, so the result is only as
//              trustworthy as its weaker half, and a short bad continuation
//              must not hide behind a long clean host.
//
// Items with an unset (< 0) confidence are skipped. Returns -1 when nothing
// scoreable was seen, preserving the "never scored" sentinel.
[[nodiscard]] float aggregate_confidence(const std::vector<OCRResultItem> &results,
                                          const std::vector<int> &members);

// Fold member provenance into one value: unanimous sources survive, any
// disagreement becomes "mixed" (see merge_source).
[[nodiscard]] std::string aggregate_source(const std::vector<OCRResultItem> &results,
                                            const std::vector<int> &members);

} // namespace hp::core::docassembly
