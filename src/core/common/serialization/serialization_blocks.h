#pragma once

#include "core/docassembly/block_builder.h"
#include "core/common/text_flow.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/common/geometry/box.h"
#include "core/common/serialization/serialization_primitives.h"
#include "core/common/types.h"
#include "core/layout/layout_types.h"

namespace hp::core {

namespace detail {

// Append `"blocks":[ ... ]` — paragraph-level aggregate, one entry per
// non-empty layout cell, in reading order. Mirrors the granularity
// PaddleX's PP-StructureV3 emits via parsing_res_list (one LayoutBlock
// per layout region with joined `content`).
//
// Each entry: {id, layout_id, class, bounding_box, content, order_index}
// `content` joins child texts with ' ' on the same line and '\n' on a
// new line. Line boundaries are detected from y-jump > text_line_height
// (or 8px when text_line_height is 0). The first non-empty layout cell
// in reading order gets order_index=0.
inline void append_blocks_array(std::string &j,
                                 const std::vector<OCRResultItem> &results,
                                 const std::vector<layout::LayoutBox> &layout,
                                 const std::vector<int> &reading_order) {
  j += "\"blocks\":[";
  if (reading_order.empty() || layout.empty()) {
    j += ']';
    return;
  }

  // Group results in reading order by layout_id, preserving first-
  // encounter order so the emitted blocks are themselves in reading
  // order. We walk reading_order and bucket each result.
  std::vector<int> block_order_of_layout(layout.size(), -1);
  std::vector<std::vector<int>> grouped(layout.size());
  int next_block_idx = 0;
  for (int ri : reading_order) {
    if (ri < 0 || static_cast<size_t>(ri) >= results.size()) continue;
    const int lid = results[ri].layout_id;
    if (lid < 0 || static_cast<size_t>(lid) >= layout.size()) continue;
    if (block_order_of_layout[lid] < 0) {
      block_order_of_layout[lid] = next_block_idx++;
    }
    grouped[lid].push_back(ri);
  }

  // Iterate layouts in their reading-order rank.
  std::vector<int> layout_emit_order(layout.size(), -1);
  for (size_t li = 0; li < layout.size(); ++li) {
    if (block_order_of_layout[li] >= 0)
      layout_emit_order[block_order_of_layout[li]] = static_cast<int>(li);
  }

  bool first = true;
  for (int oi = 0; oi < next_block_idx; ++oi) {
    const int li = layout_emit_order[oi];
    if (li < 0) continue;
    const auto &cell = layout[li];
    const auto &members = grouped[li];
    if (members.empty()) continue;

    if (!first) j += ',';
    first = false;
    j += '{';
    j += "\"id\":";
    j += std::to_string(oi);
    j += ",\"layout_id\":";
    j += std::to_string(li);
    j += ",\"class\":\"";
    auto name = layout::label_name(cell.class_id);
    for (char c : name) j += c;
    j += "\",\"bounding_box\":";
    append_box(j, cell.box);
    j += ",\"content\":\"";

    // Build the content with smart line-joining:
    // Row grouping and the space-vs-newline rule are text_flow.h's, shared
    // with the docassembly path. Both were written out longhand in the two
    // places, which is exactly how they drifted.
    auto [cell_x0, cell_y0, cell_x1, cell_y1] = hp::core::aabb(cell.box);
    (void)cell_x0;
    (void)cell_y0;
    (void)cell_y1;
    std::vector<hp::core::TextSpan> spans;
    spans.reserve(members.size());
    for (size_t mi = 0; mi < members.size(); ++mi) {
      const auto &it = results[members[mi]];
      int sy = 0, x0 = INT_MAX, x1 = INT_MIN;
      for (int k = 0; k < 4; ++k) {
        sy += it.box[k][1];
        x0 = std::min(x0, it.box[k][0]);
        x1 = std::max(x1, it.box[k][0]);
      }
      spans.push_back({sy / 4, x0, x1});
    }
    const auto rows =
        hp::core::plan_rows(spans, cell_x1, cell.text_line_height);
    bool first_row = true;
    for (size_t r = 0; r < rows.size(); ++r) {
      if (rows[r].continuation) continue;  // rendered by its host row
      if (!first_row) {
        if (rows[r].join == hp::core::RowJoin::kSpace) j += ' ';
        else                                            j += "\\n";
      }
      first_row = false;
      for (int i = rows[r].begin; i < rows[r].end; ++i) {
        if (i > rows[r].begin) j += ' ';
        // Same JSON escaping + UTF-8 backstop as every other text field.
        append_escaped_string(j, results[members[static_cast<size_t>(i)]].text);
        if (i == rows[r].begin)
          for (size_t c = r + 1; c < rows.size() && rows[c].continuation; ++c)
            for (int m = rows[c].begin; m < rows[c].end; ++m) {
              j += ' ';
              append_escaped_string(
                  j, results[members[static_cast<size_t>(m)]].text);
            }
      }
    }
    j += "\",\"order_index\":";
    j += std::to_string(oi);
    // Detection coverage for this region — the fraction of its area that
    // detection boxes actually cover, as a union (not a sum, which would
    // over-count overlaps and make an unread region look read).
    //
    // Confidence alone cannot see the failure this catches: it averages over
    // the text that WAS read, so a region whose text was never detected has
    // nothing to average and scores unremarkably. Measured on `dense_text`:
    // confidence 0.706, page coverage 0.307, worst block 0.027 — against
    // 0.976-0.981 / 0.77-0.86 on clean fixtures. Emitted for text-bearing classes
    // only, and emitted even at 0.0 — a text region with no detections is
    // precisely the case worth reporting.
    // Length-weighted mean recognition confidence over this block's members.
    // Emitted so a consumer — and the semantic layer's per-field provenance —
    // has a confidence to attribute; without it every extracted field reports
    // an unknown confidence even though the recognizer scored every line.
    {
      const float bc = docassembly::aggregate_confidence(results, members);
      if (bc >= 0.0f) {
        j += ",\"confidence\":";
        append_score(j, bc);
      }
    }
    if (docassembly::class_expects_text(cell.class_id)) {
      std::vector<std::array<int, 4>> boxes;
      boxes.reserve(members.size());
      for (int mi : members) {
        const auto [bx0, by0, bx1, by1] = hp::core::aabb(results[mi].box);
        boxes.push_back({bx0, by0, bx1, by1});
      }
      const float cov = docassembly::region_coverage(
          {cell_x0, cell_y0, cell_x1, cell_y1}, boxes);
      if (cov >= 0.0f) {
        j += ",\"coverage\":";
        append_score(j, cov);
      }
    }
    j += '}';
  }
  j += ']';
}

} // namespace detail

} // namespace hp::core
