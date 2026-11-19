#pragma once

#include <vector>

#include "core/common/serialization/serialization.h"
#include "core/layout/order/reading_order.h"

namespace hp::core::pipeline {

// Shared assign-ids + XY-cut pair used by the GPU single/batch paths, the CPU
// pipeline, and the PDF job — one definition so the orphan-result contract
// (synthetic XY-cut entries for unmatched detections) cannot drift.
inline void maybe_assign_reading_order(bool want,
                                       std::vector<OCRResultItem> &results,
                                       std::vector<layout::LayoutBox> &layout,
                                       std::vector<int> &reading_order) {
  if (!want) return;
  // `want` IS the "caller asked for layout-aware output" signal, so a layout
  // that came back empty here means the model found nothing on a page that
  // may still be full of text — let assign_layout_ids rescue the orphans
  // rather than short-circuiting on `layout.empty()` as this used to.
  hp::core::assign_layout_ids(results, layout,
                               /*synthesise_when_empty=*/true);
  if (layout.empty()) return;  // genuinely nothing on the page
  reading_order =
      layout::assign_reading_order_for_results(results, layout);
}

} // namespace hp::core::pipeline
