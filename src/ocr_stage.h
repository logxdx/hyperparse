#pragma once
// OCR for one page: text detection -> filtering against the text layer -> line
// recognition. Only the parts of the page the router marked are read.
//
//   full-OCR page    det on the whole page (input written by multi_resize).
//   partial page     det on the OCR regions only, shelf-packed into one small
//                    canvas (region_pack.h) when that is cheaper than the page;
//                    plus text-layer boxes whose text was unusable go straight
//                    to rec at their own box, with no det.
//   detections       kept only inside OCR regions (partial) and never over
//                    native text: native wins on overlap (liteparse rule).
//   rec results      with confidence <= 0.1 dropped (liteparse rule).

#include <vector>

#include "assemble.h"
#include "model_set.h"
#include "region_pack.h"
#include "text_layer.h"

namespace hp {

// Decided before the page's resize launch, because full-page det needs its
// input written by the same multi_resize kernel as the layout input.
struct OcrPlan {
  bool ocr = false;         // any OCR work on this page
  bool need_det = false;    // run det at all (a partial page may only have rec boxes)
  bool region_det = false;  // det on packed regions instead of the full page
  RegionPack pack;
};

struct PageGeometry {
  int w = 0, h = 0;  // rendered page px
  float dpi = 72;
  [[nodiscard]] float px_per_pt() const { return dpi / 72.f; }
};

struct OcrPageResult {
  std::vector<PageLine> lines;
  int det_boxes = 0, rec_boxes = 0, direct_rec_boxes = 0;
  double canvas_px = 0;      // region det canvas area
  double det_ms = 0, rec_ms = 0;
};

[[nodiscard]] OcrPlan plan_ocr(ModelSet &models, const Route &rt, const PageGeometry &g, bool ocr_enabled);

// Runs on models.ocr_stream(), after `lane.resized`. Synchronizes the OCR stream.
[[nodiscard]] OcrPageResult run_ocr(ModelSet &models, LayoutLane &lane, const OcrPlan &plan, const PageText &pt,
                                    const Route &rt, const PageGeometry &g);

}  // namespace hp
