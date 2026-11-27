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
  double det_ms = 0, rec_ms = 0;  // GPU-thread time in the det and rec phases (launch + wait)
};

// One page's OCR, run in three phases on models.ocr_stream() so that the GPU
// thread can overlap them with other pages (page_pipeline.cpp):
//   launch_det   after the page's resize launch: det queued behind `lane.resized`;
//   launch_rec   waits for det, filters the boxes, queues rec on the page buffer;
//   finish_ocr   waits for rec, decodes, applies the confidence filter.
// det and rec each have one set of buffers: a page's launch_rec must come before
// the next page's launch_det input is written (its resize launch), and its
// finish_ocr before the next page's launch_rec. The lane's page buffer must stay
// untouched until finish_ocr.
struct OcrJob {
  OcrPlan plan;
  PageGeometry g;
  const uint8_t *page = nullptr;  // lane page buffer (u8 BGR)
  std::vector<hp::core::Box> keep;  // boxes sent to rec: detected first, then text-layer-guided
  size_t n_detected = 0;
  OcrPageResult out;
};

[[nodiscard]] OcrPlan plan_ocr(ModelSet &models, const Route &rt, const PageGeometry &g, bool ocr_enabled);

void launch_det(ModelSet &models, LayoutLane &lane, OcrJob &job);
void launch_rec(ModelSet &models, OcrJob &job, const PageText &pt, const Route &rt);
// Returns the page's OCR result (moved out of `job`).
[[nodiscard]] OcrPageResult finish_ocr(ModelSet &models, OcrJob &job);

}  // namespace hp
