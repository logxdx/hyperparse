#include "ocr_stage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "core/common/cuda/cuda_check.h"
#include "script.h"

namespace hp {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// liteparse drops OCR lines at or below this confidence.
constexpr float kMinRecConfidence = 0.1f;

// Text-layer-guided reads (unusable glyphs re-read from pixels at their own
// box) on a page whose usable text layer holds real text and no CJK exclude the
// CJK classes (script.h): such a glyph sits inside Latin text, and a CJK read of
// it is a misread math glyph, fraction bar or bracket. Detected text (images,
// OCR regions, scans) always keeps the full dictionary, so CJK in a figure of
// an English page is still read.
constexpr int kMinCharsForScript = 100;

bool non_cjk_page(const PageText &pt) {
  int chars = 0;
  for (const TextItem &it : pt.items) {
    if (it.unusable || it.dropped) continue;
    for (size_t i = 0; i < it.text.size();) {
      const char32_t c = next_codepoint(it.text, i);
      if (is_cjk(c)) return false;
      if (c > U' ') ++chars;
    }
  }
  return chars >= kMinCharsForScript;
}
// Page px of context kept around each OCR region when packing.
constexpr int kRegionMargin = 12;
// Pack only if the canvas is < 60% of the full-page det input. Heuristic, not
// tuned: above it, the saving in det pixels is small next to the extra crop
// launch and the risk of splitting text at region edges.
constexpr double kMaxPackedShare = 0.6;

bool center_in_regions(const Route &rt, const RectF &b) {
  const float cx = 0.5f * (b.x0 + b.x1), cy = 0.5f * (b.y0 + b.y1);
  for (const auto &r : rt.regions)
    if (cx >= r.x0 - 2 && cx <= r.x1 + 2 && cy >= r.y0 - 2 && cy <= r.y1 + 2) return true;
  return false;
}

// Pack a partial page's OCR regions for region-only det. False when a full-page
// det is cheaper (regions cover most of the page) or the canvas would not fit.
bool plan_region_det(const Route &rt, const PageGeometry &g, const TextDet &det, RegionPack &pack) {
  if (rt.kind != kPartial || rt.regions.empty()) return false;
  const float k = g.px_per_pt(), s = det.page_scale(g.w, g.h);
  std::vector<PackedRegion> rects;
  for (const auto &r : rt.regions) {
    const int x0 = std::max(0, static_cast<int>(r.x0 * k) - kRegionMargin);
    const int y0 = std::max(0, static_cast<int>(r.y0 * k) - kRegionMargin);
    const int x1 = std::min(g.w, static_cast<int>(r.x1 * k) + kRegionMargin + 1);
    const int y1 = std::min(g.h, static_cast<int>(r.y1 * k) + kRegionMargin + 1);
    if (x1 - x0 >= 4 && y1 - y0 >= 4) rects.push_back({x0, y0, x1 - x0, y1 - y0, 0, 0, 0, 0});
  }
  if (rects.empty()) return false;
  const int full_w = std::max(32, static_cast<int>(g.w * s + 0.5f));
  const int full_h = std::max(32, static_cast<int>(g.h * s + 0.5f));
  pack.pack(rects, s, std::min(det.max_side(), full_w));
  return pack.canvas_w <= det.max_side() && pack.canvas_h <= det.max_side() &&
         static_cast<double>(pack.canvas_w) * pack.canvas_h < kMaxPackedShare * full_w * full_h;
}

// Det on the packed canvas: write the canvas and queue det (no wait).
void launch_region_det(ModelSet &m, LayoutLane &lane, const RegionPack &pack, const PageGeometry &g) {
  TextDet &det = m.det();
  const cudaStream_t s = m.ocr_stream();
  float *in = det.canvas_input(pack.canvas_w, pack.canvas_h);
  CUDA_CHECK(cudaMemsetAsync(in, 0, sizeof(float) * 3 * pack.canvas_w * pack.canvas_h, s));
  CropBatch &crops = m.det_crops();
  crops.clear();
  int pw = 0, ph = 0;
  for (const auto &r : pack.regions)
    crops.add_det_region(r.x, r.y, r.w, r.h, static_cast<float>(r.pw) / r.w, in, pack.canvas_w, pack.canvas_h, r.ox,
                         r.oy, pw, ph);
  crops.launch(lane.page.get(), static_cast<size_t>(g.w) * 3, g.w, g.h, s);
  det.enqueue(s);
}

// Canvas boxes of a region det mapped back to page px.
std::vector<hp::core::Box> region_boxes_to_page(const std::vector<hp::core::Box> &canvas_boxes, const RegionPack &pack,
                                                const PageGeometry &g) {
  std::vector<hp::core::Box> boxes;
  for (const auto &q : canvas_boxes) {
    float cx = 0, cy = 0;
    for (int k = 0; k < 4; ++k) {
      cx += q[k][0] * 0.25f;
      cy += q[k][1] * 0.25f;
    }
    const PackedRegion *r = nullptr;
    if (!pack.to_page(cx, cy, r)) continue;  // in the gap between packed regions
    const float sx = static_cast<float>(r->w) / r->pw, sy = static_cast<float>(r->h) / r->ph;
    hp::core::Box b;
    for (int k = 0; k < 4; ++k)
      b[k] = {std::clamp(static_cast<int>(std::lround(r->x + (q[k][0] - r->ox) * sx)), 0, g.w - 1),
              std::clamp(static_cast<int>(std::lround(r->y + (q[k][1] - r->oy) * sy)), 0, g.h - 1)};
    boxes.push_back(b);
  }
  return boxes;
}

}  // namespace

OcrPlan plan_ocr(ModelSet &models, const Route &rt, const PageGeometry &g, bool ocr_enabled) {
  OcrPlan p;
  p.ocr = ocr_enabled && models.has_ocr() && rt.kind != kNative;
  p.need_det = p.ocr && (rt.kind == kFullOcr || !rt.regions.empty());
  p.region_det = p.need_det && plan_region_det(rt, g, models.det(), p.pack);
  return p;
}

void launch_det(ModelSet &m, LayoutLane &lane, OcrJob &job) {
  const auto t = Clock::now();
  job.page = lane.page.get();
  CUDA_CHECK(cudaStreamWaitEvent(m.ocr_stream(), lane.resized));
  if (job.plan.region_det) {
    launch_region_det(m, lane, job.plan.pack, job.g);
    job.out.canvas_px = static_cast<double>(job.plan.pack.canvas_w) * job.plan.pack.canvas_h;
  } else if (job.plan.need_det) {
    m.det().enqueue(m.ocr_stream());
  }
  job.out.det_ms += ms_since(t);
}

void launch_rec(ModelSet &m, OcrJob &job, const PageText &pt, const Route &rt) {
  const auto t_det = Clock::now();
  const PageGeometry &g = job.g;
  std::vector<hp::core::Box> boxes;
  if (job.plan.region_det) boxes = region_boxes_to_page(m.det().collect(), job.plan.pack, g);
  else if (job.plan.need_det) boxes = m.det().collect();
  job.out.det_boxes = static_cast<int>(boxes.size());
  // Box order is left as det emits it. That order is run-dependent: the core
  // CCL/JFA postprocess compacts with atomics, and boxes also jitter by 1-2 px.
  // So two parses of mix100.pdf differ on about half the pages, with run-to-run
  // word F1 0.998-0.999 and identical layout. Sorting boxes (top, left) cut the
  // differing pages from 54 to 24 but lowered Textract word F1 on the 751 IDL
  // scans from 0.8944 to 0.8925 (merge/batching order), so it is not done.

  const float px = g.px_per_pt();
  std::vector<hp::core::Box> &keep = job.keep;
  keep.clear();
  for (const auto &b : boxes) {
    const auto bb = hp::core::aabb(b);
    const RectF r{bb[0] / px, bb[1] / px, bb[2] / px, bb[3] / px};
    if (rt.kind == kPartial && !center_in_regions(rt, r)) continue;
    if (rt.keep_native && overlaps_native(pt, r)) continue;
    keep.push_back(b);
  }
  job.n_detected = keep.size();
  // Text-layer-guided rec: unusable native items, read from pixels at their own
  // box. Padding is vertical only; horizontal neighbours are native text.
  for (const auto &r : rt.rec_boxes) {
    const int pad = std::max(1, static_cast<int>(std::lround(0.15f * r.h() * px)));
    const int x0 = std::clamp(static_cast<int>(std::floor(r.x0 * px)), 0, g.w - 1);
    const int y0 = std::clamp(static_cast<int>(r.y0 * px) - pad, 0, g.h - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(r.x1 * px)), 0, g.w - 1);
    const int y1 = std::clamp(static_cast<int>(r.y1 * px) + pad, 0, g.h - 1);
    if (x1 - x0 >= 3 && y1 - y0 >= 3) keep.push_back(hp::core::Box{{{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}}});
  }
  job.out.direct_rec_boxes = static_cast<int>(rt.rec_boxes.size());
  job.out.rec_boxes = static_cast<int>(keep.size());
  job.out.det_ms += ms_since(t_det);

  const auto t_rec = Clock::now();
  const hp::core::GpuImage img{const_cast<uint8_t *>(job.page), static_cast<size_t>(g.w) * 3, g.h, g.w};
  // Detected lines and text-layer-guided reads are batched apart (the guided ones
  // without the CJK classes on a non-CJK page), in one enqueue.
  std::vector<hp::core::Box> detected, guided;
  std::vector<RecGroup> groups;
  if (job.n_detected < keep.size() && non_cjk_page(pt)) {
    detected.assign(keep.begin(), keep.begin() + static_cast<long>(job.n_detected));
    guided.assign(keep.begin() + static_cast<long>(job.n_detected), keep.end());
    if (!detected.empty()) groups.push_back({&detected, false});
    groups.push_back({&guided, true});
  } else {
    groups.push_back({&keep, false});
  }
  m.rec().enqueue(img, groups, m.ocr_stream());
  job.out.rec_ms += ms_since(t_rec);
}

OcrPageResult finish_ocr(ModelSet &m, OcrJob &job) {
  const auto t_rec = Clock::now();
  const std::vector<std::pair<std::string, float>> texts = m.rec().collect();
  OcrPageResult out = std::move(job.out);
  out.rec_ms += ms_since(t_rec);

  const std::vector<hp::core::Box> &keep = job.keep;
  static const bool debug = std::getenv("HP_DEBUG_REC") != nullptr;  // dump text-layer-guided reads
  for (size_t i = 0; i < keep.size(); ++i) {
    const auto bb = hp::core::aabb(keep[i]);
    const bool direct = i >= job.n_detected;
    if (debug && direct)
      std::fprintf(stderr, "[rec] %d,%d %dx%d conf %.2f '%s'\n", bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1],
                   texts[i].second, texts[i].first.c_str());
    if (texts[i].second <= kMinRecConfidence || texts[i].first.empty()) continue;
    out.lines.push_back({texts[i].first, static_cast<float>(bb[0]), static_cast<float>(bb[1]), static_cast<float>(bb[2]),
                         static_cast<float>(bb[3]), texts[i].second, true, direct});
  }
  return out;
}

}  // namespace hp
