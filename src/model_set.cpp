#include "model_set.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "multi_resize.h"
#include "core/common/cuda/cuda_check.h"

namespace hp {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

uint8_t *LayoutLane::page_buffer(size_t bytes) {
  if (bytes > page_cap) {
    page.reset(bytes);
    page_cap = bytes;
  }
  return page.get();
}

ModelSet::ModelSet(const ModelConfig &cfg) : cfg_(cfg) {
  const std::string &d = cfg_.dir;

  // Two private D-FINE engines, one per lane. Sharing one
  // engine across execution contexts corrupts output under concurrency
  // (engine_loader.h), and the plans are batch-1 because a batch-8 plan ran out
  // of memory on this 6 GB card with two lanes.
  const std::string dfine = d + "/dfine_" + std::to_string(kLayoutSize) + "_fp16.plan";
  auto t = Clock::now();
  for (auto &lane : lanes_) {
    lane = std::make_unique<LayoutLane>();
    if (!lane->layout.load(dfine, kLayoutSize)) throw std::runtime_error("D-FINE load failed: " + dfine);
  }
  load_times_.layout_ms = ms_since(t);
  if (cfg_.ocr) {
    t = Clock::now();
    det_ = std::make_unique<TextDet>();
    if (!det_->load(d + "/det_small_fp16.plan", cfg_.det_max_side)) throw std::runtime_error("det load failed: " + d);
    load_times_.det_ms = ms_since(t);
    t = Clock::now();
    rec_ = load_text_recognizer(cfg_.rec, d + "/rec_small_fp16.plan", d + "/keys_small.txt", ocr_stream_);
    load_times_.rec_ms = ms_since(t);
  }
  if (cfg_.formula) {
    t = Clock::now();
    formula_ = std::make_unique<FormulaRecognizer>();
    if (!formula_->load(d + "/formula")) throw std::runtime_error("formula load failed: " + d + "/formula");
    if (const char *dump = std::getenv("HP_DUMP_FORMULA")) formula_->dump_dir = dump;  // debug: encoder inputs + tokens
    load_times_.formula_ms = ms_since(t);
  }
}

ModelSet::~ModelSet() {
  // Destructors must not throw: drain queued work, ignore errors (the device may
  // already be in a failed state if we are unwinding from one).
  (void)cudaDeviceSynchronize();
  (void)cudaGetLastError();
}

void ModelSet::synchronize() {
  for (auto &lane : lanes_) lane->stream.sync();
  CUDA_CHECK(cudaStreamSynchronize(ocr_stream_));
  CUDA_CHECK(cudaStreamSynchronize(formula_stream_));
}

// Warm-up runs one synthetic page down every GPU path a real page can take, on
// every lane:
//   upload -> multi_resize (layout + det inputs) -> D-FINE;
//   full-page det -> CCL/JFA postprocess -> rec on the boxes found, plus a spread
//   of extra line widths, so the column-budget packer runs multi-batch shapes;
//   region-packed det through CropBatch (the partial-page path);
//   one formula crop through encoder, cross-KV prep and decode.
// The page is white with dark word-like bars, so det finds boxes and rec and
// formula see ink. Outputs are discarded. Measured effect: see engine.h.
void ModelSet::warmup() {
  constexpr int W = 1275, H = 1650;  // US Letter at 150 DPI, the default OCR render
  std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 255);
  auto ink = [&](int x0, int y0, int x1, int y1) {
    for (int y = y0; y < y1; ++y) std::fill(px.begin() + (static_cast<size_t>(y) * W + x0) * 3,
                                            px.begin() + (static_cast<size_t>(y) * W + x1) * 3, 30);
  };
  for (int row = 0; row < 40; ++row)  // 40 "lines" of 3-9 "words"
    for (int x = 120, k = 0; k < 3 + row % 7 && x < W - 200; ++k) {
      const int w = 40 + (row * 37 + k * 53) % 140;
      ink(x, 140 + row * 34, x + w, 140 + row * 34 + 14);
      x += w + 18;
    }
  const size_t bytes = px.size();

  for (auto &lane_ptr : lanes_) {
    LayoutLane &L = *lane_ptr;
    uint8_t *page = L.page_buffer(bytes);
    CUDA_CHECK(cudaMemcpyAsync(page, px.data(), bytes, cudaMemcpyHostToDevice, L.stream));
    ResizeTable table;
    table.add(L.layout.resize_spec(W, H));
    if (det_) table.add(det_->resize_spec(W, H));
    launch_multi_resize(page, static_cast<size_t>(W) * 3, W, H, table, L.stream);
    CUDA_CHECK(cudaEventRecord(L.resized, L.stream));
    L.layout.enqueue(L.stream);

    if (det_) {
      CUDA_CHECK(cudaStreamWaitEvent(ocr_stream_, L.resized));
      std::vector<hp::core::Box> boxes = det_->run(ocr_stream_);
      for (int i = 0; i < 24; ++i) {  // widths 60..1150 px, all on the page
        const int x0 = 20, y0 = 60 + i * 60, x1 = x0 + 60 + i * 47, y1 = y0 + 30;
        boxes.push_back(hp::core::Box{{{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}}});
      }
      const hp::core::GpuImage img{page, static_cast<size_t>(W) * 3, H, W};
      (void)rec_->run(img, boxes, ocr_stream_, false);

      // Region-packed det (partial pages): two regions on one small canvas.
      const int cw = 640, ch = 320;
      float *canvas = det_->canvas_input(cw, ch);
      CUDA_CHECK(cudaMemsetAsync(canvas, 0, sizeof(float) * 3 * cw * ch, ocr_stream_));
      det_crops_.clear();
      int pw = 0, ph = 0;
      const float s = det_->page_scale(W, H);
      det_crops_.add_det_region(100, 130, 600, 200, s, canvas, cw, ch, 0, 0, pw, ph);
      det_crops_.add_det_region(100, 500, 600, 150, s, canvas, cw, ch, 0, ph + 8, pw, ph);
      det_crops_.launch(page, static_cast<size_t>(W) * 3, W, H, ocr_stream_);
      (void)det_->run(ocr_stream_);
    }
    L.stream.sync();
    (void)L.layout.collect();
  }

  if (formula_) {
    const int x0 = 120, y0 = 130, w = 560, h = 120;  // three rows of "words"
    const FormulaCrop crop{px.data() + (static_cast<size_t>(y0) * W + x0) * 3, static_cast<size_t>(W) * 3, w, h};
    (void)formula_->run({crop}, formula_stream_);
  }
  synchronize();
}

}  // namespace hp
