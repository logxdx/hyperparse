#include "page_pipeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "assemble.h"
#include "multi_resize.h"
#include "ocr_stage.h"
#include "orient.h"
#include "pdf_render.h"
#include "tablenet/features.h"
#include "tablenet/page.h"
#include "tablenet/text.h"
#include "tablenet/units.h"
#include "core/common/cuda/cuda_check.h"

namespace hp {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

const char *status_text(int s) {
  switch (s) {
    case RenderPool::kRenderFailed: return "render failed";
    case RenderPool::kWorkerCrashed: return "page worker crashed";
    case RenderPool::kTimedOut: return "page timed out";
    default: return "ok";
  }
}

// A page between the GPU thread and the assembly thread.
struct PageWork {
  PageResult result;                          // filled progressively
  std::vector<PageLine> items;                // native + OCR items, before line merging
  std::vector<std::vector<uint8_t>> crop_px;  // display-formula crops (packed BGR), copied out of the slot
  std::vector<std::array<int, 3>> crop_meta;  // {layout index, w, h}
  // LayoutNet (tables on): the worker's text-layer units, and for pages with a Table region the
  // render in grey (copied out of the slot), for ruling lines and the image branch
  tn::PageUnits units;
  std::vector<uint8_t> grey;
  // A scan parsed upright (orient.h): how far its content was turned clockwise, and the page's size
  // as rendered; the assembled page's coordinates are mapped back to that frame.
  int rot = 0, w0 = 0, h0 = 0;
};

bool has_table_region(const std::vector<LayoutBox> &layout) {
  for (const auto &b : layout)
    if (kDfineLabels[b.cls] == "Table") return true;
  return false;
}

// LayoutNet's input for one page, in points: the text layer's units, OCR lines as units, or both
// (partial pages: native units outside the OCR'd parts). As the reference, a page whose text is
// under 20 characters native, or 80%+ OCR, is an OCR page.
tn::PageInput table_input(PageWork &w) {
  const PageResult &r = w.result;
  const float k = r.dpi / 72.f;
  tn::PageInput in;
  in.w = r.width / k;
  in.h = r.height / k;
  std::vector<tn::Unit> ocr;
  size_t ocr_chars = 0;
  for (const PageLine &l : w.items) {
    if (!l.ocr || l.inline_ocr) continue;
    tn::Unit u = tn::ocr_unit(l.text, l.x0 / k, l.y0 / k, l.x1 / k, l.y1 / k);
    if (u.text.empty() || u.bx[2] <= u.bx[0] || u.bx[3] <= u.bx[1]) continue;
    ocr_chars += tn::char_count(u.text);
    ocr.push_back(std::move(u));
  }
  std::vector<tn::Unit> native;
  size_t native_chars = 0;
  if (r.route != kFullOcr) {
    for (tn::Unit &u : w.units.units) {
      const float cx = static_cast<float>((u.pbox[0] + u.pbox[2]) / 2) * k, cy = static_cast<float>((u.pbox[1] + u.pbox[3]) / 2) * k;
      bool covered = false;
      for (const RectF &g : r.ocr_regions) covered |= cx >= g.x0 && cx <= g.x1 && cy >= g.y0 && cy <= g.y1;
      for (const PageLine &l : w.items) covered |= l.ocr && cx >= l.x0 && cx <= l.x1 && cy >= l.y0 && cy <= l.y1;
      if (covered) continue;
      native_chars += tn::char_count(u.text);
      native.push_back(std::move(u));
    }
  }
  in.ocr = native_chars < 20 || ocr_chars * 5 >= (ocr_chars + native_chars) * 4;
  in.units = in.ocr ? std::move(ocr) : std::move(native);
  if (!in.ocr) in.units.insert(in.units.end(), std::make_move_iterator(ocr.begin()), std::make_move_iterator(ocr.end()));
  in.grey = std::move(w.grey);
  in.px_w = r.width;
  in.px_h = r.height;
  in.px_per_pt = k;
  return in;
}

// Copy every display-formula crop out of the page slot so the slot can be reused
// while the formulas wait for the assembly thread.
void copy_formula_crops(const uint8_t *page, int w, int h, PageWork &pw) {
  const auto &boxes = pw.result.layout;
  for (size_t i = 0; i < boxes.size(); ++i) {
    const auto &b = boxes[i];
    if (kDfineLabels[b.cls] != "Formula") continue;
    const int x0 = std::clamp(static_cast<int>(b.x0), 0, w - 1), y0 = std::clamp(static_cast<int>(b.y0), 0, h - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(b.x1)), x0 + 1, w);
    const int y1 = std::clamp(static_cast<int>(std::ceil(b.y1)), y0 + 1, h);
    const size_t row = static_cast<size_t>(x1 - x0) * 3;
    std::vector<uint8_t> px(row * (y1 - y0));
    for (int y = y0; y < y1; ++y)
      std::memcpy(px.data() + (y - y0) * row, page + (static_cast<size_t>(y) * w + x0) * 3, row);
    pw.crop_px.push_back(std::move(px));
    pw.crop_meta.push_back({static_cast<int>(i), x1 - x0, y1 - y0});
  }
}

// ── assembly thread ─────────────────────────────────────────────────────────
// Drains every page queued so far, decodes all of their formula crops together,
// then assembles and delivers each page. Owns its own statistics; they are read
// only after join().
class AssemblyStage {
 public:
  AssemblyStage(ModelSet &models, const PageSink &sink, bool ordernet)
      : models_(models), sink_(sink), ordernet_(ordernet && models.has_ordernet()), thread_([this] { loop(); }) {}
  ~AssemblyStage() { close(); }

  void push(PageWork &&w) {
    {
      std::lock_guard l(mu_);
      q_.push_back(std::move(w));
    }
    cv_.notify_one();
  }
  // Discard queued pages (after an error elsewhere) instead of delivering them.
  void cancel() {
    std::lock_guard l(mu_);
    cancelled_ = true;
    q_.clear();
  }
  [[nodiscard]] bool failed() const {
    std::lock_guard l(mu_);
    return error_ != nullptr;
  }
  // Finish queued work and join; rethrows the first error from the thread.
  void finish() {
    close();
    if (error_) std::rethrow_exception(error_);
  }

  int formulas = 0, formula_batches = 0, lines = 0, table_pages = 0, model_table_count = 0, order_pages = 0;
  double formula_ms = 0, assemble_ms = 0, table_ms = 0, order_ms = 0, busy_ms = 0;

 private:
  void close() {
    {
      std::lock_guard l(mu_);
      closed_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

  void loop() {
    for (;;) {
      std::vector<PageWork> batch;
      {
        std::unique_lock l(mu_);
        cv_.wait(l, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return;
        while (!q_.empty()) {
          batch.push_back(std::move(q_.front()));
          q_.pop_front();
        }
        if (cancelled_ || error_) continue;
      }
      try {
        const auto t = Clock::now();
        process(batch);
        busy_ms += ms_since(t);
      } catch (...) {
        std::lock_guard l(mu_);
        if (!error_) error_ = std::current_exception();
      }
    }
  }

  void process(std::vector<PageWork> &batch) {
    std::vector<FormulaCrop> crops;
    for (auto &w : batch)
      for (size_t k = 0; k < w.crop_px.size(); ++k)
        crops.push_back({w.crop_px[k].data(), static_cast<size_t>(w.crop_meta[k][1]) * 3, w.crop_meta[k][1],
                         w.crop_meta[k][2]});
    std::vector<std::string> latex;
    if (!crops.empty()) {
      const auto t = Clock::now();
      latex = models_.formula().run(crops, models_.formula_stream());
      formula_ms += ms_since(t);
      formulas += static_cast<int>(crops.size());
      ++formula_batches;
    }
    size_t c = 0;
    for (auto &w : batch) {
      PageResult &r = w.result;
      if (r.ok) {
        for (size_t k = 0; k < w.crop_px.size(); ++k, ++c) r.formulas.push_back({w.crop_meta[k][0], latex[c], 1.f});
        std::vector<ModelTable> model_tables;
        if (!w.grey.empty()) {
          const auto tt = Clock::now();
          const float k = r.dpi / 72.f;
          const tn::PageInput in = table_input(w);
          for (tn::Table &t : tn::run_page(models_.tablenet(), in, models_.table_stream()))
            model_tables.push_back({static_cast<float>(t.bbox[0]) * k, static_cast<float>(t.bbox[1]) * k,
                                    static_cast<float>(t.bbox[2]) * k, static_cast<float>(t.bbox[3]) * k,
                                    std::move(t.grid), k});
          table_ms += ms_since(tt);
          ++table_pages;
          model_table_count += static_cast<int>(model_tables.size());
        }
        const auto t = Clock::now();
        AssembledPage page = assemble_page(r.page, r.width, r.height, r.layout, w.items, merge_into_lines(w.items),
                                           r.formulas, model_tables, ordernet_ ? &models_.ordernet() : nullptr,
                                           models_.table_stream());
        const double ams = ms_since(t);
        assemble_ms += ams;
        if (page.ordered_by_model) ++order_pages, order_ms += ams;
        r.markdown = std::move(page.markdown);
        r.tables = std::move(page.tables);
        r.reading_order = std::move(page.order);
        r.lines = std::move(page.lines);
        lines += static_cast<int>(r.lines.size());
        if (w.rot) {
          const float W0 = static_cast<float>(w.w0), H0 = static_cast<float>(w.h0);
          for (LayoutBox &b : r.layout) box_to_original(b.x0, b.y0, b.x1, b.y1, w.rot, W0, H0);
          for (PageLine &l : r.lines) box_to_original(l.x0, l.y0, l.x1, l.y1, w.rot, W0, H0);
          r.width = w.w0, r.height = w.h0, r.rotation = w.rot;
        }
      }
      sink_(std::move(r));
    }
  }

  ModelSet &models_;
  const PageSink &sink_;
  const bool ordernet_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<PageWork> q_;
  bool closed_ = false, cancelled_ = false;
  std::exception_ptr error_;
  std::thread thread_;  // last: starts after every other member is initialised
};

// ── GPU thread ──────────────────────────────────────────────────────────────
class DocumentRun {
 public:
  DocumentRun(ModelSet &m, RenderPool &pool, const ParseOptions &opt, const PageSink &sink)
      : m_(m), pool_(pool), opt_(opt), sink_(sink) {
    job_opt_.ocr_dpi = opt.ocr_dpi;
    job_opt_.force = opt.ocr == OcrMode::kAll ? 1 : opt.ocr == OcrMode::kNone ? 2 : 0;
    job_opt_.tables = opt.tables && m.has_tables();
    job_opt_.tables_dpi = opt.tables_dpi;
  }

  ParseStats run(std::span<const uint8_t> pdf) {
    const auto t_start = Clock::now();
    {
      PdfRaster probe(pdf.data(), pdf.size());  // parent-side: page count, and reject unreadable PDFs early
      if (!probe.ok()) throw InvalidPdfError("not a readable PDF");
      n_pages_ = probe.page_count();
    }
    if (opt_.max_pages > 0 && n_pages_ > opt_.max_pages) throw PageLimitError(n_pages_, opt_.max_pages);
    if (opt_.on_start) opt_.on_start(n_pages_);
    stats_.pages = n_pages_;
    if (n_pages_ == 0) return stats_;
    gen_ = pool_.set_document(pdf);
    received_.assign(n_pages_, 0);
    for (int s = pool_.slots() - 1; s >= 0; --s) free_slots_.push_back(s);
    const int formula_steps0 = m_.has_formula() ? m_.formula().steps_run : 0;

    AssemblyStage assembly(m_, sink_, opt_.ordernet);
    assembly_ = &assembly;
    try {
      submit_ready();
      for (int done = 0; done < n_pages_;) {
        const auto t_wait = Clock::now();
        const RenderPool::Done d = pool_.wait_done();
        stats_.render_wait_ms += ms_since(t_wait);
        if (!accept(d)) continue;
        --outstanding_;
        ++done;
        if (d.status != RenderPool::kOk) {
          fail_page(d);
          continue;
        }
        LaneJob &job = lanes_[next_lane_];
        if (job.busy) finish(next_lane_);
        start(next_lane_, d);
        if (LaneJob &prev = lanes_[next_lane_ ^ 1]; prev.rec_pending) complete_ocr(prev);
        if (job.det_pending) queue_rec(job);
        next_lane_ ^= 1;
        if (assembly.failed()) throw std::runtime_error("assembly failed");  // replaced by its own error below
      }
      for (auto &lane : lanes_)
        if (lane.rec_pending) complete_ocr(lane);
      for (int i = 0; i < ModelSet::kLanes; ++i)
        if (lanes_[i].busy) finish(i);
    } catch (...) {
      abort_and_drain(assembly);
      pool_.release_document();
      assembly.finish();  // the assembly thread's error, if any, is the root cause
      throw;
    }
    pool_.release_document();  // every job is done: nobody reads the memfd any more
    const auto t_drain = Clock::now();
    assembly.finish();
    stats_.drain_ms = ms_since(t_drain);
    stats_.assembly_busy_ms = assembly.busy_ms;

    stats_.formulas = assembly.formulas;
    stats_.formula_batches = assembly.formula_batches;
    stats_.formula_ms = assembly.formula_ms;
    stats_.assemble_ms = assembly.assemble_ms;
    stats_.table_pages = assembly.table_pages;
    stats_.model_tables = assembly.model_table_count;
    stats_.table_ms = assembly.table_ms;
    stats_.order_pages = assembly.order_pages;
    stats_.order_ms = assembly.order_ms;
    stats_.lines = assembly.lines;
    stats_.formula_steps = m_.has_formula() ? m_.formula().steps_run - formula_steps0 : 0;
    stats_.wall_ms = ms_since(t_start);
    return stats_;
  }

 private:
  struct LaneJob {
    bool busy = false;
    RenderPool::Done done{};
    PageText pt;
    Route rt;
    PageWork work;
    Clock::time_point t0;
    OcrJob ocr;
    bool det_pending = false;  // det queued, rec not yet
    bool rec_pending = false;  // rec queued, not yet collected
  };

  // Ignore duplicates and stale results. The supervisor may report a crashed
  // worker's page that was in fact delivered, and a previous document's late
  // duplicates carry an older generation.
  bool accept(const RenderPool::Done &d) {
    if (d.gen != gen_ || d.page < 0 || d.page >= n_pages_ || received_[d.page]) return false;
    received_[d.page] = 1;
    return true;
  }

  void submit_ready() {
    while (next_page_ < n_pages_ && !free_slots_.empty()) {
      const int slot = free_slots_.back();
      free_slots_.pop_back();
      pool_.submit(next_page_++, slot, gen_, job_opt_);
      ++outstanding_;
    }
  }

  void release_slot(int slot) {
    free_slots_.push_back(slot);
    submit_ready();
  }

  void fail_page(const RenderPool::Done &d) {
    ++stats_.failed;
    PageWork w;
    w.result.page = d.page;
    w.result.error = status_text(d.status);
    release_slot(d.slot);
    assembly_->push(std::move(w));
  }

  // Upload, resize, layout, and the OCR det launch. Layout stays in flight on the
  // lane stream, det on the OCR stream.
  //
  // OCR is pipelined across pages (ocr_stage.h): the main loop queues page i's
  // det here, then collects page i-1's rec (decoding it on the host while page
  // i's det runs), then queues page i's rec. The OCR stream so always holds the
  // next piece of work, and page i+1's upload, resize and layout run beside page
  // i's rec. Det and rec each have one set of buffers, which this order
  // respects: page i's det is collected before page i+1's resize writes the det
  // input, and page i-1's rec is collected before page i's rec is queued.
  void start(int lane_index, const RenderPool::Done &d) {
    LaneJob &job = lanes_[lane_index];
    LayoutLane &lane = m_.lane(lane_index);
    job = LaneJob{};
    job.busy = true;
    job.done = d;
    job.t0 = Clock::now();
    const uint32_t text_len = d.units_off ? d.units_off : d.meta_len;
    if (!deserialize(pool_.meta(d.slot), text_len, job.pt, job.rt)) throw std::runtime_error("corrupt page metadata");
    if (d.units_off &&
        !tn::deserialize(pool_.meta(d.slot) + d.units_off, d.meta_len - d.units_off, job.work.units))
      throw std::runtime_error("corrupt page units");
    ++stats_.routes[job.rt.kind];
    stats_.text_ms += d.text_ms;
    stats_.render_ms += d.render_ms;

    // A scan turned sideways or upside down (orient.h) is turned upright in the slot before anything
    // reads it; the assembly thread maps the page's coordinates back. Only pages read wholly from
    // pixels: no text-layer item or box (page coordinates of the unturned page) takes part.
    job.work.w0 = d.w, job.work.h0 = d.h;
    if (m_.has_orient() && job.rt.kind == kFullOcr && job.rt.rec_boxes.empty() &&
        std::none_of(job.pt.items.begin(), job.pt.items.end(), [](const TextItem &it) { return !it.dropped; })) {
      // On the lane stream: the OCR stream may still hold the previous page's rec.
      const OrientResult o = m_.orient().classify(pool_.pixels(d.slot), d.w, d.h, lane.stream);
      if (o.degrees_cw != 0 && o.conf >= kOrientMinConf) {
        rotate_upright_bgr(pool_.pixels(d.slot), job.done.w, job.done.h, o.degrees_cw);
        job.work.rot = o.degrees_cw;
      }
    }
    const int W = job.done.w, H = job.done.h;
    const PageGeometry g{W, H, d.dpi};
    const size_t bytes = static_cast<size_t>(W) * H * 3;
    uint8_t *page = lane.page_buffer(bytes);
    CUDA_CHECK(cudaMemcpyAsync(page, pool_.pixels(d.slot), bytes, cudaMemcpyHostToDevice, lane.stream));
    const OcrPlan plan = plan_ocr(m_, job.rt, g, true);
    ResizeTable table;
    table.add(lane.layout.resize_spec(W, H));
    if (plan.need_det && !plan.region_det) table.add(m_.det().resize_spec(W, H));
    launch_multi_resize(page, static_cast<size_t>(W) * 3, W, H, table, lane.stream);
    CUDA_CHECK(cudaEventRecord(lane.resized, lane.stream));
    lane.layout.enqueue(lane.stream);

    PageResult &r = job.work.result;
    r.page = d.page;
    r.width = W;
    r.height = H;
    r.dpi = d.dpi;
    r.route = job.rt.kind;
    r.route_reasons = job.rt.reasons;
    r.ms.text = d.text_ms;
    r.ms.render = d.render_ms;
    const float k = g.px_per_pt();
    for (const auto &q : job.rt.regions) r.ocr_regions.push_back({q.x0 * k, q.y0 * k, q.x1 * k, q.y1 * k});
    for (const auto &it : job.pt.items)
      if (!it.dropped) {
        job.work.items.push_back({it.text, it.box.x0 * k, it.box.y0 * k, it.box.x1 * k, it.box.y1 * k, 1.f, false});
        if (it.style) job.work.items.back().styles.push_back({0, static_cast<uint32_t>(it.text.size()), it.style});
      }

    if (!plan.ocr) return;
    job.ocr.plan = plan;
    job.ocr.g = g;
    launch_det(m_, lane, job.ocr);
    job.det_pending = true;
  }

  // Wait for the page's det, filter the boxes, queue its rec.
  void queue_rec(LaneJob &job) {
    launch_rec(m_, job.ocr, job.pt, job.rt);
    job.det_pending = false;
    job.rec_pending = true;
  }

  // Wait for the page's rec and add its OCR lines to the page.
  void complete_ocr(LaneJob &job) {
    const bool region_det = job.ocr.plan.region_det;
    OcrPageResult ocr = finish_ocr(m_, job.ocr);
    job.rec_pending = false;
    stats_.ocr_ms += ocr.det_ms + ocr.rec_ms;
    stats_.det_ms += ocr.det_ms;
    stats_.rec_ms += ocr.rec_ms;
    stats_.det_boxes += ocr.det_boxes;
    stats_.rec_boxes += ocr.rec_boxes;
    stats_.direct_rec_boxes += ocr.direct_rec_boxes;
    if (region_det) {
      ++stats_.region_det_pages;
      stats_.region_det_canvas_px += ocr.canvas_px;
    }
    PageResult &r = job.work.result;
    r.det_boxes = ocr.det_boxes;
    r.rec_boxes = ocr.rec_boxes;
    r.region_det = region_det;
    // Debug: every OCR line as recognised (before line merging), one per row:
    // page, box (page px), confidence, text. For A/B checks of the OCR stage.
    static FILE *const dump = [] {
      const char *p = std::getenv("HP_DUMP_OCR");
      return p ? std::fopen(p, "w") : nullptr;
    }();
    if (dump)
      for (const PageLine &l : ocr.lines)
        std::fprintf(dump, "%d\t%.0f\t%.0f\t%.0f\t%.0f\t%.3f\t%s\n", r.page, l.x0, l.y0, l.x1, l.y1, l.conf,
                     l.text.c_str());
    for (auto &l : ocr.lines) job.work.items.push_back(std::move(l));
  }

  // Wait for the lane's layout, take what the assembly thread needs out of the
  // slot, free the slot, hand the page on.
  void finish(int lane_index) {
    LaneJob &job = lanes_[lane_index];
    LayoutLane &lane = m_.lane(lane_index);
    // The main loop has always completed this page's OCR by now; kept for safety
    // (the other lane's rec first: rec has one set of buffers).
    if (job.det_pending) {
      if (LaneJob &other = lanes_[lane_index ^ 1]; other.rec_pending) complete_ocr(other);
      queue_rec(job);
    }
    if (job.rec_pending) complete_ocr(job);
    const auto t_wait = Clock::now();
    lane.stream.sync();
    stats_.layout_wait_ms += ms_since(t_wait);
    PageWork &w = job.work;
    w.result.layout = lane.layout.collect();
    w.result.ok = true;
    if (opt_.formulas && m_.has_formula()) copy_formula_crops(pool_.pixels(job.done.slot), job.done.w, job.done.h, w);
    if (job_opt_.tables && has_table_region(w.result.layout))
      w.grey = tn::grey_from_bgr(pool_.pixels(job.done.slot), job.done.w, job.done.h, static_cast<size_t>(job.done.w) * 3);
    w.result.ms.gpu = ms_since(job.t0);
    job.busy = false;
    release_slot(job.done.slot);
    assembly_->push(std::move(w));
  }

  // Error path: stop feeding work, wait for every job already submitted (their
  // slots are being written), quiesce the GPU, and drop undelivered pages.
  void abort_and_drain(AssemblyStage &assembly) noexcept {
    assembly.cancel();
    next_page_ = n_pages_;  // submit nothing more
    try {
      while (outstanding_ > 0) {
        if (accept(pool_.wait_done())) --outstanding_;
      }
    } catch (...) {
      // Pool stalled or dead; nothing more can be drained. The engine reports
      // the original error, and the next parse fails fast on the dead pool.
    }
    try {
      m_.synchronize();
    } catch (...) {
    }
  }

  ModelSet &m_;
  RenderPool &pool_;
  const ParseOptions &opt_;
  const PageSink &sink_;
  PageJobOptions job_opt_;
  AssemblyStage *assembly_ = nullptr;

  ParseStats stats_;
  int n_pages_ = 0, next_page_ = 0, outstanding_ = 0, next_lane_ = 0;
  uint32_t gen_ = 0;
  std::vector<char> received_;
  std::vector<int> free_slots_;
  std::array<LaneJob, ModelSet::kLanes> lanes_;
};

}  // namespace

ParseStats parse_document(ModelSet &models, RenderPool &pool, std::span<const uint8_t> pdf, const ParseOptions &opt,
                          const PageSink &sink) {
  return DocumentRun(models, pool, opt, sink).run(pdf);
}

}  // namespace hp
