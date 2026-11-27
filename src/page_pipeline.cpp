#include "page_pipeline.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
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

#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

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

struct Doc;

// A page between the GPU thread and the assembly thread.
struct PageWork {
  Doc *doc = nullptr;
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

// ── per-document state ──────────────────────────────────────────────────────

// One document in flight. It lives on its parse() caller's stack; the pipeline
// threads hold pointers to it until `unfinished` reaches zero.
struct Doc {
  // Fixed before the document is handed to the GPU thread.
  const PageSink *sink = nullptr;
  PageJobOptions job_opt;
  bool formulas = false, ordernet = false;
  RenderPool::DocRef ref;
  int n_pages = 0;

  // GPU thread only.
  int next_page = 0;    // next page to submit
  int outstanding = 0;  // submitted, Done not yet received
  int gpu_left = 0;     // pages the GPU thread has not yet handed on or dropped
  std::vector<char> received;
  ParseStats gpu;       // the GPU thread's share of the stats
  Clock::time_point t_gpu_done;

  // Assembly thread only: formulas, tables, reading order, assembly, lines.
  ParseStats asm_stats;

  // Shared.
  std::atomic<bool> failed{false};
  // Pages not yet delivered or dropped, plus one reference the GPU thread holds
  // while the document is on its list. Whoever takes it to zero completes the
  // document (Impl::finish_pages) and must not touch it afterwards.
  std::atomic<int> unfinished{0};
  std::exception_ptr error;  // first error; guarded by Impl::mu_
  bool done = false;         // guarded by Impl::mu_
  Clock::time_point t_done;  // guarded by Impl::mu_
};

}  // namespace

class PagePipeline::Impl {
 public:
  Impl(ModelSet &m, RenderPool &pool) : m_(m), pool_(pool) {
    wake_fd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (wake_fd_ < 0) throw std::runtime_error("PagePipeline: eventfd failed");
    for (int s = pool_.documents() - 1; s >= 0; --s) free_doc_slots_.push_back(s);
    reset_slots();
    assembly_ = std::thread([this] { assembly_loop(); });
    try {
      gpu_ = std::thread([this] { gpu_loop(); });
    } catch (...) {
      stop_assembly();
      ::close(wake_fd_);
      throw;
    }
  }

  ~Impl() {
    {
      std::lock_guard l(mu_);
      stop_ = true;
    }
    wake();
    gpu_.join();
    stop_assembly();
    ::close(wake_fd_);
  }

  ParseStats parse(std::span<const uint8_t> pdf, const ParseOptions &opt, const PageSink &sink) {
    const auto t_start = Clock::now();
    Doc doc;
    {
      PdfRaster probe(pdf.data(), pdf.size());  // parent-side: page count, and reject unreadable PDFs early
      if (!probe.ok()) throw InvalidPdfError("not a readable PDF");
      doc.n_pages = probe.page_count();
    }
    const int n = doc.n_pages;
    if (opt.max_pages > 0 && n > opt.max_pages) throw PageLimitError(n, opt.max_pages);
    if (opt.on_start) opt.on_start(n);
    if (n == 0) return {};

    doc.sink = &sink;
    doc.job_opt.ocr_dpi = opt.ocr_dpi;
    doc.job_opt.force = opt.ocr == OcrMode::kAll ? 1 : opt.ocr == OcrMode::kNone ? 2 : 0;
    doc.job_opt.tables = opt.tables && m_.has_tables();
    doc.job_opt.tables_dpi = opt.tables_dpi;
    doc.formulas = opt.formulas && m_.has_formula();
    doc.ordernet = opt.ordernet && m_.has_ordernet();
    doc.received.assign(n, 0);
    doc.gpu_left = n;
    doc.unfinished.store(n + 1);

    const int slot = acquire_doc_slot();
    try {
      doc.ref = pool_.set_document(slot, pdf);
    } catch (...) {
      release_doc_slot(slot);
      throw;
    }
    const int64_t wait0 = render_wait_ns_.load(), busy0 = assembly_busy_ns_.load();
    {
      std::lock_guard l(mu_);
      incoming_.push_back(&doc);
    }
    wake();
    {
      std::unique_lock l(mu_);
      cv_.wait(l, [&] { return doc.done; });
    }
    pool_.release_document(slot);  // every job of the document is done: nobody reads its memfd any more
    release_doc_slot(slot);
    if (doc.error) std::rethrow_exception(doc.error);

    ParseStats s = doc.gpu;
    const ParseStats &a = doc.asm_stats;
    s.pages = n;
    s.formulas = a.formulas;
    s.formula_batches = a.formula_batches;
    s.formula_steps = a.formula_steps;
    s.formula_ms = a.formula_ms;
    s.assemble_ms = a.assemble_ms;
    s.table_pages = a.table_pages;
    s.model_tables = a.model_tables;
    s.table_ms = a.table_ms;
    s.order_pages = a.order_pages;
    s.order_ms = a.order_ms;
    s.lines = a.lines;
    // While this document was in flight (with documents overlapping, the GPU
    // thread's waits and the assembly thread's work also served the others).
    s.render_wait_ms = static_cast<double>(render_wait_ns_.load() - wait0) / 1e6;
    s.assembly_busy_ms = static_cast<double>(assembly_busy_ns_.load() - busy0) / 1e6;
    s.drain_ms = std::chrono::duration<double, std::milli>(doc.t_done - doc.t_gpu_done).count();
    s.wall_ms = ms_since(t_start);
    return s;
  }

 private:
  // Fairness between documents (submit_ready, start_next).
  static constexpr uint64_t kFairEvery = 3;

  struct Ready {
    Doc *doc;
    RenderPool::Done done;
  };

  struct LaneJob {
    bool busy = false;
    Doc *doc = nullptr;
    RenderPool::Done done{};
    PageText pt;
    Route rt;
    PageWork work;
    Clock::time_point t0;
    OcrJob ocr;
    bool det_pending = false;  // det queued, rec not yet
    bool rec_pending = false;  // rec queued, not yet collected
  };

  // ── shared ────────────────────────────────────────────────────────────────

  void wake() {
    const uint64_t one = 1;
    (void)!::write(wake_fd_, &one, sizeof one);
  }

  int acquire_doc_slot() {
    std::unique_lock l(mu_);
    cv_.wait(l, [&] { return !free_doc_slots_.empty(); });
    const int s = free_doc_slots_.back();
    free_doc_slots_.pop_back();
    return s;
  }

  void release_doc_slot(int s) {
    {
      std::lock_guard l(mu_);
      free_doc_slots_.push_back(s);
    }
    cv_.notify_all();
  }

  // Record the document's first error; it gets no more slots and its pages are
  // dropped instead of delivered.
  void fail(Doc &d, std::exception_ptr e) {
    {
      std::lock_guard l(mu_);
      if (!d.error) d.error = e;
    }
    d.failed.store(true);
    wake();
  }

  // `n` of the document's pages delivered or dropped (or the GPU thread let go).
  void finish_pages(Doc &d, int n) {
    if (n <= 0 || d.unfinished.fetch_sub(n, std::memory_order_acq_rel) != n) return;
    std::lock_guard l(mu_);
    d.done = true;
    d.t_done = Clock::now();
    cv_.notify_all();
  }

  // ── GPU thread ────────────────────────────────────────────────────────────

  void gpu_loop() {
    const int stall_ms = static_cast<int>((pool_.page_timeout() + std::chrono::seconds(30)).count());
    for (;;) {
      try {
        bool stop;
        {
          std::lock_guard l(mu_);
          docs_.insert(docs_.end(), incoming_.begin(), incoming_.end());
          incoming_.clear();
          stop = stop_;
        }
        prune();
        if (stop && docs_.empty() && ready_.empty() && !lanes_busy()) return;
        submit_ready();
        // Next event. Rendered pages come first; with none, finish a lane rather
        // than leave its page waiting for the next render.
        int ev = poll_events(0);
        if (ev == 0 && ready_.empty()) {
          if (flush_one_lane()) continue;
          const bool rendering = outstanding_ > 0;
          const auto t = Clock::now();
          ev = poll_events(rendering ? stall_ms : -1);
          if (rendering) render_wait_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count();
          if (ev == 0) throw std::runtime_error("RenderPool: no page finished within the page timeout (pool stalled)");
        }
        if (ev & kWoken) {
          uint64_t v;
          (void)!::read(wake_fd_, &v, sizeof v);
        }
        if (ev & kDone) {
          do on_done(read_done());
          while (poll_events(0) & kDone);
        }
        if (!ready_.empty()) start_next();
      } catch (...) {
        fatal(std::current_exception());
      }
    }
  }

  static constexpr int kDone = 1, kWoken = 2;

  int poll_events(int timeout_ms) {
    pollfd p[2] = {{pool_dead_ ? -1 : pool_.done_fd(), POLLIN, 0}, {wake_fd_, POLLIN, 0}};
    for (;;) {
      const int r = ::poll(p, 2, timeout_ms);
      if (r < 0 && errno == EINTR) continue;
      if (r < 0) throw std::runtime_error("PagePipeline: poll failed");
      // POLLHUP/POLLERR on the result pipe also count: read_done() then reports the dead pool.
      return (p[0].revents ? kDone : 0) | (p[1].revents ? kWoken : 0);
    }
  }

  // A closed result pipe means the supervisor is gone: stop polling it (it would
  // stay readable forever); submitting then fails every new document at once.
  RenderPool::Done read_done() {
    try {
      return pool_.read_done();
    } catch (...) {
      pool_dead_ = true;
      throw;
    }
  }

  bool lanes_busy() const {
    return std::any_of(lanes_.begin(), lanes_.end(), [](const LaneJob &j) { return j.busy; });
  }

  // Finish the lane holding the older page, if any.
  bool flush_one_lane() {
    for (const int i : {next_lane_, next_lane_ ^ 1})
      if (lanes_[i].busy) {
        finish_lane(i);
        return true;
      }
    return false;
  }

  Doc *find(uint32_t gen) const {
    for (Doc *d : docs_)
      if (d->ref.gen == gen) return d;
    return nullptr;
  }

  // Drop documents that need nothing more from the GPU thread's list; for a
  // failed one, drop the pages it never submitted first.
  void prune() {
    for (size_t i = 0; i < docs_.size();) {
      Doc &d = *docs_[i];
      if (d.failed.load() && d.next_page < d.n_pages) {
        const int left = d.n_pages - d.next_page;
        d.next_page = d.n_pages;
        drop(d, left);
      }
      if (d.next_page == d.n_pages && d.outstanding == 0) {
        docs_.erase(docs_.begin() + static_cast<std::ptrdiff_t>(i));
        if (rr_ > i) --rr_;
        finish_pages(d, 1);  // the GPU thread's reference: `d` may be gone now
        continue;
      }
      ++i;
    }
    if (rr_ >= docs_.size()) rr_ = 0;
  }

  // Free slots go to the document with the fewest pages left to submit (ties:
  // the earliest), so a short document overtakes a long one; every
  // kFairEvery-th slot goes round-robin, so a long one still gets its share.
  void submit_ready() {
    while (!free_slots_.empty()) {
      Doc *pick = nullptr;
      const bool fair_turn = ++slot_turns_ % kFairEvery == 0;
      for (size_t k = 0; k < docs_.size(); ++k) {
        const size_t i = fair_turn ? (rr_ + k) % docs_.size() : k;
        Doc *d = docs_[i];
        if (d->next_page >= d->n_pages || d->failed.load()) continue;
        if (fair_turn) {
          pick = d;
          rr_ = (i + 1) % docs_.size();
          break;
        }
        if (!pick || d->n_pages - d->next_page < pick->n_pages - pick->next_page) pick = d;
      }
      if (!pick) return;
      pool_.submit(pick->next_page, free_slots_.back(), pick->ref, pick->job_opt);
      free_slots_.pop_back();
      ++pick->next_page;
      ++pick->outstanding;
      ++outstanding_;
    }
  }

  void reset_slots() {
    free_slots_.clear();
    for (int s = pool_.slots() - 1; s >= 0; --s) free_slots_.push_back(s);
  }

  // A page leaves the GPU thread: to the assembly thread, or dropped.
  void left_gpu(Doc &d, int n) {
    d.gpu_left -= n;
    if (d.gpu_left == 0) d.t_gpu_done = Clock::now();
  }
  void hand_on(PageWork &&w) {
    Doc &d = *w.doc;
    left_gpu(d, 1);  // before the push: the assembly thread may complete `d` at once
    try {
      std::lock_guard l(amu_);
      aq_.push_back(std::move(w));
    } catch (...) {
      fail(d, std::current_exception());
      finish_pages(d, 1);
      return;
    }
    acv_.notify_one();
  }
  void drop(Doc &d, int n) {
    left_gpu(d, n);
    finish_pages(d, n);
  }

  void on_done(const RenderPool::Done &r) {
    // Ignore duplicates and stale results. The supervisor may report a crashed
    // worker's page that was in fact delivered, and a finished document's late
    // duplicates carry a generation no longer on the list.
    Doc *d = find(r.gen);
    if (!d || r.page < 0 || r.page >= d->n_pages || d->received[r.page]) return;
    d->received[r.page] = 1;
    --d->outstanding;
    --outstanding_;
    if (d->failed.load()) {
      free_slots_.push_back(r.slot);
      drop(*d, 1);
      return;
    }
    if (r.status != RenderPool::kOk) {
      ++d->gpu.failed;
      free_slots_.push_back(r.slot);
      PageWork w;
      w.doc = d;
      w.result.page = r.page;
      w.result.error = status_text(r.status);
      hand_on(std::move(w));
      return;
    }
    ready_.push_back({d, r});
  }

  // Start one rendered page on the GPU: the one whose document has the fewest
  // pages left on the GPU thread (ties: the earliest), except that every
  // kFairEvery-th page is the earliest, so a long document keeps a share of the
  // GPU under a stream of short ones.
  void start_next() {
    std::erase_if(ready_, [&](const Ready &r) {  // failed documents' pages: dropped, not parsed
      if (!r.doc->failed.load()) return false;
      free_slots_.push_back(r.done.slot);
      drop(*r.doc, 1);
      return true;
    });
    if (ready_.empty()) return;
    size_t pick = 0;
    if (++gpu_turns_ % kFairEvery != 0)
      for (size_t i = 1; i < ready_.size(); ++i)
        if (ready_[i].doc->gpu_left < ready_[pick].doc->gpu_left) pick = i;
    const int li = next_lane_;
    if (lanes_[li].busy) finish_lane(li);
    LaneJob &job = lanes_[li];
    job = LaneJob{};
    job.busy = true;  // from here on, fatal() or page_error() accounts for the page
    job.doc = ready_[pick].doc;
    job.done = ready_[pick].done;
    job.t0 = Clock::now();
    ready_.erase(ready_.begin() + static_cast<std::ptrdiff_t>(pick));
    submit_ready();  // the slot just freed goes back to the workers before this page's OCR
    try {
      start(li);
    } catch (...) {
      page_error(li, std::current_exception());
    }
    if (LaneJob &prev = lanes_[li ^ 1]; prev.busy && prev.rec_pending) {
      try {
        complete_ocr(prev);
      } catch (...) {
        page_error(li ^ 1, std::current_exception());
      }
    }
    if (job.busy && job.det_pending) {
      try {
        queue_rec(job);
      } catch (...) {
        page_error(li, std::current_exception());
      }
    }
    next_lane_ ^= 1;
  }

  void finish_lane(int li) {
    try {
      finish(li);
    } catch (...) {
      page_error(li, std::current_exception());
    }
  }

  // A GPU stage threw for the lane's page: fail its document only. The lane may
  // still be reading the slot, so quiesce the GPU before freeing it.
  void page_error(int li, std::exception_ptr e) {
    LaneJob &job = lanes_[li];
    try {
      m_.synchronize();
    } catch (...) {
    }
    Doc &d = *job.doc;
    fail(d, e);
    if (!job.busy) return;  // the page already went on (finish() failed after handing it over)
    job.busy = job.det_pending = job.rec_pending = false;
    free_slots_.push_back(job.done.slot);
    drop(d, 1);
  }

  // The pool is broken (dead or stalled) or the GPU thread failed outside any
  // one page: fail every document in flight and start over with an empty ring.
  void fatal(std::exception_ptr e) noexcept {
    try {
      m_.synchronize();
    } catch (...) {
    }
    for (const Ready &r : ready_) {
      fail(*r.doc, e);
      drop(*r.doc, 1);
    }
    ready_.clear();
    for (LaneJob &job : lanes_) {
      if (!job.busy) continue;
      job.busy = job.det_pending = job.rec_pending = false;
      fail(*job.doc, e);
      drop(*job.doc, 1);
    }
    for (Doc *dp : docs_) {
      Doc &d = *dp;
      fail(d, e);
      drop(d, d.n_pages - d.next_page + d.outstanding);
      d.next_page = d.n_pages;
      d.outstanding = 0;
      finish_pages(d, 1);
    }
    docs_.clear();
    rr_ = 0;
    outstanding_ = 0;
    reset_slots();
  }

  // Upload, resize, layout, and the OCR det launch. Layout stays in flight on the
  // lane stream, det on the OCR stream.
  //
  // OCR is pipelined across pages (ocr_stage.h): start_next() queues page i's det
  // here, then collects page i-1's rec (decoding it on the host while page i's
  // det runs), then queues page i's rec. The OCR stream so always holds the next
  // piece of work, and page i+1's upload, resize and layout run beside page i's
  // rec. Det and rec each have one set of buffers, which this order respects:
  // page i's det is collected before page i+1's resize writes the det input, and
  // page i-1's rec is collected before page i's rec is queued. The pages may
  // belong to different documents.
  void start(int lane_index) {
    LaneJob &job = lanes_[lane_index];
    LayoutLane &lane = m_.lane(lane_index);
    Doc &doc = *job.doc;
    ParseStats &st = doc.gpu;
    const RenderPool::Done &d = job.done;
    const uint32_t text_len = d.units_off ? d.units_off : d.meta_len;
    if (!deserialize(pool_.meta(d.slot), text_len, job.pt, job.rt)) throw std::runtime_error("corrupt page metadata");
    if (d.units_off &&
        !tn::deserialize(pool_.meta(d.slot) + d.units_off, d.meta_len - d.units_off, job.work.units))
      throw std::runtime_error("corrupt page units");
    ++st.routes[job.rt.kind];
    st.text_ms += d.text_ms;
    st.render_ms += d.render_ms;

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
    ParseStats &st = job.doc->gpu;
    const bool region_det = job.ocr.plan.region_det;
    OcrPageResult ocr = finish_ocr(m_, job.ocr);
    job.rec_pending = false;
    st.ocr_ms += ocr.det_ms + ocr.rec_ms;
    st.det_ms += ocr.det_ms;
    st.rec_ms += ocr.rec_ms;
    st.det_boxes += ocr.det_boxes;
    st.rec_boxes += ocr.rec_boxes;
    st.direct_rec_boxes += ocr.direct_rec_boxes;
    if (region_det) {
      ++st.region_det_pages;
      st.region_det_canvas_px += ocr.canvas_px;
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
    Doc &doc = *job.doc;
    // start_next() has always queued this page's rec by now; kept for safety
    // (the other lane's rec first: rec has one set of buffers).
    if (job.det_pending) {
      if (LaneJob &other = lanes_[lane_index ^ 1]; other.rec_pending) complete_ocr(other);
      queue_rec(job);
    }
    if (job.rec_pending) complete_ocr(job);
    const auto t_wait = Clock::now();
    lane.stream.sync();
    doc.gpu.layout_wait_ms += ms_since(t_wait);
    PageWork &w = job.work;
    w.doc = &doc;
    w.result.layout = lane.layout.collect();
    w.result.ok = true;
    const bool live = !doc.failed.load();
    if (live && doc.formulas) copy_formula_crops(pool_.pixels(job.done.slot), job.done.w, job.done.h, w);
    if (live && doc.job_opt.tables && has_table_region(w.result.layout))
      w.grey = tn::grey_from_bgr(pool_.pixels(job.done.slot), job.done.w, job.done.h, static_cast<size_t>(job.done.w) * 3);
    w.result.ms.gpu = ms_since(job.t0);
    job.busy = false;
    free_slots_.push_back(job.done.slot);
    if (live) hand_on(std::move(w));
    else drop(doc, 1);
  }

  // ── assembly thread ───────────────────────────────────────────────────────
  // Drains every page queued so far, decodes all of their formula crops together,
  // then assembles and delivers each page.

  void stop_assembly() {
    {
      std::lock_guard l(amu_);
      astop_ = true;
    }
    acv_.notify_all();
    if (assembly_.joinable()) assembly_.join();
  }

  void assembly_loop() {
    for (;;) {
      std::vector<PageWork> batch;
      {
        std::unique_lock l(amu_);
        acv_.wait(l, [&] { return !aq_.empty() || astop_; });
        if (aq_.empty()) return;
        while (!aq_.empty()) {
          batch.push_back(std::move(aq_.front()));
          aq_.pop_front();
        }
      }
      process(batch);
    }
  }

  // Never throws: every page of the batch is delivered or dropped.
  void process(std::vector<PageWork> &batch) {
    auto t_last = Clock::now();
    auto busy = [&] {  // assembly time so far, counted before each page's document can complete
      const auto now = Clock::now();
      assembly_busy_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_last).count();
      t_last = now;
    };

    // Formula crops of the live documents' pages, decoded in one run. If that
    // fails, so does every document with crops in the batch.
    std::vector<size_t> first_crop(batch.size(), SIZE_MAX);
    std::vector<std::string> latex;
    try {
      std::vector<FormulaCrop> crops;
      std::vector<Doc *> crop_docs;
      for (size_t i = 0; i < batch.size(); ++i) {
        PageWork &w = batch[i];
        if (w.crop_px.empty() || w.doc->failed.load()) continue;
        first_crop[i] = crops.size();
        for (size_t k = 0; k < w.crop_px.size(); ++k)
          crops.push_back({w.crop_px[k].data(), static_cast<size_t>(w.crop_meta[k][1]) * 3, w.crop_meta[k][1],
                           w.crop_meta[k][2]});
        w.doc->asm_stats.formulas += static_cast<int>(w.crop_px.size());
        if (std::find(crop_docs.begin(), crop_docs.end(), w.doc) == crop_docs.end()) crop_docs.push_back(w.doc);
      }
      if (!crops.empty()) {
        const int steps0 = m_.formula().steps_run;
        const auto t = Clock::now();
        latex = m_.formula().run(crops, m_.formula_stream());
        const double ms = ms_since(t);
        // A shared run serves each of its documents in full.
        for (Doc *d : crop_docs) {
          d->asm_stats.formula_ms += ms;
          ++d->asm_stats.formula_batches;
          d->asm_stats.formula_steps += m_.formula().steps_run - steps0;
        }
      }
    } catch (...) {
      for (PageWork &w : batch)
        if (!w.crop_px.empty()) fail(*w.doc, std::current_exception());
    }

    for (size_t i = 0; i < batch.size(); ++i) {
      PageWork &w = batch[i];
      Doc &d = *w.doc;
      if (!d.failed.load()) {
        try {
          deliver(w, d, first_crop[i] == SIZE_MAX ? nullptr : latex.data() + first_crop[i]);
        } catch (...) {
          fail(d, std::current_exception());
        }
      }
      busy();
      finish_pages(d, 1);  // `d` may be gone now
    }
  }

  void deliver(PageWork &w, Doc &d, const std::string *latex) {
    ParseStats &st = d.asm_stats;
    PageResult &r = w.result;
    if (r.ok) {
      if (latex)
        for (size_t k = 0; k < w.crop_px.size(); ++k) r.formulas.push_back({w.crop_meta[k][0], latex[k], 1.f});
      std::vector<ModelTable> model_tables;
      if (!w.grey.empty()) {
        const auto tt = Clock::now();
        const float k = r.dpi / 72.f;
        const tn::PageInput in = table_input(w);
        for (tn::Table &t : tn::run_page(m_.tablenet(), in, m_.table_stream()))
          model_tables.push_back({static_cast<float>(t.bbox[0]) * k, static_cast<float>(t.bbox[1]) * k,
                                  static_cast<float>(t.bbox[2]) * k, static_cast<float>(t.bbox[3]) * k,
                                  std::move(t.grid), k});
        st.table_ms += ms_since(tt);
        ++st.table_pages;
        st.model_tables += static_cast<int>(model_tables.size());
      }
      const auto t = Clock::now();
      AssembledPage page = assemble_page(r.page, r.width, r.height, r.layout, w.items, merge_into_lines(w.items),
                                         r.formulas, model_tables, d.ordernet ? &m_.ordernet() : nullptr,
                                         m_.table_stream());
      const double ams = ms_since(t);
      st.assemble_ms += ams;
      if (page.ordered_by_model) ++st.order_pages, st.order_ms += ams;
      r.markdown = std::move(page.markdown);
      r.tables = std::move(page.tables);
      r.reading_order = std::move(page.order);
      r.lines = std::move(page.lines);
      st.lines += static_cast<int>(r.lines.size());
      if (w.rot) {
        const float W0 = static_cast<float>(w.w0), H0 = static_cast<float>(w.h0);
        for (LayoutBox &b : r.layout) box_to_original(b.x0, b.y0, b.x1, b.y1, w.rot, W0, H0);
        for (PageLine &l : r.lines) box_to_original(l.x0, l.y0, l.x1, l.y1, w.rot, W0, H0);
        r.width = w.w0, r.height = w.h0, r.rotation = w.rot;
      }
    }
    (*d.sink)(std::move(r));
  }

  ModelSet &m_;
  RenderPool &pool_;
  int wake_fd_ = -1;  // eventfd: new document, failed document, or stop

  // Guarded by mu_: document hand-over, completion, errors, document slots.
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<Doc *> incoming_;
  std::vector<int> free_doc_slots_;
  bool stop_ = false;

  // While-in-flight totals for ParseStats::render_wait_ms / assembly_busy_ms.
  std::atomic<int64_t> render_wait_ns_{0}, assembly_busy_ns_{0};

  // GPU thread only.
  std::vector<Doc *> docs_;  // documents that may still get or return page jobs
  size_t rr_ = 0;            // round-robin position in docs_
  std::vector<int> free_slots_;
  int outstanding_ = 0;      // page jobs submitted, Done not yet received
  bool pool_dead_ = false;
  std::array<LaneJob, ModelSet::kLanes> lanes_;
  int next_lane_ = 0;
  std::vector<Ready> ready_;  // rendered pages waiting for the GPU (start_next)
  uint64_t slot_turns_ = 0, gpu_turns_ = 0;

  // Assembly queue.
  std::mutex amu_;
  std::condition_variable acv_;
  std::deque<PageWork> aq_;
  bool astop_ = false;

  std::thread assembly_, gpu_;
};

PagePipeline::PagePipeline(ModelSet &models, RenderPool &pool) : impl_(std::make_unique<Impl>(models, pool)) {}
PagePipeline::~PagePipeline() = default;

ParseStats PagePipeline::parse(std::span<const uint8_t> pdf, const ParseOptions &opt, const PageSink &sink) {
  return impl_->parse(pdf, opt, sink);
}

}  // namespace hp
