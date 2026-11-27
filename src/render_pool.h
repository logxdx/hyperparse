#pragma once
// Persistent multi-process page workers: text layer -> routing -> render.
//
// Why processes: pdfium is process-global and not thread-safe (every call takes
// one global lock), so parallel PDF work needs one pdfium per process.
//
// Process tree (built once, by the constructor, before the parent touches CUDA):
//
//   parent (CUDA, engine) ── job pipe ──> worker × N ── done pipe ──> parent
//      └── supervisor (single-threaded; no CUDA, no pdfium)
//             └── forks the N workers; respawns a worker that crashes or overruns
//                 the page timeout and reports its page as failed, so the parent
//                 always receives exactly one Done per submitted job.
//
// Choices, with evidence:
//  * Persistent workers, not a fork per document. Forking after the parent has
//    created a CUDA context and helper threads is unsafe: the child inherits
//    locks held by threads that do not exist in it. The supervisor is forked while the parent is still
//    single-threaded and CUDA-free, so every later fork (including respawns
//    mid-parse) happens from a clean process.
//  * Crash/hang recovery. A long-lived engine must not hang on one bad PDF. A
//    crashed or stuck worker (120 s per page) is respawned at page granularity:
//    only the page in flight fails, and later pages go to the replacement worker.
//    tests/render_pool_test.cpp kills a worker and checks this.
//  * Document hand-off through anonymous memfds, one per document slot
//    (Options::documents), so several documents can be in flight at once: the
//    parent writes a document's bytes into a free slot; a worker copies them into
//    private memory when it first sees that document generation, and keeps the
//    last few documents open (pages of concurrent documents interleave). Unlike
//    a /dev/shm temp file, a memfd needs no path or cleanup and cannot leak files
//    when the process dies. The private copy means the parent can rewrite a slot
//    for the next document while an idle worker still has the old one open.
//  * Pixels go straight into a MAP_SHARED slot ring that the parent pins
//    (cudaHostRegister), so the H2D copy reads the worker's output in place:
//    no image files, no decoding.
//  * Pipe messages are fixed-size and smaller than PIPE_BUF, so each write is
//    atomic with many readers and writers on one pipe.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <sys/types.h>

namespace hp {

// Per-document render settings, sent with every job.
struct PageJobOptions {
  int ocr_dpi = 150;  // render DPI for pages with OCR work
  int force = 0;      // 0 = route each page, 1 = OCR every page, 2 = never OCR
  bool tables = false;  // build LayoutNet text units (tablenet/units.h) after the page meta
  int tables_dpi = 0;   // with tables: text-only pages render at least this DPI (LayoutNet's rules / image)
};

class RenderPool {
 public:
  struct Options {
    int workers = 4;
    int documents = 1;           // document slots: documents with pages in flight at once
    int slots = 10;              // pixel slots in flight (each pinned)
    size_t pixel_bytes = 0;      // per slot: largest page raster (BGR u8)
    size_t meta_bytes = 0;       // per slot: serialized text layer + route
    int canvas = 960;            // text-only pages render just large enough for this layout canvas
    int ocr_max_side = 2200;     // OCR pages are capped at this long side (huge scan MediaBoxes)
    std::chrono::milliseconds page_timeout{120000};  // per page; a stuck worker is killed
  };

  enum Status : int32_t { kOk = 0, kRenderFailed = 1, kWorkerCrashed = 2, kTimedOut = 3 };

  struct Done {
    int32_t page, slot, w, h, status, kind;
    uint32_t gen;  // document generation the job belonged to
    uint32_t meta_len;
    float dpi, text_ms, render_ms;
    uint32_t units_off = 0;  // tables: LayoutNet units start here in the meta (0 = none)
  };

  // A published document: what a job needs to find it.
  struct DocRef {
    uint32_t gen = 0;  // unique per publication
    int32_t slot = 0;  // document slot (memfd)
    uint64_t len = 0;
  };

  explicit RenderPool(const Options &opt);
  ~RenderPool();
  RenderPool(const RenderPool &) = delete;
  RenderPool &operator=(const RenderPool &) = delete;

  // Publish a document in document slot `doc_slot` (0..documents-1) and give it a
  // new generation. The slot must be free: no jobs outstanding for the document
  // it held before. Thread-safe for distinct slots, and against submit().
  DocRef set_document(int doc_slot, std::span<const uint8_t> pdf);
  // Empty a document slot's memfd. Call after the document drained; idle workers
  // drop their private copies on their own after 2 s.
  void release_document(int doc_slot);

  // One page of a published document into pixel slot `slot`. Single caller thread.
  void submit(int page, int slot, const DocRef &doc, const PageJobOptions &o);
  // Blocks for the next finished job. Throws if the pool is dead.
  Done wait_done();
  // Readable when a Done is pending (for poll() alongside other fds); read it with
  // read_done(), which blocks until one is there. Throws if the pool is dead.
  [[nodiscard]] int done_fd() const { return done_r_; }
  Done read_done();

  // Slot memory. Lock it (cudaHostRegister) after CUDA init for direct DMA; unlock
  // and give its pages back to the OS when the engine unloads.
  void pin();
  void unpin_and_release();
  [[nodiscard]] bool pinned() const { return pinned_; }

  [[nodiscard]] int slots() const { return opt_.slots; }
  [[nodiscard]] int documents() const { return opt_.documents; }
  [[nodiscard]] int workers() const { return opt_.workers; }
  [[nodiscard]] std::chrono::milliseconds page_timeout() const { return opt_.page_timeout; }
  [[nodiscard]] uint8_t *pixels(int s) const { return base_ + static_cast<size_t>(s) * stride_; }
  [[nodiscard]] const uint8_t *meta(int s) const { return pixels(s) + opt_.pixel_bytes; }

 private:
  struct Job;
  struct WorkerState;
  void supervisor_main();
  pid_t spawn_worker(int index);
  [[noreturn]] void worker_main(int index);

  Options opt_;
  uint8_t *base_ = nullptr;           // slot ring (MAP_SHARED)
  size_t map_bytes_ = 0, stride_ = 0;
  WorkerState *state_ = nullptr;      // per-worker in-flight job (MAP_SHARED), read by the supervisor
  std::vector<int> doc_fd_;           // memfd per document slot
  std::atomic<uint32_t> gen_{0};
  int job_r_ = -1, job_w_ = -1, done_r_ = -1, done_w_ = -1;
  pid_t supervisor_ = -1;
  bool pinned_ = false;
};

}  // namespace hp
