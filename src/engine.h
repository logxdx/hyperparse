#pragma once
// hp::Engine: the public API. PDF bytes in; per-page layout, text, formulas
// and Markdown out.
//
//   hp::EngineConfig cfg;
//   cfg.models.dir = "models";
//   cfg.idle_unload = std::chrono::minutes(5);   // optional auto-unload
//   hp::Engine engine(cfg);                      // forks page workers; no GPU work yet
//   engine.warmup();                             // optional: load + pay first-use costs now
//   engine.parse(pdf_bytes, {}, [](hp::PageResult &&p) { ... });
//   engine.unload();                             // optional: give GPU + pinned memory back
//
// Lifecycle
//   unloaded ── load() / warmup() / parse() ──> loaded ── unload() / idle timeout ──> unloaded
//   * parse() loads on demand, so an unloaded engine (never loaded, unloaded,
//     or idle-unloaded) is always usable. That parse just pays the load time,
//     reported as ParseStats::load_ms.
//   * unload() destroys the ModelSet: every TensorRT engine, context and device
//     buffer, and the streams. It also unpins the page slot ring and returns its
//     pages to the OS. The page workers stay (they are CPU-only), and so does
//     the CUDA context, which this process may share with other CUDA users.
//   * Idle unload: a watchdog thread unloads once no load(), warmup() or parse()
//     has been running for `idle_unload`. It never interrupts an operation:
//     the watchdog only unloads when no operation is running or waiting, and
//     takes the engine exclusively to do it.
//
// Measured (RTX 4050 laptop 6 GB, models on an external drive, 960 layout, rec
// budget, FP16 formula step; tests/engine_lifecycle_test.cpp):
//   load     564-588 ms per reload with the 395 MB of plans in the page cache:
//            D-FINE x2 177-188, det 92-96, rec 97-104, formula 102-106, pin 95-97
//            (`loadtimes`, reload cycles 2-6 in one process).
//            The first load in a process: 777 ms (CUDA context + TRT runtime).
//            Plans evicted from the page cache: 1.67-1.77 s.
//            Most of a load is TRT deserialize + context + buffers, not I/O:
//            det is an 8.7 MB plan and still takes ~93 ms.
//   unload   126-133 ms: free engines/buffers 98-103, unpin + release slots 27-29.
//   memory   device 2996 MiB loaded -> 126 MiB unloaded (106 MiB CUDA context);
//            host shmem +615 MiB loaded -> +8-10 MiB unloaded (mix100.pdf run).
//   warmup   ~180 ms after a load. Without it, the first parse is slower than
//            steady state: 115-120 vs 58-62 ms (1-page scan), 176-178 vs 112-113
//            ms (1-page paper). With it, the first parse is at steady state
//            (`latency`, 3 fresh processes per mode).
//   Pinning (95 ms of each reload) is redone on every load so that unload also
//   returns the 615 MiB of locked host memory.
//
// Threading: all methods are thread-safe. Up to EngineConfig::concurrent_documents
// parse() calls run at once and share one GPU pipeline (page_pipeline.h): pages
// of several documents are in flight together, so one document's rendering
// overlaps another's GPU work and assembly. load(), warmup() and unload() run
// alone. Every operation waits its turn in arrival order (FIFO): an operation
// that cannot start yet (a lifecycle operation while parses run, a parse past the
// document limit or behind a lifecycle operation) holds up the ones behind it.
// The PageSink runs on one of the pipeline's assembly threads, never twice at
// once for one document; the threads are shared by every document in flight.
//
// Process model: the constructor forks the page-worker supervisor
// (render_pool.h). Construct the Engine early in the process lifetime, before
// other threads exist and ideally before any CUDA use in the host.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>

#include "model_set.h"
#include "parse_types.h"

namespace hp {

class RenderPool;
class PagePipeline;

struct EngineConfig {
  ModelConfig models;
  // Page worker processes. 0 = hardware threads - 2, clamped to [1, 8].
  int workers = 0;
  // Documents parsed at once (concurrent parse() calls); more wait their turn.
  // 1 = one document at a time, end to end.
  int concurrent_documents = 8;
  // Threads that assemble pages (LayoutNet, OrderNet, Markdown, sinks); formulas
  // have a thread of their own.
  int assembly_threads = 3;
  // A page whose text layer + render runs longer than this is killed and reported
  // as failed (120 s).
  std::chrono::milliseconds page_timeout{120000};
  // Unload the models after this long without activity. 0 = never.
  std::chrono::milliseconds idle_unload{0};
};

struct LifecycleStats {
  int loads = 0, unloads = 0, idle_unloads = 0;  // unloads includes idle_unloads
  // Last load: ModelSet construction (per model) + pinning the page slot ring.
  ModelLoadTimes last_load;
  double last_pin_ms = 0;
  double last_load_ms = 0;    // total, including CUDA context creation on the first load
  double last_warmup_ms = 0;  // ModelSet::warmup only
  // Last unload: ModelSet destruction (engines, contexts, buffers) + unpin/release.
  double last_unload_models_ms = 0;
  double last_unload_unpin_ms = 0;
  double last_unload_ms = 0;  // total
};

class Engine {
 public:
  explicit Engine(const EngineConfig &cfg);
  ~Engine();
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  // Load every model now; no-op when loaded. Throws if a model fails to load (the
  // engine stays unloaded and usable).
  void load();
  // load() if needed, then run the warm-up page through every GPU path.
  void warmup();
  // Release the models and pinned memory; no-op when unloaded. Waits for the
  // running parses to finish.
  void unload();
  [[nodiscard]] bool loaded() const;

  // Change the idle-unload timeout at runtime (0 = never); measured from the last
  // activity, not from this call.
  void set_idle_unload(std::chrono::milliseconds timeout);
  [[nodiscard]] std::chrono::milliseconds idle_unload() const;

  // Parse one PDF. Pages go to `sink` as they complete. Throws InvalidPdfError
  // for an unreadable PDF, PageLimitError past ParseOptions::max_pages, and
  // std::invalid_argument for options the loaded models cannot serve. Per-page
  // failures arrive as PageResult::ok == false instead.
  ParseStats parse(std::span<const uint8_t> pdf, const ParseOptions &opt, const PageSink &sink);

  [[nodiscard]] LifecycleStats lifecycle() const;
  [[nodiscard]] const EngineConfig &config() const { return cfg_; }

 private:
  class Activity;
  class Exclusive;
  bool enter(bool exclusive);
  void leave(bool exclusive);
  void load_locked();
  void unload_locked(bool idle);
  void watchdog();

  EngineConfig cfg_;
  std::unique_ptr<RenderPool> pool_;  // first: forked before any CUDA use, destroyed last
  std::unique_ptr<ModelSet> models_;  // null while unloaded; changed only by an exclusive operation
  std::unique_ptr<PagePipeline> pipeline_;  // with the models

  // Admission (enter/leave) and watchdog state, guarded by state_mu_.
  mutable std::mutex state_mu_;
  std::condition_variable state_cv_;
  uint64_t next_ticket_ = 0, serving_ = 0;  // FIFO turns
  int parsing_ = 0;                          // parses running
  bool exclusive_ = false;                   // load / warmup / unload (or a parse loading) running
  int active_ = 0;  // operations running or waiting for their turn
  bool loaded_ = false, stop_ = false;
  std::chrono::steady_clock::time_point last_use_;
  std::chrono::milliseconds idle_unload_{0};
  LifecycleStats life_;

  std::thread watchdog_;  // last: started after the pool has forked
};

}  // namespace hp
