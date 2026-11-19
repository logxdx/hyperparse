#include "engine.h"

#include <algorithm>
#include <stdexcept>

#include "page_pipeline.h"
#include "render_pool.h"

namespace hp {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Slot geometry. A slot holds the largest page raster a worker may write:
// US Letter at 300 DPI. Bigger pages are rendered at a lower DPI to fit
// (RenderPool::worker_main). The meta area holds the serialized text layer + route.
constexpr size_t kSlotPixelBytes = static_cast<size_t>(2560) * 3328 * 3;
constexpr size_t kSlotMetaBytes = 8u << 20;

int default_workers() {
  return std::clamp(static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 8);
}

}  // namespace

// Marks an operation as running (or queued behind op_mu_) for the watchdog, and
// stamps the idle clock when it ends.
class Engine::Activity {
 public:
  explicit Activity(Engine &e) : e_(e) {
    std::lock_guard l(e_.state_mu_);
    ++e_.active_;
  }
  ~Activity() {
    {
      std::lock_guard l(e_.state_mu_);
      --e_.active_;
      e_.last_use_ = Clock::now();
    }
    e_.state_cv_.notify_all();
  }
  Activity(const Activity &) = delete;
  Activity &operator=(const Activity &) = delete;

 private:
  Engine &e_;
};

Engine::Engine(const EngineConfig &cfg) : cfg_(cfg), last_use_(Clock::now()), idle_unload_(cfg.idle_unload) {
  if (cfg_.workers <= 0) cfg_.workers = default_workers();
  RenderPool::Options po;
  po.workers = cfg_.workers;
  po.slots = 2 * cfg_.workers + 2;
  po.pixel_bytes = kSlotPixelBytes;
  po.meta_bytes = kSlotMetaBytes;
  po.canvas = kLayoutSize;
  po.page_timeout = cfg_.page_timeout;
  pool_ = std::make_unique<RenderPool>(po);  // forks here: no threads, no CUDA yet
  watchdog_ = std::thread([this] { watchdog(); });
}

Engine::~Engine() {
  {
    std::lock_guard l(state_mu_);
    stop_ = true;
  }
  state_cv_.notify_all();
  watchdog_.join();
  std::lock_guard op(op_mu_);
  models_.reset();  // before the pool: the slot ring is unregistered after the GPU is idle
  pool_.reset();
}

// ── lifecycle ───────────────────────────────────────────────────────────────

void Engine::load_locked() {
  if (models_) return;
  const auto t = Clock::now();
  auto models = std::make_unique<ModelSet>(cfg_.models);  // throws -> stays unloaded
  const auto t_pin = Clock::now();
  pool_->pin();
  const double pin_ms = ms_since(t_pin);
  models_ = std::move(models);
  std::lock_guard l(state_mu_);
  loaded_ = true;
  ++life_.loads;
  life_.last_load = models_->load_times();
  life_.last_pin_ms = pin_ms;
  life_.last_load_ms = ms_since(t);
}

void Engine::unload_locked(bool idle) {
  if (!models_) return;
  const auto t = Clock::now();
  models_.reset();
  const double models_ms = ms_since(t);
  const auto t_unpin = Clock::now();
  pool_->unpin_and_release();
  std::lock_guard l(state_mu_);
  loaded_ = false;
  ++life_.unloads;
  if (idle) ++life_.idle_unloads;
  life_.last_unload_models_ms = models_ms;
  life_.last_unload_unpin_ms = ms_since(t_unpin);
  life_.last_unload_ms = ms_since(t);
}

void Engine::load() {
  Activity a(*this);
  std::lock_guard op(op_mu_);
  load_locked();
}

void Engine::warmup() {
  Activity a(*this);
  std::lock_guard op(op_mu_);
  load_locked();
  const auto t = Clock::now();
  models_->warmup();
  std::lock_guard l(state_mu_);
  life_.last_warmup_ms = ms_since(t);
}

void Engine::unload() {
  Activity a(*this);
  std::lock_guard op(op_mu_);
  unload_locked(false);
}

bool Engine::loaded() const {
  std::lock_guard l(state_mu_);
  return loaded_;
}

void Engine::set_idle_unload(std::chrono::milliseconds timeout) {
  {
    std::lock_guard l(state_mu_);
    idle_unload_ = std::max(timeout, std::chrono::milliseconds(0));
  }
  state_cv_.notify_all();
}

std::chrono::milliseconds Engine::idle_unload() const {
  std::lock_guard l(state_mu_);
  return idle_unload_;
}

LifecycleStats Engine::lifecycle() const {
  std::lock_guard l(state_mu_);
  return life_;
}

// Sleeps until the idle deadline (or a state change), then unloads if the
// engine is still loaded and idle. A parse that starts while the watchdog is
// waiting for op_mu_ is seen through active_ and cancels the unload.
void Engine::watchdog() {
  std::unique_lock l(state_mu_);
  while (!stop_) {
    if (!loaded_ || idle_unload_.count() == 0 || active_ > 0) {
      state_cv_.wait(l);
      continue;
    }
    const auto deadline = last_use_ + idle_unload_;
    if (Clock::now() < deadline) {
      state_cv_.wait_until(l, deadline);
      continue;  // re-evaluate: activity, timeout change or stop may have happened
    }
    l.unlock();
    {
      std::lock_guard op(op_mu_);
      bool still_idle;
      {
        std::lock_guard s(state_mu_);
        still_idle = !stop_ && active_ == 0 && idle_unload_.count() > 0 &&
                     Clock::now() >= last_use_ + idle_unload_;
      }
      if (still_idle) unload_locked(true);
    }
    l.lock();
  }
}

// ── work ────────────────────────────────────────────────────────────────────

ParseStats Engine::parse(std::span<const uint8_t> pdf, const ParseOptions &opt, const PageSink &sink) {
  if (opt.ocr == OcrMode::kAll && !cfg_.models.ocr)
    throw std::invalid_argument("OcrMode::kAll needs an engine with ModelConfig::ocr = true");
  if (opt.ocr_dpi < 36 || opt.ocr_dpi > 600) throw std::invalid_argument("ocr_dpi must be in [36, 600]");
  Activity a(*this);
  std::lock_guard op(op_mu_);
  const auto t = Clock::now();
  const bool cold = !models_;
  load_locked();
  const double load_ms = cold ? ms_since(t) : 0.0;
  ParseStats stats = parse_document(*models_, *pool_, pdf, opt, sink);
  stats.load_ms = load_ms;
  stats.wall_ms += load_ms;
  return stats;
}

}  // namespace hp
