#include "render_pool.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <clocale>
#include <cmath>
#include <csignal>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "pdf_render.h"
#include "text_layer.h"
#include "tablenet/units.h"

namespace hp {

struct RenderPool::Job {
  int32_t page, slot;
  uint32_t gen;
  int32_t ocr_dpi, force, tables, tables_dpi;
  uint64_t doc_len;
};

// One per worker, in shared memory. The worker publishes the job it is working
// on; the supervisor reads it to report the page if the worker dies or overruns
// the timeout. started_ns != 0 means "busy since"; it is stored with release
// order after the job fields and cleared only after the Done is written, so a
// crash can at worst produce a duplicate report, which the parent ignores
// (PagePipeline drops Dones for pages already completed).
struct RenderPool::WorkerState {
  std::atomic<int64_t> started_ns{0};
  int32_t page = 0, slot = 0;
  uint32_t gen = 0;
};
static_assert(std::atomic<int64_t>::is_always_lock_free, "cross-process atomics must be lock-free");

namespace {

using Clock = std::chrono::steady_clock;

// A worker that has worked and then sees no job for this long exits and is
// replaced by a fresh fork (frees pdfium's caches and the document copy).
constexpr int kIdleRecycleMs = 2000;
constexpr int kExitRecycle = 3;  // worker exit code: "replace me", not a crash

int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }

float ms_since(Clock::time_point t) { return std::chrono::duration<float, std::milli>(Clock::now() - t).count(); }

bool read_full(int fd, void *p, size_t n) {
  auto *c = static_cast<char *>(p);
  while (n) {
    const ssize_t r = ::read(fd, c, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    c += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

bool write_full(int fd, const void *p, size_t n) {
  const auto *c = static_cast<const char *>(p);
  while (n) {
    const ssize_t r = ::write(fd, c, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    c += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

// Block SIGPIPE on the
// calling thread while writing to a pipe whose reader may be gone, draining a
// pending instance afterwards. Without it, a write to a dead pool raises SIGPIPE
// and terminates the whole host process instead of returning EPIPE.
struct SigpipeBlocker {
  sigset_t old_set;
  bool blocked = false;
  SigpipeBlocker() {
    sigset_t pipe_set;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    if (pthread_sigmask(SIG_BLOCK, &pipe_set, &old_set) == 0) blocked = true;
  }
  ~SigpipeBlocker() {
    if (!blocked) return;
    sigset_t pipe_set;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    struct timespec zero{0, 0};
    while (sigtimedwait(&pipe_set, nullptr, &zero) >= 0) {
    }
    pthread_sigmask(SIG_SETMASK, &old_set, nullptr);
  }
};

void *map_shared(size_t bytes) {
  void *m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (m == MAP_FAILED) throw std::runtime_error("RenderPool: mmap failed");
  return m;
}

}  // namespace

RenderPool::RenderPool(const Options &opt) : opt_(opt) {
  static_assert(sizeof(Job) < PIPE_BUF && sizeof(Done) < PIPE_BUF, "pipe messages must be atomic");
  if (opt_.workers < 1 || opt_.slots < 1 || opt_.pixel_bytes == 0) throw std::invalid_argument("RenderPool: bad options");
  stride_ = (opt_.pixel_bytes + opt_.meta_bytes + 4095) / 4096 * 4096;
  map_bytes_ = stride_ * static_cast<size_t>(opt_.slots);
  base_ = static_cast<uint8_t *>(map_shared(map_bytes_));
  state_ = new (map_shared(sizeof(WorkerState) * opt_.workers)) WorkerState[opt_.workers];

  doc_fd_ = memfd_create("hp-document", MFD_CLOEXEC);
  int job[2], done[2];
  if (doc_fd_ < 0 || pipe2(job, O_CLOEXEC) || pipe2(done, O_CLOEXEC)) throw std::runtime_error("RenderPool: memfd/pipe failed");
  job_r_ = job[0], job_w_ = job[1], done_r_ = done[0], done_w_ = done[1];

  const pid_t parent = getpid();
  supervisor_ = fork();
  if (supervisor_ < 0) throw std::runtime_error("RenderPool: fork failed");
  if (supervisor_ == 0) {
    prctl(PR_SET_PDEATHSIG, SIGKILL);  // never outlive the engine process
    if (getppid() != parent) _exit(0);
    supervisor_main();
  }
  // The parent only writes jobs and reads results.
  ::close(job_r_);
  ::close(done_w_);
  job_r_ = done_w_ = -1;
}

RenderPool::~RenderPool() {
  // EOF on the job pipe ends every worker; the supervisor exits once they are
  // reaped. Grace period, then SIGKILL;
  // workers die with the supervisor through PR_SET_PDEATHSIG.
  if (job_w_ >= 0) ::close(job_w_);
  if (supervisor_ > 0) {
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    while (waitpid(supervisor_, nullptr, WNOHANG) == 0) {
      if (Clock::now() > deadline) {
        kill(supervisor_, SIGKILL);
        waitpid(supervisor_, nullptr, 0);
        break;
      }
      usleep(2000);
    }
  }
  if (done_r_ >= 0) ::close(done_r_);
  if (doc_fd_ >= 0) ::close(doc_fd_);
  if (pinned_) cudaHostUnregister(base_);
  if (base_) munmap(base_, map_bytes_);
  if (state_) munmap(state_, sizeof(WorkerState) * opt_.workers);
}

// ── parent side ─────────────────────────────────────────────────────────────

void RenderPool::release_document() {
  if (ftruncate(doc_fd_, 0) == 0) doc_len_ = 0;
}

uint32_t RenderPool::set_document(std::span<const uint8_t> pdf) {
  if (ftruncate(doc_fd_, static_cast<off_t>(pdf.size())) != 0) throw std::runtime_error("RenderPool: ftruncate failed");
  size_t off = 0;
  while (off < pdf.size()) {
    const ssize_t w = pwrite(doc_fd_, pdf.data() + off, pdf.size() - off, static_cast<off_t>(off));
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) throw std::runtime_error("RenderPool: document write failed");
    off += static_cast<size_t>(w);
  }
  doc_len_ = pdf.size();
  return ++gen_;
}

void RenderPool::submit(int page, int slot, uint32_t gen, const PageJobOptions &o) {
  const Job j{page, slot, gen, o.ocr_dpi, o.force, o.tables ? 1 : 0, o.tables_dpi, doc_len_};
  SigpipeBlocker no_sigpipe;
  if (!write_full(job_w_, &j, sizeof j)) throw std::runtime_error("RenderPool: job pipe closed (supervisor died)");
}

RenderPool::Done RenderPool::wait_done() {
  // The supervisor guarantees one Done per job (a crashed or timed-out worker's
  // page is reported for it), so a silent pipe means the pool itself is broken.
  // Bound the wait anyway so a host process can never hang here forever.
  const auto limit = opt_.page_timeout + std::chrono::seconds(30);
  pollfd pfd{done_r_, POLLIN, 0};
  const auto deadline = Clock::now() + limit;
  for (;;) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    const int pr = ::poll(&pfd, 1, static_cast<int>(std::max<int64_t>(0, left)));
    if (pr > 0) break;
    if (pr == 0) throw std::runtime_error("RenderPool: no page finished within the page timeout (pool stalled)");
    if (errno != EINTR) throw std::runtime_error("RenderPool: poll failed");
  }
  Done d{};
  if (!read_full(done_r_, &d, sizeof d)) throw std::runtime_error("RenderPool: result pipe closed (supervisor died)");
  return d;
}

void RenderPool::pin() {
  if (pinned_) return;
  if (cudaHostRegister(base_, map_bytes_, cudaHostRegisterDefault) == cudaSuccess) pinned_ = true;
  else (void)cudaGetLastError();  // still correct unpinned, just a staged (slower) H2D copy
}

void RenderPool::unpin_and_release() {
  if (pinned_) {
    cudaHostUnregister(base_);
    pinned_ = false;
  }
  // Return the slot pages to the OS (shmem: MADV_REMOVE frees the backing store).
  // Workers fault fresh zero pages in on their next render.
  madvise(base_, map_bytes_, MADV_REMOVE);
}

// ── supervisor ──────────────────────────────────────────────────────────────

void RenderPool::supervisor_main() {
  signal(SIGINT, SIG_IGN);  // Ctrl-C is the parent's call; it tears the pool down
  ::close(job_w_);           // EOF on the job pipe must depend on the parent alone
  ::close(done_r_);
  sigset_t chld;
  sigemptyset(&chld);
  sigaddset(&chld, SIGCHLD);
  sigprocmask(SIG_BLOCK, &chld, nullptr);

  // Workers poll() then read(); non-blocking so a lost race returns EAGAIN.
  fcntl(job_r_, F_SETFL, fcntl(job_r_, F_GETFL) | O_NONBLOCK);

  const int n = opt_.workers;
  std::vector<pid_t> pid(n);
  std::vector<char> timed_out(n, 0);
  for (int i = 0; i < n; ++i) pid[i] = spawn_worker(i);
  int alive = n;
  bool shutting_down = false;
  const int64_t timeout_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(opt_.page_timeout).count();

  for (;;) {
    const timespec tick{0, 100'000'000};  // timeout checks need ~0.1 s resolution only
    sigtimedwait(&chld, nullptr, &tick);
    int status = 0;
    for (pid_t p; (p = waitpid(-1, &status, WNOHANG)) > 0;) {
      const int i = static_cast<int>(std::find(pid.begin(), pid.end(), p) - pid.begin());
      if (i == n) continue;
      --alive;
      pid[i] = -1;
      if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {  // job pipe EOF: the parent is closing the pool
        shutting_down = true;
        continue;
      }
      if (shutting_down) continue;
      // Anything else (kExitRecycle from an idle worker, a signal, a nonzero exit)
      // gets a replacement; a page still marked in flight is reported as failed.
      WorkerState &s = state_[i];
      if (s.started_ns.load(std::memory_order_acquire) != 0) {
        Done d{};
        d.page = s.page;
        d.slot = s.slot;
        d.gen = s.gen;
        d.status = timed_out[i] ? kTimedOut : kWorkerCrashed;
        write_full(done_w_, &d, sizeof d);
        s.started_ns.store(0, std::memory_order_release);
      }
      timed_out[i] = 0;
      pid[i] = spawn_worker(i);
      ++alive;
    }
    if (shutting_down && alive == 0) _exit(0);
    const int64_t now = now_ns();
    for (int i = 0; i < n; ++i) {
      const int64_t t = state_[i].started_ns.load(std::memory_order_acquire);
      if (pid[i] > 0 && t != 0 && !timed_out[i] && now - t > timeout_ns) {
        timed_out[i] = 1;
        kill(pid[i], SIGKILL);
      }
    }
  }
}

pid_t RenderPool::spawn_worker(int index) {
  const pid_t supervisor = getpid();
  const pid_t p = fork();
  if (p == 0) {
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    if (getppid() != supervisor) _exit(0);
    sigset_t none;
    sigemptyset(&none);
    sigprocmask(SIG_SETMASK, &none, nullptr);
    worker_main(index);
  }
  return p;  // < 0: fork failed; the slot stays empty and the pool runs with fewer workers
}

// ── worker ──────────────────────────────────────────────────────────────────

void RenderPool::worker_main(int index) {
  signal(SIGINT, SIG_IGN);
  std::setlocale(LC_CTYPE, "C.UTF-8");  // iswalpha & co. for the legibility checks
  WorkerState &self = state_[index];
  std::vector<uint8_t> doc, meta;  // private copy of the document: see header
  std::unique_ptr<PdfRaster> raster;
  uint32_t doc_gen = 0;
  Job j{};
  for (;;) {
    // Wait for a job. The job pipe is shared and non-blocking (set by the
    // supervisor): one job wakes every idle worker, the losers get EAGAIN and
    // poll again. A blocking read would park them in read() for good, where the
    // idle timer below never fires (seen as 7 of 8 workers stuck in pipe_read).
    //
    // Idle after work: exit so the supervisor forks a fresh worker. Freeing the
    // document is not enough, because pdfium keeps its caches. Measured on
    // mix100.pdf (22 MB, 8 workers), anon RSS 2.5 s after the parse: 939 MiB
    // with the document freed, 770 MiB with malloc_trim as well. With
    // recycling, workers' Pss_Anon falls from a 674 MiB peak to 51 MiB (fresh
    // forks) within 1 s of the parse. A recycled worker that gets a job
    // mid-document reloads from the memfd, which stays valid until the parse ends.
    pollfd pfd{job_r_, POLLIN, 0};
    const int pr = ::poll(&pfd, 1, doc_gen != 0 ? kIdleRecycleMs : -1);
    if (pr == 0) _exit(kExitRecycle);
    if (pr < 0) {
      if (errno == EINTR) continue;
      _exit(1);
    }
    const ssize_t got = ::read(job_r_, &j, sizeof j);
    if (got == 0) break;  // EOF: the parent closed the pool
    if (got < 0) {
      if (errno == EAGAIN || errno == EINTR) continue;  // another worker took it
      _exit(1);
    }
    if (got != static_cast<ssize_t>(sizeof j)) _exit(1);  // impossible: jobs are written atomically
    self.page = j.page;
    self.slot = j.slot;
    self.gen = j.gen;
    self.started_ns.store(now_ns(), std::memory_order_release);

    if (j.gen != doc_gen) {
      raster.reset();
      doc.resize(j.doc_len);
      size_t off = 0;
      while (off < doc.size()) {
        const ssize_t r = pread(doc_fd_, doc.data() + off, doc.size() - off, static_cast<off_t>(off));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        off += static_cast<size_t>(r);
      }
      if (off == doc.size()) raster = std::make_unique<PdfRaster>(doc.data(), doc.size());
      doc_gen = j.gen;
    }

    Done d{j.page, j.slot, 0, 0, kRenderFailed, kNative, j.gen, 0, 0.f, 0.f, 0.f};
    if (raster && raster->ok()) {
      auto t = Clock::now();
      // LayoutNet units first: PDFium reports character boxes differently after extraction.
      tn::PageUnits units;
      if (j.tables) units = tn::native_units_doc(raster->doc(), j.page);
      PageText pt = extract_page_text(raster->doc(), j.page);
      Route rt = route_page(pt);
      if (j.force == 1 && rt.kind != kFullOcr) { rt.kind = kFullOcr; rt.reasons = 0; }
      if (j.force == 2) { rt.kind = kNative; rt.regions.clear(); }
      d.kind = rt.kind;
      d.text_ms = ms_since(t);

      // Render only as large as the consumers need: text-only pages feed just the
      // layout canvas; OCR pages render at ocr_dpi, capped by ocr_max_side and the slot.
      const float long_pt = std::max(std::max(pt.w_pt, pt.h_pt), 1.f);
      float dpi = rt.kind == kNative ? std::min<float>(j.ocr_dpi, opt_.canvas * 72.f / long_pt)
                                     : std::min<float>(j.ocr_dpi, opt_.ocr_max_side * 72.f / long_pt);
      if (j.tables && rt.kind == kNative && j.tables_dpi > 0)
        dpi = std::max(dpi, std::min<float>(j.tables_dpi, opt_.ocr_max_side * 72.f / long_pt));
      int w = 0, h = 0;
      raster->page_size_px(j.page, dpi, w, h);
      if (static_cast<size_t>(w) * h * 3 > opt_.pixel_bytes) {
        dpi *= static_cast<float>(std::sqrt(double(opt_.pixel_bytes) / (3.0 * w * h)) * 0.999);
        raster->page_size_px(j.page, dpi, w, h);
      }
      serialize(pt, rt, meta);
      if (j.tables) {
        d.units_off = static_cast<uint32_t>(meta.size());
        tn::serialize(units, meta);
      }
      t = Clock::now();
      uint8_t *slot = pixels(j.slot);
      const bool ok = meta.size() <= opt_.meta_bytes && raster->render_bgr(j.page, slot, w, h);
      if (ok) std::memcpy(slot + opt_.pixel_bytes, meta.data(), meta.size());
      d.render_ms = ms_since(t);
      d.w = w;
      d.h = h;
      d.dpi = dpi;
      d.status = ok ? kOk : kRenderFailed;
      d.meta_len = static_cast<uint32_t>(meta.size());
    }
    write_full(done_w_, &d, sizeof d);
    self.started_ns.store(0, std::memory_order_release);
  }
  _exit(0);
}

}  // namespace hp
