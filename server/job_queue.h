#pragma once
// The engine's request queue: every request that needs the engine (parse,
// load, warmup, unload) runs here, one at a time, in arrival order.
//
// One thread on core/server's WorkPool: its bounded FIFO and
// PoolExhaustedError-on-full are the admission control (-> 503 SERVER_BUSY).
// One thread because Engine::parse holds the engine for a whole document, so
// more threads would only block on the engine mutex, which doesn't guarantee
// FIFO order. Pages within a document are still parallel: the engine fans them
// out to its page workers and two layout lanes.

#include <algorithm>
#include <chrono>
#include <functional>

#include "core/server/work_pool.h"

namespace hp::server {

struct JobStart {
  double queue_ms = 0;   // time spent waiting in the queue
  bool expired = false;  // waited longer than the queue timeout: answer, don't run
};

class JobQueue {
 public:
  // max_waiting >= 1 requests may wait behind the running one.
  JobQueue(size_t max_waiting, std::chrono::milliseconds max_wait)
      : max_wait_(max_wait), pool_(1, std::max<size_t>(1, max_waiting)) {}

  // Throws hp::core::PoolExhaustedError when the queue is full.
  void submit(std::function<void(const JobStart &)> job) {
    const auto queued = std::chrono::steady_clock::now();
    pool_.submit([this, queued, job = std::move(job)] {
      const auto waited = std::chrono::steady_clock::now() - queued;
      JobStart s;
      s.queue_ms = std::chrono::duration<double, std::milli>(waited).count();
      s.expired = max_wait_.count() > 0 && waited > max_wait_;
      job(s);
    });
  }

  [[nodiscard]] size_t waiting() const { return pool_.queue_depth(); }
  [[nodiscard]] size_t running() const { return pool_.inflight(); }
  [[nodiscard]] size_t max_waiting() const { return pool_.max_depth(); }
  // True once nothing is waiting or running; false on timeout.
  bool wait_drain(std::chrono::milliseconds timeout) { return pool_.wait_drain(timeout); }

 private:
  std::chrono::milliseconds max_wait_;
  hp::core::server::WorkPool pool_;  // last: its thread must stop before max_wait_ goes
};

}  // namespace hp::server
