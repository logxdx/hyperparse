#pragma once
// The engine's request queue: every request that needs the engine (parse,
// load, warmup, unload) runs here, started in arrival order (FIFO).
//
// Parses run up to `max_running` at a time (EngineConfig::concurrent_documents):
// the engine overlaps their pages in one GPU pipeline, and each request's
// response is serialized on its own thread. Model operations (exclusive) start
// only once every job before them has finished, and nothing behind them starts
// until they are done. A job that cannot start yet holds up the jobs behind it,
// so the order never changes. Full queue -> PoolExhaustedError (-> 503
// SERVER_BUSY); `waiting` counts the jobs not yet started.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "core/common/errors.h"

namespace hp::server {

struct JobStart {
  double queue_ms = 0;   // time spent waiting in the queue
  bool expired = false;  // waited longer than the queue timeout: answer, don't run
};

class JobQueue {
 public:
  // max_waiting >= 1 requests may wait behind the running ones.
  JobQueue(size_t max_waiting, std::chrono::milliseconds max_wait, int max_running = 1)
      : max_wait_(max_wait), max_waiting_(std::max<size_t>(1, max_waiting)), max_running_(std::max(1, max_running)) {
    for (int i = 0; i < max_running_; ++i) threads_.emplace_back([this] { worker(); });
  }

  // Jobs still queued run (they see the server draining and answer at once).
  ~JobQueue() {
    {
      std::lock_guard l(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &t : threads_) t.join();
  }

  JobQueue(const JobQueue &) = delete;
  JobQueue &operator=(const JobQueue &) = delete;

  // Throws hp::core::PoolExhaustedError when the queue is full. `exclusive`: run
  // alone (model operations).
  void submit(std::function<void(const JobStart &)> job, bool exclusive = false) {
    {
      std::lock_guard l(mu_);
      if (q_.size() >= max_waiting_)
        throw hp::core::PoolExhaustedError("queue full");
      q_.push_back({std::move(job), std::chrono::steady_clock::now(), exclusive});
    }
    cv_.notify_all();
  }

  [[nodiscard]] size_t waiting() const {
    std::lock_guard l(mu_);
    return q_.size();
  }
  [[nodiscard]] size_t running() const {
    std::lock_guard l(mu_);
    return running_;
  }
  [[nodiscard]] size_t max_waiting() const { return max_waiting_; }
  [[nodiscard]] size_t max_running() const { return static_cast<size_t>(max_running_); }

  // True once nothing is waiting or running; false on timeout.
  bool wait_drain(std::chrono::milliseconds timeout) {
    std::unique_lock l(mu_);
    return drain_cv_.wait_for(l, timeout, [this] { return q_.empty() && running_ == 0; });
  }

 private:
  struct Item {
    std::function<void(const JobStart &)> fn;
    std::chrono::steady_clock::time_point queued;
    bool exclusive = false;
  };

  // The head of the queue may start: nothing exclusive running, and room for it.
  bool head_may_start() const {
    if (q_.empty() || exclusive_running_) return false;
    return q_.front().exclusive ? running_ == 0 : running_ < static_cast<size_t>(max_running_);
  }

  void worker() {
    for (;;) {
      Item item;
      {
        std::unique_lock l(mu_);
        cv_.wait(l, [this] { return head_may_start() || (stop_ && q_.empty()); });
        if (!head_may_start()) return;  // stopping, nothing left
        item = std::move(q_.front());
        q_.pop_front();
        ++running_;
        exclusive_running_ = item.exclusive;
      }
      cv_.notify_all();  // the next job may start beside this one
      const auto waited = std::chrono::steady_clock::now() - item.queued;
      JobStart s;
      s.queue_ms = std::chrono::duration<double, std::milli>(waited).count();
      s.expired = max_wait_.count() > 0 && waited > max_wait_;
      try {
        item.fn(s);
      } catch (...) {
        // Jobs answer their own errors; one escaping must not kill the thread.
      }
      {
        std::lock_guard l(mu_);
        --running_;
        if (item.exclusive) exclusive_running_ = false;
        if (q_.empty() && running_ == 0) drain_cv_.notify_all();
      }
      cv_.notify_all();
    }
  }

  const std::chrono::milliseconds max_wait_;
  const size_t max_waiting_;
  const int max_running_;
  mutable std::mutex mu_;
  std::condition_variable cv_, drain_cv_;
  std::deque<Item> q_;
  size_t running_ = 0;
  bool exclusive_running_ = false, stop_ = false;
  std::vector<std::thread> threads_;  // last: started after every other member
};

}  // namespace hp::server
