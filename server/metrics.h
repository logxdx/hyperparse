#pragma once
// Prometheus metrics for hyperparse_server (GET /metrics), with no dependencies.
//
// Lock-free counters; histograms publish count and sum before the bucket
// (release/acquire), so a scrape never sees a bucket above _count. Buckets are
// sized for whole-document parses (a 100-page PDF takes ~6 s at ~16 pages/s),
// plus engine gauges (queue, models loaded, load/unload counts and times).

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>

namespace hp {
class Engine;
}

namespace hp::server {

class JobQueue;

// Latency histogram over kBuckets (seconds). Thread-safe, lock-free.
class Histogram {
 public:
  static constexpr std::array<double, 12> kBuckets = {0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300};
  void observe(double seconds);
  void serialize(std::string &out, std::string_view name, std::string_view labels) const;

 private:
  std::array<std::atomic<uint64_t>, kBuckets.size()> buckets_{};
  std::atomic<uint64_t> sum_us_{0}, count_{0};
};

class Metrics {
 public:
  enum Route : int { kParse, kParseStream, kModels, kHealth, kMetrics, kOther, kRouteCount };
  static Route route_from_path(std::string_view path);

  static Metrics &instance();

  void record_request(Route r, int http_status, double seconds);
  void record_request_bytes(size_t n) { request_bytes_.fetch_add(n, std::memory_order_relaxed); }
  void record_rejected() { rejected_.fetch_add(1, std::memory_order_relaxed); }  // 503: queue full
  // One finished parse: pages, failed pages, time waiting in the queue, time parsing.
  void record_parse(int pages, int failed, double queue_s, double parse_s);

  // Text exposition; engine and queue gauges are read at scrape time.
  [[nodiscard]] std::string serialize(const Engine &engine, const JobQueue &queue) const;

 private:
  struct RouteCounters {
    std::atomic<uint64_t> ok{0}, client_err{0}, server_err{0};
    Histogram latency;
  };
  std::array<RouteCounters, kRouteCount> routes_{};
  std::atomic<uint64_t> request_bytes_{0}, rejected_{0};
  std::atomic<uint64_t> documents_{0}, pages_{0}, failed_pages_{0};
  Histogram queue_wait_, parse_time_;
};

}  // namespace hp::server
