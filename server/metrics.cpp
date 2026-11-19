#include "metrics.h"

#include <cinttypes>
#include <cstdio>

#include <cuda_runtime_api.h>

#include "engine.h"
#include "job_queue.h"

namespace hp::server {
namespace {

constexpr const char *kRouteNames[] = {"/parse", "/parse/stream", "/models", "/health", "/metrics", "other"};
static_assert(std::size(kRouteNames) == Metrics::kRouteCount);

void header(std::string &out, const char *name, const char *type, const char *help) {
  out += "# HELP ";
  out += name;
  out += ' ';
  out += help;
  out += "\n# TYPE ";
  out += name;
  out += ' ';
  out += type;
  out += '\n';
}

void sample(std::string &out, std::string_view name, std::string_view labels, double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, " %.10g\n", v);
  out += name;
  if (!labels.empty()) {
    out += '{';
    out += labels;
    out += '}';
  }
  out += buf;
}

}  // namespace

void Histogram::observe(double s) {
  // _count and _sum first, then the bucket with release: a scrape that sees the
  // bucket also sees the count.
  count_.fetch_add(1, std::memory_order_relaxed);
  sum_us_.fetch_add(static_cast<uint64_t>(s * 1e6), std::memory_order_relaxed);
  for (size_t i = 0; i < kBuckets.size(); ++i)
    if (s <= kBuckets[i]) {
      buckets_[i].fetch_add(1, std::memory_order_release);
      break;
    }
}

void Histogram::serialize(std::string &out, std::string_view name, std::string_view labels) const {
  const std::string bucket = std::string(name) + "_bucket";
  const std::string sep = labels.empty() ? "" : std::string(labels) + ",";
  uint64_t cumulative = 0;
  for (size_t i = 0; i < kBuckets.size(); ++i) {
    cumulative += buckets_[i].load(std::memory_order_acquire);
    char le[32];
    std::snprintf(le, sizeof le, "le=\"%g\"", kBuckets[i]);
    sample(out, bucket, sep + le, static_cast<double>(cumulative));
  }
  const auto count = count_.load(std::memory_order_relaxed);
  sample(out, bucket, sep + "le=\"+Inf\"", static_cast<double>(count));
  sample(out, std::string(name) + "_sum", labels, sum_us_.load(std::memory_order_relaxed) / 1e6);
  sample(out, std::string(name) + "_count", labels, static_cast<double>(count));
}

Metrics::Route Metrics::route_from_path(std::string_view p) {
  if (p == "/parse") return kParse;
  if (p == "/parse/stream") return kParseStream;
  if (p == "/models" || p.starts_with("/models/")) return kModels;
  if (p == "/health" || p.starts_with("/health/")) return kHealth;
  if (p == "/metrics") return kMetrics;
  return kOther;
}

Metrics &Metrics::instance() {
  static Metrics m;
  return m;
}

void Metrics::record_request(Route r, int status, double seconds) {
  auto &c = routes_[r];
  if (status < 400) c.ok.fetch_add(1, std::memory_order_relaxed);
  else if (status < 500) c.client_err.fetch_add(1, std::memory_order_relaxed);
  else c.server_err.fetch_add(1, std::memory_order_relaxed);
  c.latency.observe(seconds);
}

void Metrics::record_parse(int pages, int failed, double queue_s, double parse_s) {
  documents_.fetch_add(1, std::memory_order_relaxed);
  pages_.fetch_add(pages, std::memory_order_relaxed);
  failed_pages_.fetch_add(failed, std::memory_order_relaxed);
  queue_wait_.observe(queue_s);
  parse_time_.observe(parse_s);
}

std::string Metrics::serialize(const Engine &engine, const JobQueue &queue) const {
  std::string out;
  out.reserve(8192);

  header(out, "hp_requests_total", "counter", "HTTP requests by route and status class.");
  for (int i = 0; i < kRouteCount; ++i) {
    const auto &c = routes_[i];
    const std::string r = std::string("route=\"") + kRouteNames[i] + "\"";
    sample(out, "hp_requests_total", r + ",status=\"2xx\"", static_cast<double>(c.ok.load()));
    sample(out, "hp_requests_total", r + ",status=\"4xx\"", static_cast<double>(c.client_err.load()));
    sample(out, "hp_requests_total", r + ",status=\"5xx\"", static_cast<double>(c.server_err.load()));
  }
  // Time to the response headers: whole parse for /parse, queue wait only for
  // /parse/stream (its body streams afterwards; see hp_parse_seconds).
  header(out, "hp_request_duration_seconds", "histogram", "Time from request to response headers.");
  for (int i = 0; i < kRouteCount; ++i)
    if (i != kHealth && i != kMetrics)
      routes_[i].latency.serialize(out, "hp_request_duration_seconds", std::string("route=\"") + kRouteNames[i] + "\"");

  header(out, "hp_request_bytes_total", "counter", "Request body bytes received.");
  sample(out, "hp_request_bytes_total", "", static_cast<double>(request_bytes_.load()));
  header(out, "hp_rejected_total", "counter", "Requests answered 503 because the queue was full.");
  sample(out, "hp_rejected_total", "", static_cast<double>(rejected_.load()));

  header(out, "hp_documents_total", "counter", "Documents parsed (including ones with failed pages).");
  sample(out, "hp_documents_total", "", static_cast<double>(documents_.load()));
  header(out, "hp_pages_total", "counter", "Pages parsed.");
  sample(out, "hp_pages_total", "", static_cast<double>(pages_.load()));
  header(out, "hp_pages_failed_total", "counter", "Pages that failed (render error, worker crash, page timeout).");
  sample(out, "hp_pages_failed_total", "", static_cast<double>(failed_pages_.load()));
  header(out, "hp_queue_wait_seconds", "histogram", "Time a parse waited for the engine.");
  queue_wait_.serialize(out, "hp_queue_wait_seconds", "");
  header(out, "hp_parse_seconds", "histogram", "Engine time per document, including any model load.");
  parse_time_.serialize(out, "hp_parse_seconds", "");

  header(out, "hp_queue_waiting", "gauge", "Requests waiting for the engine.");
  sample(out, "hp_queue_waiting", "", static_cast<double>(queue.waiting()));
  header(out, "hp_queue_running", "gauge", "Requests running on the engine (0 or 1).");
  sample(out, "hp_queue_running", "", static_cast<double>(queue.running()));
  header(out, "hp_queue_max_waiting", "gauge", "Queue ceiling; 503 above it.");
  sample(out, "hp_queue_max_waiting", "", static_cast<double>(queue.max_waiting()));

  const LifecycleStats life = engine.lifecycle();
  header(out, "hp_models_loaded", "gauge", "1 while the models are on the GPU.");
  sample(out, "hp_models_loaded", "", engine.loaded() ? 1 : 0);
  header(out, "hp_model_loads_total", "counter", "Model loads (explicit, warm-up or on demand).");
  sample(out, "hp_model_loads_total", "", life.loads);
  header(out, "hp_model_unloads_total", "counter", "Model unloads by cause.");
  sample(out, "hp_model_unloads_total", "cause=\"request\"", life.unloads - life.idle_unloads);
  sample(out, "hp_model_unloads_total", "cause=\"idle\"", life.idle_unloads);
  header(out, "hp_model_last_load_seconds", "gauge", "Duration of the last model load.");
  sample(out, "hp_model_last_load_seconds", "", life.last_load_ms / 1e3);
  header(out, "hp_model_last_unload_seconds", "gauge", "Duration of the last model unload.");
  sample(out, "hp_model_last_unload_seconds", "", life.last_unload_ms / 1e3);

  // Whole-device numbers (every process on the GPU), Read only
  // once a load has created this process's CUDA context: on a fresh server,
  // cudaMemGetInfo would create one (~106 MiB of device memory) just to report.
  if (life.loads > 0) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
      header(out, "hp_gpu_memory_used_bytes", "gauge", "Device memory in use by all processes (cudaMemGetInfo).");
      sample(out, "hp_gpu_memory_used_bytes", "", static_cast<double>(total_b - free_b));
      header(out, "hp_gpu_memory_total_bytes", "gauge", "Device memory.");
      sample(out, "hp_gpu_memory_total_bytes", "", static_cast<double>(total_b));
    }
  }
  return out;
}

}  // namespace hp::server
