#include "bootstrap.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string_view>
#include <thread>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <drogon/HttpAppFramework.h>

#include "core/common/log/logger.h"

namespace hp::server {
namespace {

// After the grace period the running parse is cancelled at its next page.
// Pages complete every ~60-100 ms in steady state, so a few seconds suffice
// unless a page is stuck in a worker (bounded by EngineConfig::page_timeout).
constexpr std::chrono::seconds kCancelWait{5};

void begin_graceful_shutdown(ServerContext &ctx, const char *signal_name) {
  if (ctx.draining.exchange(true)) return;
  HP_LOG_INFO("Graceful shutdown requested", "signal", std::string_view(signal_name), "grace_seconds",
                ctx.cfg.shutdown_grace_s);
  std::thread([&ctx] {
    bool drained = ctx.jobs.wait_drain(std::chrono::seconds(ctx.cfg.shutdown_grace_s));
    if (!drained) {
      ctx.cancel = true;
      drained = ctx.jobs.wait_drain(kCancelWait);
    }
    HP_LOG_INFO("Work drained", "drained", drained);
    drogon::app().quit();
  }).detach();
}

}  // namespace

// glibc tuning for a long-lived server: under bursts of large requests glibc parks freed multi-MB buffers in per-arena
// free lists and its dynamic mmap threshold ratchets upward, so RSS climbs to
// the high-water mark and stays. A fixed 1 MiB mmap threshold returns every
// large buffer (here: PDF bodies, decoded uploads, JSON responses) to the OS on
// free; the trim threshold and arena cap bound the rest, and a 5 s malloc_trim
// returns idle free pages. Not re-measured for hp.
void tune_glibc_arenas() {
#if defined(__GLIBC__)
  mallopt(M_MMAP_THRESHOLD, 1 * 1024 * 1024);
  mallopt(M_TRIM_THRESHOLD, 4 * 1024 * 1024);
  mallopt(M_ARENA_MAX, 8);
  if (!std::getenv("HP_DISABLE_MALLOC_REAPER")) {
    std::thread([] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        malloc_trim(0);
      }
    }).detach();
  }
#endif
}

void run_http_server(ServerContext &ctx) {
  const ServerConfig &c = ctx.cfg;
  drogon::app()
      .setLogLevel(trantor::Logger::kWarn)
      .setTermSignalHandler([&ctx] { begin_graceful_shutdown(ctx, "SIGTERM"); })
      .setIntSignalHandler([&ctx] { begin_graceful_shutdown(ctx, "SIGINT"); })
      .addListener(c.host, static_cast<uint16_t>(c.port))
      .setThreadNum(static_cast<size_t>(c.io_threads))
      .setIdleConnectionTimeout(120)
      .setClientMaxBodySize(static_cast<size_t>(c.max_body_mb) << 20)
      .setClientMaxMemoryBodySize(static_cast<size_t>(c.max_body_mem_mb) << 20)
      // Drogon creates <upload path>/tmp/00..ff at startup (default: ./uploads in
      // the working directory) and spills bodies above the memory limit there.
      .setUploadPath((std::filesystem::temp_directory_path() / "hyperparse_server-uploads").string())
      .run();
}

}  // namespace hp::server
