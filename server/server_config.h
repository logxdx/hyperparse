#pragma once
// hyperparse_server configuration: command-line flags, each with an HP_* environment
// fallback (--max-queue <-> HP_MAX_QUEUE) so a container can be configured
// without a wrapper script. Flags win over the environment.

#include <chrono>
#include <string>

#include "engine.h"

namespace hp::server {

struct ServerConfig {
  std::string host = "0.0.0.0";
  int port = 8080;
  // Drogon I/O threads. They only parse HTTP, extract the PDF and enqueue; the
  // parse itself runs on the job thread. 4 = a sensible minimum of io_threads
  // (gpu_server_main.cpp: max(pool_size, 4)).
  int io_threads = 4;

  // Requests waiting for the engine (not counting the one running). Past this,
  // new parse/model requests get 503 SERVER_BUSY + Retry-After. Each waiting
  // request holds its PDF in memory, so the worst case is max_queue *
  // max_body_mb (16 * 100 MB = 1.6 GB); and at the measured 13-17 pages/s,
  // 16 queued 10-page documents is already ~10 s of waiting.
  int max_queue = 16;
  // A request that waited longer than this is answered 504 QUEUE_TIMEOUT
  // instead of being parsed (its client has likely given up). 0 = no limit.
  std::chrono::milliseconds queue_timeout{0};

  int max_body_mb = 100;
  int max_body_mem_mb = 100;  // bodies above this spill to a temp file (Drogon)
  int max_pages = 2000;       // 0 = no limit
  int shutdown_grace_s = 30;
  bool warmup = true;         // load + warm up before listening (first request at steady state)

  EngineConfig engine;
};

// Parses argv (and HP_* env). Throws std::runtime_error with a usage message.
ServerConfig parse_server_config(int argc, char **argv);

}  // namespace hp::server
