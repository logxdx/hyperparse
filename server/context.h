#pragma once
// State shared by hyperparse_server's routes.

#include <atomic>
#include <stdexcept>

#include "engine.h"
#include "job_queue.h"
#include "server_config.h"

namespace hp::server {

struct ServerContext {
  Engine &engine;
  JobQueue &jobs;
  const ServerConfig &cfg;
  // SIGTERM/SIGINT: stop admitting work (new and queued requests get 503).
  std::atomic<bool> draining{false};
  // Grace period over: abort the running parse at its next page.
  std::atomic<bool> cancel{false};
};

// Thrown from a PageSink to abort a parse (Engine::parse rethrows it).
struct ShuttingDown : std::runtime_error {
  ShuttingDown() : std::runtime_error("server is shutting down") {}
};
struct ClientGone : std::runtime_error {
  ClientGone() : std::runtime_error("client disconnected") {}
};

}  // namespace hp::server
