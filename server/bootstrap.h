#pragma once
// Process setup and the Drogon run loop with graceful shutdown.

#include "context.h"

namespace hp::server {

// glibc malloc settings for a long-running server holding multi-MB request
// bodies and responses (see the .cpp). Call after the Engine has forked its page
// workers, so the workers keep the allocator settings they were measured with.
void tune_glibc_arenas();

// Configures Drogon from ctx.cfg and runs it until SIGTERM/SIGINT. On a signal:
// stop admitting work, give running/queued work shutdown_grace_s to finish,
// then cancel the running parse and quit.
void run_http_server(ServerContext &ctx);

}  // namespace hp::server
