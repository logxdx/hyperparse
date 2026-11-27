// hyperparse_server: hp::Engine over HTTP (Drogon). API: routes.h; flags: server_config.cpp.
//
//   hyperparse_server [--port 8080] [--idle-unload 300] [--max-queue 16] [--models DIR] ...
//   curl --data-binary @doc.pdf -H 'Content-Type: application/pdf' localhost:8080/parse

#include <clocale>
#include <cstdio>
#include <exception>
#include <string>

#include "bootstrap.h"
#include "model_store.h"
#include "routes.h"
#include "core/common/log/logger.h"

int main(int argc, char **argv) try {
  std::setlocale(LC_CTYPE, "C.UTF-8");
  hp::server::ServerConfig cfg = hp::server::parse_server_config(argc, argv);
  // Before the Engine: downloads/builds missing models in a child process.
  cfg.engine.models.dir = hp::prepare_models(cfg.engine.models.dir, {cfg.engine.models.ocr, cfg.engine.models.formula, cfg.engine.models.tables,
                                                                         cfg.engine.models.ordernet,
                                                                         cfg.engine.models.tables_precision,
                                                                         cfg.engine.models.order_precision,
                                                                       cfg.engine.models.rec == hp::RecMode::kWidthBucket});

  // First: the Engine forks its page-worker supervisor, which must happen
  // before any other thread exists (render_pool.h).
  hp::Engine engine(cfg.engine);
  if (cfg.warmup) {
    engine.warmup();  // fail fast on missing/broken models; first request at steady state
    const auto life = engine.lifecycle();
    HP_LOG_INFO("Models loaded", "load_ms", life.last_load_ms, "warmup_ms", life.last_warmup_ms);
  }
  hp::server::tune_glibc_arenas();

  hp::server::JobQueue jobs(static_cast<size_t>(cfg.max_queue), cfg.queue_timeout, engine.config().concurrent_documents);
  hp::server::ServerContext ctx{engine, jobs, cfg};
  hp::server::register_admin_routes(ctx);
  hp::server::register_parse_routes(ctx);
  hp::server::register_model_routes(ctx);

  HP_LOG_INFO("hyperparse_server listening", "host", std::string_view(cfg.host), "port", cfg.port, "page_workers",
                engine.config().workers, "concurrent_docs", engine.config().concurrent_documents, "max_queue", cfg.max_queue, "idle_unload_s",
                static_cast<int>(cfg.engine.idle_unload.count() / 1000));
  hp::server::run_http_server(ctx);

  // Drogon has stopped. Queued jobs still in the queue see `draining` and
  // answer at once; then the queue thread joins and the Engine unloads.
  ctx.draining = true;
  ctx.cancel = true;
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
