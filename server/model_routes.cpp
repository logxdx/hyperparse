// /models: engine lifecycle over HTTP (API in routes.h).
//
// load/warmup/unload go through the job queue, so they run between documents
// in arrival order and never block a Drogon I/O thread (a load takes 0.6-1.8 s,
// an unload ~130 ms; engine.h). Status reads and the idle-timeout change only
// touch the engine's state mutex and are answered inline.

#include <memory>

#include <drogon/HttpAppFramework.h>

#include "metrics.h"
#include "request.h"
#include "routes.h"

namespace hp::server {
namespace {

nlohmann::json status_json(const ServerContext &ctx) {
  const Engine &e = ctx.engine;
  const LifecycleStats l = e.lifecycle();
  const ModelConfig &m = e.config().models;
  return {
      {"loaded", e.loaded()},
      {"idle_unload_s", e.idle_unload().count() / 1000.0},
      {"config",
       {{"dir", m.dir},
        {"layout_size", kLayoutSize},
        {"det_max_side", m.det_max_side},
        {"ocr", m.ocr},
        {"formula", m.formula},
        {"rec", m.rec == RecMode::kWidthBucket ? "bucket" : "budget"},
        {"page_workers", e.config().workers}}},
      {"lifecycle",
       {{"loads", l.loads},
        {"unloads", l.unloads},
        {"idle_unloads", l.idle_unloads},
        {"last_load_ms", l.last_load_ms},
        {"last_load",
         {{"layout_ms", l.last_load.layout_ms},
          {"det_ms", l.last_load.det_ms},
          {"rec_ms", l.last_load.rec_ms},
          {"formula_ms", l.last_load.formula_ms},
          {"pin_ms", l.last_pin_ms}}},
        {"last_warmup_ms", l.last_warmup_ms},
        {"last_unload_ms", l.last_unload_ms},
        {"last_unload", {{"models_ms", l.last_unload_models_ms}, {"unpin_ms", l.last_unload_unpin_ms}}}}},
      {"queue",
       {{"waiting", ctx.jobs.waiting()}, {"running", ctx.jobs.running()}, {"max_waiting", ctx.jobs.max_waiting()}}}};
}

// Queue `op` on the engine; answer with the status after it ran.
void queued_op(ServerContext &ctx, Callback &&callback, void (Engine::*op)(), const char *route) {
  if (ctx.draining.load())
    return callback(error_response({drogon::k503ServiceUnavailable, "SHUTTING_DOWN", "server is shutting down"}));
  auto cb = std::make_shared<Callback>(std::move(callback));
  try {
    ctx.jobs.submit([&ctx, cb, op, route](const JobStart &js) {
      try {
        if (ctx.draining.load()) throw ShuttingDown();
        if (js.expired)
          return (*cb)(error_response({drogon::k504GatewayTimeout, "QUEUE_TIMEOUT", "waited too long for the engine"}));
        (ctx.engine.*op)();
        (*cb)(json_response(status_json(ctx)));
      } catch (...) {
        (*cb)(error_response(classify(std::current_exception(), route)));
      }
    });
  } catch (const hp::core::PoolExhaustedError &) {
    Metrics::instance().record_rejected();
    (*cb)(error_response(queue_full(ctx)));
  }
}

}  // namespace

void register_model_routes(ServerContext &ctx) {
  auto &app = drogon::app();
  app.registerHandler(
      "/models",
      [&ctx](const drogon::HttpRequestPtr &, Callback &&cb) { cb(json_response(status_json(ctx))); },
      {drogon::Get});
  app.registerHandler(
      "/models/load",
      [&ctx](const drogon::HttpRequestPtr &, Callback &&cb) {
        queued_op(ctx, std::move(cb), &Engine::load, "/models/load");
      },
      {drogon::Post});
  app.registerHandler(
      "/models/warmup",
      [&ctx](const drogon::HttpRequestPtr &, Callback &&cb) {
        queued_op(ctx, std::move(cb), &Engine::warmup, "/models/warmup");
      },
      {drogon::Post});
  app.registerHandler(
      "/models/unload",
      [&ctx](const drogon::HttpRequestPtr &, Callback &&cb) {
        queued_op(ctx, std::move(cb), &Engine::unload, "/models/unload");
      },
      {drogon::Post});
  app.registerHandler(
      "/models/idle_unload",
      [&ctx](const drogon::HttpRequestPtr &req, Callback &&cb) {
        const std::string &v = req->getParameter("seconds");
        char *end = nullptr;
        const double s = std::strtod(v.c_str(), &end);
        if (v.empty() || *end != '\0' || !(s >= 0) || s > 365 * 86400.0)
          return cb(error_response({drogon::k400BadRequest, "INVALID_PARAMETER", "seconds must be a number >= 0"}));
        ctx.engine.set_idle_unload(std::chrono::milliseconds(static_cast<long long>(s * 1000)));
        cb(json_response(status_json(ctx)));
      },
      {drogon::Put});
}

}  // namespace hp::server
