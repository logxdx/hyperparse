// Health, metrics, API docs, and the observability middleware (API in routes.h).

#include <charconv>
#include <chrono>
#include <string>

#include <drogon/HttpAppFramework.h>

#include "api_docs.h"
#include "metrics.h"
#include "request.h"
#include "routes.h"
#include "core/common/uuid.h"
#include "core/server/http_responses.h"

namespace hp::server {
namespace {

// Observability middleware: request id in and out, X-Inference-Time-Ms,
// Retry-After on 503, per-route metrics. The start time rides on the request
// as a header.
void register_middleware() {
  drogon::app().registerPreHandlingAdvice([](const drogon::HttpRequestPtr &req) {
    auto id = req->getHeader("X-Request-Id");
    if (id.empty()) id = hp::core::generate_uuid_v7();
    req->addHeader("X-Request-Id", id);
    req->addHeader("X-Start-Ns", std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  });
  drogon::app().registerPostHandlingAdvice([](const drogon::HttpRequestPtr &req, const drogon::HttpResponsePtr &resp) {
    if (const auto &id = req->getHeader("X-Request-Id"); !id.empty()) resp->addHeader("X-Request-Id", id);
    double seconds = 0;
    const std::string &start = req->getHeader("X-Start-Ns");
    long long start_ns = 0;
    if (auto [p, ec] = std::from_chars(start.data(), start.data() + start.size(), start_ns);
        ec == std::errc{} && p == start.data() + start.size()) {
      const long long ns = std::chrono::steady_clock::now().time_since_epoch().count() - start_ns;
      resp->addHeader("X-Inference-Time-Ms", std::to_string(ns / 1'000'000));
      seconds = ns / 1e9;
    }
    // One document takes ~0.1-10 s, so 1 s is a reasonable first retry.
    if (resp->statusCode() == drogon::k503ServiceUnavailable) resp->addHeader("Retry-After", "1");
    Metrics::instance().record_request(Metrics::route_from_path(req->path()), static_cast<int>(resp->statusCode()),
                                       seconds);
    Metrics::instance().record_request_bytes(req->body().size());
  });
}

}  // namespace

void register_admin_routes(ServerContext &ctx) {
  register_middleware();
  auto &app = drogon::app();
  auto ok = [](const drogon::HttpRequestPtr &, Callback &&cb) {
    cb(hp::core::server::make_response(drogon::k200OK, "ok"));
  };
  const auto started = std::chrono::steady_clock::now();
  app.registerHandler(
      "/health",
      [&ctx, started](const drogon::HttpRequestPtr &, Callback &&cb) {
        const double up = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        cb(json_response({{"status", ctx.draining.load() ? "shutting_down" : "ok"},
                          {"models_loaded", ctx.engine.loaded()},
                          {"queue",
                           {{"waiting", ctx.jobs.waiting()},
                            {"running", ctx.jobs.running()},
                            {"max_waiting", ctx.jobs.max_waiting()},
                            {"max_running", ctx.jobs.max_running()}}},
                          {"uptime_s", static_cast<int64_t>(up)}}));
      },
      {drogon::Get});
  app.registerHandler("/health/live", ok, {drogon::Get});
  app.registerHandler(
      "/health/ready",
      [&ctx](const drogon::HttpRequestPtr &, Callback &&cb) {
        if (ctx.draining.load())
          return cb(error_response({drogon::k503ServiceUnavailable, "NOT_READY", "server is shutting down"}));
        cb(hp::core::server::make_response(drogon::k200OK, "ok"));
      },
      {drogon::Get});
  app.registerHandler(
      "/metrics",
      [&ctx](const drogon::HttpRequestPtr &, Callback &&cb) {
        auto resp = hp::core::server::make_response(drogon::k200OK, Metrics::instance().serialize(ctx.engine, ctx.jobs));
        resp->setContentTypeString("text/plain; version=0.0.4; charset=utf-8");
        cb(resp);
      },
      {drogon::Get});

  // Self-describing API: GET / (JSON index), GET /docs (text), JSON 404 pointing at both.
  app.registerHandler(
      "/", [](const drogon::HttpRequestPtr &, Callback &&cb) { cb(json_response(api_index())); }, {drogon::Get});
  app.registerHandler(
      "/docs",
      [&ctx](const drogon::HttpRequestPtr &req, Callback &&cb) {
        const std::string host = req->getHeader("host");
        auto resp = hp::core::server::make_response(
            drogon::k200OK, api_docs(ctx.cfg, "http://" + (host.empty() ? "localhost:" + std::to_string(ctx.cfg.port) : host)));
        resp->setContentTypeString("text/plain; charset=utf-8");
        cb(resp);
      },
      {drogon::Get});
  auto not_found = error_response({drogon::k404NotFound, "NOT_FOUND", "no such endpoint; see GET / or GET /docs"});
  app.setCustom404Page(not_found);
}

}  // namespace hp::server
