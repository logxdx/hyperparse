// POST /parse and POST /parse/stream (API in routes.h).

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <drogon/HttpAppFramework.h>

#include "page_json.h"
#include "metrics.h"
#include "request.h"
#include "routes.h"
#include "core/common/log/logger.h"

namespace hp::server {
namespace {

// Drogon calls the stream callback when it writes the response headers, right
// after the handler returns. Ten seconds only ever runs out when the
// connection closed before that.
constexpr std::chrono::seconds kStreamAttachTimeout{10};

// Reasons not to start a job that reached the front of the queue.
std::optional<ErrorInfo> refuse(const ServerContext &ctx, const JobStart &js) {
  if (ctx.draining.load()) return ErrorInfo{drogon::k503ServiceUnavailable, "SHUTTING_DOWN", "server is shutting down"};
  if (js.expired)
    return ErrorInfo{drogon::k504GatewayTimeout, "QUEUE_TIMEOUT",
                     "waited " + std::to_string(static_cast<int>(js.queue_ms)) + " ms for the engine"};
  return std::nullopt;
}

// Engine::parse with shutdown cancellation checked at every page.
ParseStats run_parse(ServerContext &ctx, ParseRequest &pr, const PageSink &sink) {
  return ctx.engine.parse(pr.pdf.bytes(), pr.opt, [&](PageResult &&p) {
    if (ctx.cancel.load()) throw ShuttingDown();
    sink(std::move(p));
  });
}

void record(const ParseStats &s, const JobStart &js) {
  Metrics::instance().record_parse(s.pages, s.failed, js.queue_ms / 1e3, s.wall_ms / 1e3);
}

nlohmann::json stats_json(const ParseStats &s, const JobStart &js) {
  nlohmann::json j = stats_to_json(s);
  j["queue_ms"] = js.queue_ms;
  return j;
}

// Queue full -> 503 SERVER_BUSY now; Retry-After is added by the middleware.
bool enqueue(ServerContext &ctx, std::function<void(const JobStart &)> job, const Callback &cb) {
  try {
    ctx.jobs.submit(std::move(job));
    return true;
  } catch (const hp::core::PoolExhaustedError &) {
    Metrics::instance().record_rejected();
    cb(error_response(queue_full(ctx)));
    return false;
  }
}

drogon::HttpResponsePtr markdown_response(const std::vector<PageResult> &pages) {
  std::string md;
  for (const auto &p : pages) {
    md += "<!-- page " + std::to_string(p.page + 1);
    md += p.ok ? " -->\n\n" + p.markdown : ": failed: " + p.error + " -->";
    md += "\n\n";
  }
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setBody(std::move(md));
  resp->setContentTypeString("text/markdown; charset=utf-8");
  return resp;
}

// ── /parse ──────────────────────────────────────────────────────────────────

void parse_job(ServerContext &ctx, ParseRequest &pr, const JobStart &js, const Callback &cb) {
  if (auto e = refuse(ctx, js)) return cb(error_response(*e));
  std::vector<PageResult> pages;
  ParseStats stats;
  try {
    stats = run_parse(ctx, pr, [&](PageResult &&p) { pages.push_back(std::move(p)); });
  } catch (...) {
    return cb(error_response(classify(std::current_exception(), "/parse")));
  }
  record(stats, js);
  std::sort(pages.begin(), pages.end(), [](const PageResult &a, const PageResult &b) { return a.page < b.page; });
  if (pr.markdown_only) return cb(markdown_response(pages));

  nlohmann::json out = {{"pages", nlohmann::json::array()}, {"stats", stats_json(stats, js)}};
  for (const auto &p : pages) out["pages"].push_back(page_to_json(p, pr.with_markdown));
  cb(json_response(out));
}

void handle_parse(ServerContext &ctx, const drogon::HttpRequestPtr &req, Callback &&callback) {
  auto pr = std::make_shared<ParseRequest>();
  if (!read_parse_request(req, ctx, /*stream=*/false, *pr, callback)) return;
  if (ctx.draining.load()) return callback(error_response(*refuse(ctx, {})));
  auto cb = std::make_shared<Callback>(std::move(callback));
  enqueue(
      ctx,
      [&ctx, pr, cb](const JobStart &js) {
        try {
          parse_job(ctx, *pr, js, *cb);
        } catch (...) {  // e.g. JSON serialization: the client must still get an answer
          (*cb)(error_response(classify(std::current_exception(), "/parse")));
        }
      },
      *cb);
}

// ── /parse/stream ───────────────────────────────────────────────────────────

void stream_job(ServerContext &ctx, ParseRequest &pr, NdjsonStream &out, const JobStart &js) {
  if (!out.wait_attached(kStreamAttachTimeout)) return;  // nobody to send to
  auto send_error = [&](const ErrorInfo &e) {
    out.send_line({{"event", "error"}, {"code", e.code}, {"message", e.message}});
  };
  if (auto e = refuse(ctx, js)) {
    send_error(*e);
    return out.finish();
  }
  try {
    pr.opt.on_start = [&](int pages) {
      if (!out.send_line({{"event", "meta"}, {"pages", pages}, {"queue_ms", js.queue_ms}})) throw ClientGone();
    };
    const ParseStats s = run_parse(ctx, pr, [&](PageResult &&p) {
      nlohmann::json j = page_to_json(p, pr.with_markdown);
      j["event"] = p.ok ? "page" : "page_error";
      // A client that stops reading aborts the parse at the next page, freeing
      // the engine for the queue.
      if (!out.send_line(j)) throw ClientGone();
    });
    record(s, js);
    out.send_line({{"event", "end"}, {"pages", s.pages}, {"failed", s.failed}, {"stats", stats_json(s, js)}});
  } catch (...) {
    const ErrorInfo e = classify(std::current_exception(), "/parse/stream");
    if (std::strcmp(e.code, "CLIENT_GONE") == 0) HP_LOG_INFO("Stream client disconnected; parse aborted");
    else send_error(e);
  }
  out.finish();
}

void handle_stream(ServerContext &ctx, const drogon::HttpRequestPtr &req, Callback &&callback) {
  auto pr = std::make_shared<ParseRequest>();
  if (!read_parse_request(req, ctx, /*stream=*/true, *pr, callback)) return;
  if (ctx.draining.load()) return callback(error_response(*refuse(ctx, {})));
  // Admission before the 200 goes out, so a full queue is a real 503 rather
  // than an error event in a successful response (submitting inside the stream
  // callback could only report SERVER_BUSY in-band).
  auto out = std::make_shared<NdjsonStream>();
  if (!enqueue(ctx, [&ctx, pr, out](const JobStart &js) { stream_job(ctx, *pr, *out, js); }, callback)) return;
  auto resp = drogon::HttpResponse::newAsyncStreamResponse(
      [out](drogon::ResponseStreamPtr s) { out->attach(std::move(s)); }, /*disableKickoffTimeout=*/true);
  resp->setContentTypeString("application/x-ndjson");
  callback(resp);
}

}  // namespace

void register_parse_routes(ServerContext &ctx) {
  drogon::app().registerHandler(
      "/parse",
      [&ctx](const drogon::HttpRequestPtr &req, Callback &&cb) { handle_parse(ctx, req, std::move(cb)); },
      {drogon::Post});
  drogon::app().registerHandler(
      "/parse/stream",
      [&ctx](const drogon::HttpRequestPtr &req, Callback &&cb) { handle_stream(ctx, req, std::move(cb)); },
      {drogon::Post});
}

}  // namespace hp::server
