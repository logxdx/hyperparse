#pragma once
// Request decoding, error mapping and the NDJSON stream writer shared by the
// routes.

#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <span>
#include <string>

#include <drogon/HttpRequest.h>
#include <drogon/HttpResponse.h>
#include <nlohmann/json.hpp>

#include "context.h"

namespace hp::server {

using Callback = std::function<void(const drogon::HttpResponsePtr &)>;

// The uploaded PDF. Raw bodies are read in place from the request (kept alive
// here, so a queued job needs no copy); multipart and base64 uploads are
// decoded into `decoded`.
struct PdfBody {
  drogon::HttpRequestPtr req;
  std::string decoded;
  bool use_decoded = false;
  [[nodiscard]] std::span<const uint8_t> bytes() const;
};

struct ParseRequest {
  PdfBody pdf;
  ParseOptions opt;
  bool markdown_only = false;  // /parse?format=markdown
  bool with_markdown = true;   // per-page "markdown" field in JSON
};

// Decodes and validates a /parse or /parse/stream request. On a client error,
// answers 4xx through `cb` and returns false.
bool read_parse_request(const drogon::HttpRequestPtr &req, const ServerContext &ctx, bool stream, ParseRequest &out,
                        const Callback &cb);

// Exception -> HTTP status + stable error code.
struct ErrorInfo {
  drogon::HttpStatusCode status;
  const char *code;
  std::string message;
};
ErrorInfo classify(std::exception_ptr e, const char *route);
// 503 SERVER_BUSY for a full job queue (WorkPool's own message is generic).
ErrorInfo queue_full(const ServerContext &ctx);

drogon::HttpResponsePtr error_response(const ErrorInfo &e);
drogon::HttpResponsePtr json_response(const nlohmann::json &j);
// dump() that replaces invalid UTF-8 instead of throwing (PDF text layers can
// carry it).
std::string to_json_string(const nlohmann::json &j);

// Serialized writer over Drogon's async response stream: one JSON object per line, flushed as
// produced; send_line() turns false once the client is gone.
class NdjsonStream {
 public:
  void attach(drogon::ResponseStreamPtr s);  // from Drogon's stream callback
  // Waits for attach(). Drogon attaches as soon as it writes the response
  // headers; false means it never did (the client went away first).
  bool wait_attached(std::chrono::milliseconds timeout);
  bool send_line(const nlohmann::json &j);
  void finish();

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  drogon::ResponseStreamPtr stream_;
  bool attached_ = false, closed_ = false;
};

}  // namespace hp::server
