#include "request.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string_view>

#include <drogon/MultiPart.h>
#include <drogon/utils/Utilities.h>

#include "core/common/errors.h"
#include "core/common/log/logger.h"
#include "core/server/http_responses.h"

namespace hp::server {
namespace {

// pdfium accepts a document whose "%PDF-" header starts within the first 1024
// bytes (leading junk is common in mail/web downloads), so checking the same
// window rejects non-PDFs on the I/O thread without rejecting what the engine
// would parse.
constexpr size_t kPdfHeaderWindow = 1024;

bool looks_like_pdf(std::span<const uint8_t> b) {
  const std::string_view head(reinterpret_cast<const char *>(b.data()), std::min(b.size(), kPdfHeaderWindow));
  return head.find("%PDF-") != std::string_view::npos;
}

// strtol-based int parse (std::atoi is UB on overflow): garbage or
// out-of-range gives `fallback`, which the
// caller's range check then rejects.
int query_int(const std::string &s, int fallback) {
  if (s.empty()) return fallback;
  errno = 0;
  char *end = nullptr;
  const long v = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX) return fallback;
  return static_cast<int>(v);
}

// Boolean query spelling: 1/true/yes/on and 0/false/no/off.
bool parse_bool(const std::string &v, bool &out) {
  if (v == "1" || v == "true" || v == "yes" || v == "on") return out = true, true;
  if (v == "0" || v == "false" || v == "no" || v == "off") return out = false, true;
  return false;
}

void reply_400(const Callback &cb, const char *code, const std::string &msg) {
  cb(hp::core::server::error_response(drogon::k400BadRequest, code, msg));
}

// Raw body, multipart ("file" or "pdf" field) or JSON {"pdf": "<base64>"}.
bool extract_pdf(const drogon::HttpRequestPtr &req, PdfBody &out, const Callback &cb) {
  out.req = req;
  const auto ct = req->contentType();
  if (ct == drogon::CT_MULTIPART_FORM_DATA) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) return reply_400(cb, "INVALID_MULTIPART", "Failed to parse multipart body"), false;
    for (const auto &f : parser.getFiles())
      if (f.getItemName() == "file" || f.getItemName() == "pdf") {
        out.decoded.assign(f.fileData(), f.fileLength());
        out.use_decoded = true;
        break;
      }
    if (!out.use_decoded)
      return reply_400(cb, "MISSING_FILE", "Multipart request must contain a 'file' or 'pdf' form field"), false;
  } else if (ct == drogon::CT_APPLICATION_JSON) {
    const auto json = req->getJsonObject();
    if (!json || !json->isMember("pdf") || !(*json)["pdf"].isString())
      return reply_400(cb, "MISSING_PDF", R"(JSON body must contain {"pdf": "<base64>"})"), false;
    out.decoded = drogon::utils::base64Decode((*json)["pdf"].asString());
    out.use_decoded = true;
    if (out.decoded.empty()) return reply_400(cb, "BASE64_DECODE_FAILED", "Failed to decode base64 PDF"), false;
  } else if (req->body().empty()) {
    return reply_400(cb, "EMPTY_BODY", "Empty body"), false;
  }
  if (!looks_like_pdf(out.bytes()))
    return reply_400(cb, "INVALID_PDF", "Body is not a PDF (no %PDF- header in the first 1024 bytes)"), false;
  return true;
}

}  // namespace

std::span<const uint8_t> PdfBody::bytes() const {
  if (use_decoded) return {reinterpret_cast<const uint8_t *>(decoded.data()), decoded.size()};
  const std::string_view b = req->body();
  return {reinterpret_cast<const uint8_t *>(b.data()), b.size()};
}

// Parameters from the URL's query string only. Drogon's getParameters() also
// parses an application/x-www-form-urlencoded body, which is what
// `curl --data-binary @doc.pdf` sends: the PDF bytes would become "parameters".
std::map<std::string, std::string> query_params(const drogon::HttpRequestPtr &req) {
  std::map<std::string, std::string> out;
  const std::string q(req->query());
  for (size_t pos = 0; pos <= q.size();) {
    const size_t end = std::min(q.find('&', pos), q.size());
    const std::string_view kv(q.data() + pos, end - pos);
    if (!kv.empty()) {
      const size_t eq = kv.find('=');
      const std::string_view k = kv.substr(0, eq), v = eq == std::string_view::npos ? "" : kv.substr(eq + 1);
      out[drogon::utils::urlDecode(k.data(), k.data() + k.size())] = drogon::utils::urlDecode(v.data(), v.data() + v.size());
    }
    pos = end + 1;
  }
  return out;
}

bool read_parse_request(const drogon::HttpRequestPtr &req, const ServerContext &ctx, bool stream, ParseRequest &out,
                        const Callback &cb) {
  // Unknown parameters are rejected, not ignored: a typo such as ?ocr_mode=all
  // would otherwise silently run with defaults.
  static const std::set<std::string> kParseParams = {"ocr", "dpi", "formulas", "markdown", "format"};
  static const std::set<std::string> kStreamParams = {"ocr", "dpi", "formulas", "markdown"};
  const auto &allowed = stream ? kStreamParams : kParseParams;
  const auto params = query_params(req);
  for (const auto &[k, v] : params)
    if (!allowed.count(k))
      return reply_400(cb, "INVALID_PARAMETER", "unknown parameter '" + k.substr(0, 64) + "'"), false;
  auto param = [&params](const char *k) -> const std::string & {
    static const std::string none;
    const auto it = params.find(k);
    return it == params.end() ? none : it->second;
  };

  if (const auto &v = param("ocr"); !v.empty()) {
    if (v == "auto") out.opt.ocr = OcrMode::kAuto;
    else if (v == "all") out.opt.ocr = OcrMode::kAll;
    else if (v == "none") out.opt.ocr = OcrMode::kNone;
    else return reply_400(cb, "INVALID_PARAMETER", "ocr must be auto, all or none"), false;
    if (out.opt.ocr == OcrMode::kAll && !ctx.cfg.engine.models.ocr)
      return reply_400(cb, "OCR_DISABLED", "ocr=all needs the OCR models (server started with --no-ocr)"), false;
  }
  if (const auto &v = param("dpi"); !v.empty()) {
    out.opt.ocr_dpi = query_int(v, -1);
    if (out.opt.ocr_dpi < 36 || out.opt.ocr_dpi > 600)
      return reply_400(cb, "INVALID_DPI", "dpi must be between 36 and 600"), false;
  }
  if (const auto &v = param("formulas"); !v.empty() && !parse_bool(v, out.opt.formulas))
    return reply_400(cb, "INVALID_PARAMETER", "formulas must be 0/1/true/false"), false;
  if (const auto &v = param("markdown"); !v.empty() && !parse_bool(v, out.with_markdown))
    return reply_400(cb, "INVALID_PARAMETER", "markdown must be 0/1/true/false"), false;
  if (const auto &v = param("format"); !v.empty()) {
    if (v != "json" && v != "markdown")
      return reply_400(cb, "INVALID_PARAMETER", "format must be json or markdown"), false;
    out.markdown_only = v == "markdown";
  }
  out.opt.max_pages = ctx.cfg.max_pages;
  return extract_pdf(req, out.pdf, cb);
}

ErrorInfo classify(std::exception_ptr e, const char *route) {
  try {
    std::rethrow_exception(e);
  } catch (const PageLimitError &x) {
    return {drogon::k400BadRequest, "PDF_TOO_LARGE", x.what()};
  } catch (const InvalidPdfError &x) {
    return {drogon::k400BadRequest, "INVALID_PDF", x.what()};
  } catch (const std::invalid_argument &x) {
    return {drogon::k400BadRequest, "INVALID_PARAMETER", x.what()};
  } catch (const ShuttingDown &x) {
    return {drogon::k503ServiceUnavailable, "SHUTTING_DOWN", x.what()};
  } catch (const ClientGone &x) {
    return {drogon::k400BadRequest, "CLIENT_GONE", x.what()};
  } catch (const hp::core::PoolExhaustedError &x) {
    return {drogon::k503ServiceUnavailable, "SERVER_BUSY", x.what()};
  } catch (const std::exception &x) {
    // Logged in full; the client gets a generic message (it can carry model
    // paths and CUDA details).
    HP_LOG_ERROR("Parse failed", "route", std::string_view(route), "error", std::string_view(x.what()));
  } catch (...) {
    HP_LOG_ERROR("Parse failed: unknown exception", "route", std::string_view(route));
  }
  return {drogon::k500InternalServerError, "INTERNAL_ERROR", "Internal error (see server log)"};
}

ErrorInfo queue_full(const ServerContext &ctx) {
  return {drogon::k503ServiceUnavailable, "SERVER_BUSY",
          "queue full: " + std::to_string(ctx.jobs.max_waiting()) + " requests already waiting for the engine; retry later"};
}

drogon::HttpResponsePtr error_response(const ErrorInfo &e) {
  return hp::core::server::error_response(e.status, e.code, e.message);
}

std::string to_json_string(const nlohmann::json &j) {
  return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

drogon::HttpResponsePtr json_response(const nlohmann::json &j) {
  return hp::core::server::json_response(to_json_string(j));
}

// ── NdjsonStream ────────────────────────────────────────────────────────────

void NdjsonStream::attach(drogon::ResponseStreamPtr s) {
  {
    std::lock_guard l(mu_);
    stream_ = std::move(s);
    attached_ = true;
  }
  cv_.notify_all();
}

bool NdjsonStream::wait_attached(std::chrono::milliseconds timeout) {
  std::unique_lock l(mu_);
  return cv_.wait_for(l, timeout, [&] { return attached_; });
}

bool NdjsonStream::send_line(const nlohmann::json &j) {
  std::string line = to_json_string(j);
  line += '\n';
  std::lock_guard l(mu_);
  if (closed_ || !stream_) return false;
  if (!stream_->send(line)) closed_ = true;
  return !closed_;
}

void NdjsonStream::finish() {
  std::lock_guard l(mu_);
  if (stream_ && !closed_) stream_->close();
  closed_ = true;
}

}  // namespace hp::server
