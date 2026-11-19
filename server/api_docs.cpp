#include "api_docs.h"

#include <cstdio>

namespace hp::server {

const std::vector<Endpoint> &endpoints() {
  static const std::vector<Endpoint> e = {
      {"POST", "/parse", "parse a PDF: JSON pages + stats, or Markdown with ?format=markdown"},
      {"POST", "/parse/stream", "parse a PDF: NDJSON events, each page as soon as it is done"},
      {"GET", "/models", "model state: loaded?, config, load/unload counts and timings, queue"},
      {"POST", "/models/load", "load the models (queued behind running parses)"},
      {"POST", "/models/warmup", "load and warm up the models (queued)"},
      {"POST", "/models/unload", "unload the models and free GPU memory (queued)"},
      {"PUT", "/models/idle_unload?seconds=N", "unload after N idle seconds (0 = never)"},
      {"GET", "/health", "status JSON: models loaded, queue, uptime (200 while serving)"},
      {"GET", "/health/live", "liveness probe: 200 \"ok\" while serving"},
      {"GET", "/health/ready", "readiness probe: 200, or 503 while shutting down"},
      {"GET", "/metrics", "Prometheus metrics"},
      {"GET", "/", "this endpoint list as JSON"},
      {"GET", "/docs", "full API docs (plain text)"},
  };
  return e;
}

namespace {

std::string table(const char *indent) {
  std::string out;
  char line[256];
  for (const Endpoint &e : endpoints()) {
    std::snprintf(line, sizeof line, "%s%-5s %-30s %s\n", indent, e.method, e.path, e.summary);
    out += line;
  }
  return out;
}

}  // namespace

std::string endpoints_help() { return "\nEndpoints (full docs: GET /docs):\n" + table("  "); }

nlohmann::json api_index() {
  nlohmann::json list = nlohmann::json::array();
  for (const Endpoint &e : endpoints()) list.push_back({{"method", e.method}, {"path", e.path}, {"summary", e.summary}});
  return {{"service", "hyperparse_server"}, {"docs", "/docs"}, {"endpoints", list}};
}

std::string api_docs(const ServerConfig &cfg, const std::string &base) {
  const std::string max_pages = cfg.max_pages > 0 ? std::to_string(cfg.max_pages) : "no limit";
  std::string d;
  d += "hyperparse_server API (" + base + ")\n\n";
  d += "Parses PDFs on the GPU into layout regions, text lines, formulas (LaTeX) and\n"
       "Markdown. One document is parsed at a time; up to " +
       std::to_string(cfg.max_queue) +
       " more wait in a queue. Beyond that,\n"
       "requests get 503 SERVER_BUSY with a Retry-After header.\n\n";
  d += "ENDPOINTS\n" + table("  ");

  d += "\nPARSING: POST /parse, POST /parse/stream\n";
  d += "  Body, one of:\n"
       "    raw PDF bytes     curl --data-binary @doc.pdf -H 'Content-Type: application/pdf' " + base + "/parse\n"
       "    multipart form    curl -F file=@doc.pdf " + base + "/parse          (field \"file\" or \"pdf\")\n"
       "    JSON              {\"pdf\": \"<base64>\"} with Content-Type: application/json\n";
  d += "  Limits: " + std::to_string(cfg.max_body_mb) + " MB per upload, " + max_pages + " pages per document.\n";
  d += "  Query parameters (all optional):\n"
       "    ocr=auto|all|none     auto: keep a usable text layer, OCR the rest (default)\n"
       "                          all: OCR every page; none: text layer + layout only\n"
       "    dpi=36..600           render DPI for pages that need OCR (150)\n"
       "    formulas=0|1          recognise formulas (1)\n"
       "    markdown=0|1          include each page's \"markdown\" field (1)\n"
       "    format=json|markdown  /parse only: markdown answers the whole document\n"
       "                          as text/markdown\n"
       "  Unknown or invalid parameters are rejected with 400.\n\n";
  d += "  POST /parse answers {\"pages\": [...], \"stats\": {...}}, pages in page order:\n"
       "    {\"page\": 0, \"w\": .., \"h\": .., \"route\": \"native|partial|full_ocr\",\n"
       "     \"layout\":   [{\"cls\": \"text\", \"score\": 0.97, \"bbox\": [x0, y0, x1, y1]}, ...],\n"
       "     \"lines\":    [{\"text\": \"...\", \"bbox\": [...], \"src\": \"pdf|ocr\"}, ...],\n"
       "     \"formulas\": [{\"layout\": <index into layout>, \"latex\": \"...\"}, ...],\n"
       "     \"tables\":   [{\"layout\": <index into layout>, \"markdown\": \"| a | b |...\"}, ...],\n"
       "     \"order\":    [reading order of layout indices], \"markdown\": \"...\", ...}\n"
       "    A page that failed is {\"page\": i, \"error\": \"...\"}; the others still parse.\n"
       "    Example: curl --data-binary @doc.pdf '" + base + "/parse?format=markdown' > doc.md\n\n";
  d += "  POST /parse/stream answers application/x-ndjson, one JSON event per line;\n"
       "  pages arrive as they finish (not in page order):\n"
       "    {\"event\": \"meta\", \"pages\": N, \"queue_ms\": ..}\n"
       "    {\"event\": \"page\", <page as above>}\n"
       "    {\"event\": \"page_error\", \"page\": i, \"error\": \"...\"}\n"
       "    {\"event\": \"end\", \"pages\": N, \"failed\": k, \"stats\": {...}}\n"
       "    {\"event\": \"error\", \"code\": \"...\", \"message\": \"...\"}  (a failure after the 200)\n"
       "    Example: curl -N --data-binary @doc.pdf " + base + "/parse/stream\n";

  d += "\nMODELS\n"
       "  The models stay loaded unless unloaded or idle-unloaded; a parse loads them\n"
       "  on demand. load/warmup/unload wait their turn in the parse queue and answer\n"
       "  with the same JSON as GET /models.\n"
       "    Example: curl -X PUT '" + base + "/models/idle_unload?seconds=300'\n";

  d += "\nHEALTH\n"
       "  GET /health        {\"status\": \"ok\"|\"shutting_down\", \"models_loaded\": .., \"queue\":\n"
       "                     {\"waiting\": .., \"running\": .., \"max_waiting\": ..}, \"uptime_s\": ..}\n"
       "  GET /health/live   200 \"ok\" while the process serves HTTP\n"
       "  GET /health/ready  200 \"ok\"; 503 NOT_READY while shutting down. Unloaded\n"
       "                     models still count as ready: they load on demand.\n";

  d += "\nERRORS\n"
       "  Every error is {\"error\": {\"code\": \"...\", \"message\": \"...\"}}:\n"
       "    400  INVALID_PDF, PDF_TOO_LARGE, INVALID_PARAMETER, INVALID_DPI, EMPTY_BODY,\n"
       "         MISSING_FILE, MISSING_PDF, BASE64_DECODE_FAILED, INVALID_MULTIPART, OCR_DISABLED\n"
       "    404  NOT_FOUND (no such endpoint)\n"
       "    503  SERVER_BUSY (queue full; retry after Retry-After), SHUTTING_DOWN\n"
       "    504  QUEUE_TIMEOUT (waited longer than --queue-timeout-ms)\n"
       "    500  INTERNAL_ERROR\n";

  d += "\nRESPONSE HEADERS\n"
       "  X-Request-Id          echoed from the request, or generated\n"
       "  X-Inference-Time-Ms   server time for the request\n"
       "  Retry-After           on 503\n";
  return d;
}

}  // namespace hp::server
