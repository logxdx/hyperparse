#pragma once
// hyperparse_server HTTP API. Every route answers errors as
// {"error": {"code": "...", "message": "..."}}.
//
// Parsing (queued; see job_queue.h)
//   POST /parse          PDF in (raw body, multipart "file"/"pdf", or JSON
//                        {"pdf": "<base64>"}). JSON out: {"pages": [...], "stats": {...}},
//                        pages in page order; ?format=markdown returns the
//                        document as text/markdown instead.
//   POST /parse/stream   Same input; application/x-ndjson out, one event per line,
//                        pages as they complete (out of order):
//                          {"event":"meta","pages":N,"queue_ms":..}
//                          {"event":"page", <page>} | {"event":"page_error","page":i,"error":..}
//                          {"event":"end","pages":N,"failed":k,"stats":{..}}
//                          {"event":"error","code":..,"message":..}   (after the 200)
//   Query (both): ocr=auto|all|none, dpi=36..600 (OCR render DPI), formulas=0|1,
//   markdown=0|1 (per-page "markdown" field, default 1); /parse also format=json|markdown.
//
// Models
//   GET  /models                       loaded?, config, load/unload counts + timings, queue
//   POST /models/load|warmup|unload    queued behind running/waiting parses; answer = GET /models
//   PUT  /models/idle_unload?seconds=N idle-unload timeout (0 = never); immediate
//
// Admin
//   GET /health                 200 JSON: status, models_loaded, queue, uptime_s
//   GET /health/live            200 "ok" while the process serves HTTP
//   GET /health/ready           200 unless shutting down. Models load on demand, so
//                               an unloaded engine is still ready.
//   GET /metrics                Prometheus text (metrics.h)
//   GET /, /docs                endpoint index (JSON) and full API docs (text): api_docs.h
//   anything else               404 NOT_FOUND

#include "context.h"

namespace hp::server {

void register_parse_routes(ServerContext &ctx);
void register_model_routes(ServerContext &ctx);
// Health, metrics, and the request-id / timing / metrics middleware.
void register_admin_routes(ServerContext &ctx);

}  // namespace hp::server
