#pragma once
// JSON form of engine results, shared by hyperparse (JSON lines) and hyperparse_server
// (responses and stream events), so the two can't drift apart.

#include <nlohmann/json.hpp>

#include "parse_types.h"

namespace hp {

// One page: geometry, route, layout, lines, formulas, reading order, timings.
// `with_markdown` adds the page's Markdown as "markdown". Failed pages
// (!p.ok) give {"page", "error"} only.
nlohmann::json page_to_json(const PageResult &p, bool with_markdown);

// Document totals: pages, failures, routes and the timing sums.
nlohmann::json stats_to_json(const ParseStats &s);

const char *route_name(int route_kind);  // "native" | "partial" | "full_ocr"

}  // namespace hp
