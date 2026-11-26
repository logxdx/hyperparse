#include "page_json.h"

namespace hp {

const char *route_name(int k) { return k == kNative ? "native" : k == kPartial ? "partial" : "full_ocr"; }

nlohmann::json page_to_json(const PageResult &p, bool with_markdown) {
  if (!p.ok) return {{"page", p.page}, {"error", p.error}};
  nlohmann::json lay = nlohmann::json::array();
  for (const auto &b : p.layout)
    lay.push_back({{"cls", kDfineLabels[b.cls]}, {"pp", kDfineToPP[b.cls]}, {"score", b.score},
                   {"bbox", {b.x0, b.y0, b.x1, b.y1}}});
  nlohmann::json lines = nlohmann::json::array();
  for (const auto &l : p.lines)
    lines.push_back({{"bbox", {l.x0, l.y0, l.x1, l.y1}}, {"text", l.text}, {"src", l.ocr ? "ocr" : "pdf"}});
  nlohmann::json regions = nlohmann::json::array();
  for (const auto &g : p.ocr_regions) regions.push_back({g.x0, g.y0, g.x1, g.y1});
  nlohmann::json formulas = nlohmann::json::array();
  for (const auto &f : p.formulas) formulas.push_back({{"layout", f.layout_index}, {"latex", f.latex}});
  nlohmann::json j = {{"page", p.page},
                      {"w", p.width},
                      {"h", p.height},
                      {"dpi", p.dpi},
                      {"rotation", p.rotation},
                      {"route", route_name(p.route)},
                      {"reasons", reasons_str(p.route_reasons)},
                      {"ocr_regions", regions},
                      {"det_boxes", p.det_boxes},
                      {"region_det", p.region_det},
                      {"rec_boxes", p.rec_boxes},
                      {"layout", lay},
                      {"lines", lines},
                      {"formulas", formulas},
                      {"order", p.reading_order},
                      {"ms", {{"text", p.ms.text}, {"render", p.ms.render}, {"gpu", p.ms.gpu}}}};
  nlohmann::json tables = nlohmann::json::array();
  for (const auto &t : p.tables) {
    nlohmann::json tj = {{"layout", t.layout_index}};
    if (!t.html.empty()) tj["html"] = t.html;
    else tj["markdown"] = t.markdown;
    tables.push_back(std::move(tj));
  }
  j["tables"] = tables;
  if (with_markdown) j["markdown"] = p.markdown;
  return j;
}

nlohmann::json stats_to_json(const ParseStats &s) {
  return {{"pages", s.pages},
          {"failed", s.failed},
          {"routes", {{"native", s.routes[kNative]}, {"partial", s.routes[kPartial]}, {"full_ocr", s.routes[kFullOcr]}}},
          {"wall_ms", s.wall_ms},
          {"load_ms", s.load_ms},
          {"lines", s.lines},
          {"formulas", s.formulas},
          {"ms_sum",
           {{"text", s.text_ms},
            {"render", s.render_ms},
            {"layout_wait", s.layout_wait_ms},
            {"ocr", s.ocr_ms},
            {"assemble", s.assemble_ms},
            {"formula", s.formula_ms}}}};
}

}  // namespace hp
