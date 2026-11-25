#pragma once
// Public request/result types of the engine API (engine.h).

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "assemble.h"
#include "dfine_layout.h"
#include "text_layer.h"

namespace hp {

enum class OcrMode {
  kAuto,  // route per page/region: keep a usable text layer, OCR the rest
  kAll,   // OCR every page (ignore the text layer)
  kNone,  // never OCR: text layer + layout only
};

// Per-document options; models are chosen per engine (ModelConfig).
struct ParseOptions {
  int ocr_dpi = 150;  // render DPI for pages that need OCR (capped at a 2200 px long side)
  OcrMode ocr = OcrMode::kAuto;
  bool formulas = true;  // recognise display formulas (if the engine loaded the model)
  // Table structure from LayoutNet in D-FINE Table regions (if the engine loaded it); otherwise,
  // or where the model finds no table, cells come from span geometry (table_grid.h).
  bool tables = true;
  // With tables, text-only pages render at least this DPI: LayoutNet's ruling lines and image
  // branch were trained on 150 DPI renders (text-only pages otherwise render just large enough
  // for the 960 px layout canvas, ~80 DPI for A4). 0 keeps the small render.
  int tables_dpi = 150;
  // Reading order and Markdown blocks from OrderNet (if the engine loaded it); otherwise XY-cut over
  // the layout regions and hp's region renderer.
  bool ordernet = true;
  // Reject documents with more pages than this (PageLimitError) before any page
  // is rendered. 0 = no limit.
  int max_pages = 0;
  // Called once with the page count after the PDF is opened and before any page
  // is rendered, on the thread that called Engine::parse. Throwing aborts the
  // parse (the exception is rethrown from Engine::parse).
  std::function<void(int pages)> on_start;
};

// Engine::parse errors about the document itself (both are invalid_argument,
// like the option errors, so callers that only care about "bad input" can catch
// that).
struct InvalidPdfError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};
struct PageLimitError : std::invalid_argument {
  PageLimitError(int pages_, int limit_)
      : std::invalid_argument("PDF has " + std::to_string(pages_) + " pages; the limit is " + std::to_string(limit_)),
        pages(pages_),
        limit(limit_) {}
  int pages, limit;
};

struct PageResult {
  int page = 0;  // 0-based
  bool ok = false;
  std::string error;  // set when !ok: render failure, worker crash or page timeout

  int width = 0, height = 0;  // rendered page px; every coordinate below is in these
  float dpi = 0;
  int route = kNative;        // RouteKind
  unsigned route_reasons = 0; // Reason bits; reasons_str() for text
  std::vector<RectF> ocr_regions;  // partial pages: regions sent to det (page px)

  std::vector<LayoutBox> layout;     // D-FINE boxes, score >= 0.5
  std::vector<PageLine> lines;       // native + OCR lines after merging
  std::vector<PageFormula> formulas; // LaTeX per display-formula layout box
  std::vector<int> reading_order;    // indices into `lines`
  std::vector<PageTable> tables;    // grids found in Table layout regions
  std::string markdown;

  int det_boxes = 0, rec_boxes = 0;
  bool region_det = false;  // partial page: det ran on packed regions only
  struct {
    double text = 0, render = 0, gpu = 0;  // ms: text layer + routing, raster, GPU stages
  } ms;
};

// Called once per page as pages complete: in completion order (not page order),
// never concurrently, from the engine's assembly thread. An exception thrown by
// the sink aborts the parse and is rethrown from Engine::parse.
using PageSink = std::function<void(PageResult &&)>;

struct ParseStats {
  int pages = 0, failed = 0;
  int routes[3] = {0, 0, 0};  // native, partial, full OCR
  double wall_ms = 0;          // whole parse, including any model load
  double load_ms = 0;          // > 0 when this parse had to (re)load the models

  int lines = 0, det_boxes = 0, rec_boxes = 0, direct_rec_boxes = 0;
  int region_det_pages = 0;
  double region_det_canvas_px = 0;

  // Sums over pages (divide by pages or OCR pages for means).
  double text_ms = 0, render_ms = 0;          // in the page workers
  double layout_wait_ms = 0;                   // GPU thread waiting on a lane
  double ocr_ms = 0, det_ms = 0, rec_ms = 0;   // GPU thread, OCR pages only
  double assemble_ms = 0;

  int formulas = 0, formula_batches = 0, formula_steps = 0;
  double formula_ms = 0;

  int table_pages = 0, model_tables = 0;  // pages LayoutNet ran on, tables it returned
  double table_ms = 0;                    // LayoutNet on the assembly thread (features, net, decode)
  int order_pages = 0;                     // pages ordered by OrderNet (the rest: XY-cut)
  double order_ms = 0;                     // OrderNet on the assembly thread (features, net, decode, Markdown)
};

}  // namespace hp
