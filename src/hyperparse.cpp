// hyperparse: command-line client of hp::Engine. Options: kUsage below.
//
// Models: --models DIR (or env HP_MODELS), else the Hugging Face cache plus
// engines built on first use (model_store.h); --prepare-models only does that.
//
// The engine is warmed up before the timed parse (unless --no-warmup), so the
// reported pages/s is steady-state throughput; --no-warmup shows the cold start.

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "model_store.h"
#include "page_json.h"

namespace {

constexpr const char *kUsage =
    "usage: hyperparse <in.pdf> [options]\n"
    "       hyperparse --prepare-models [--models DIR] [--no-formula] [--ocr none]\n"
    "\n"
    "Parses a PDF on the GPU into layout regions, text lines, formulas (LaTeX) and\n"
    "Markdown. Results go to --out and/or --md; throughput stats go to stderr.\n"
    "\n"
    "Output:\n"
    "  --out FILE             one JSON object per page (layout, lines, formulas, route)\n"
    "  --md FILE              Markdown of the whole document\n"
    "\n"
    "Parsing:\n"
    "  --ocr auto|all|none    auto: keep a usable text layer, OCR the rest (default)\n"
    "                         all: OCR every page; none: text layer + layout only\n"
    "  --dpi N                render DPI for pages that need OCR (150)\n"
    "  --det-side N           text-detection input, longest side in pixels (1280)\n"
    "  --no-formula           skip formula recognition\n"
    "  --rec budget|bucket    text-recognition batching (budget)\n"
    "\n"
    "Engine:\n"
    "  --workers N            page-rendering worker processes (0 = auto)\n"
    "  --no-warmup            skip the warm-up; stats then include the cold start\n"
    "\n"
    "Models:\n"
    "  --models DIR           TensorRT plans + dictionaries, or the ONNX to build them\n"
    "                         from (env HP_MODELS). Default: logxdx/hyperparse-models\n"
    "                         from the Hugging Face cache (downloaded if missing), with\n"
    "                         engines built on first use into ~/.cache/hyperparse\n"
    "  --prepare-models       download/build the models, print their folder, exit\n"
    "\n"
    "  -h, --help             show this help\n";

int to_int(const std::string &flag, const std::string &v) {
  char *end = nullptr;
  const long x = std::strtol(v.c_str(), &end, 10);
  if (v.empty() || *end != '\0' || x < 0 || x > 1'000'000) throw std::runtime_error(flag + ": expected a number >= 0, got '" + v + "'");
  return static_cast<int>(x);
}

struct Args {
  std::string pdf, out, md;
  hp::EngineConfig engine;
  hp::ParseOptions parse;
  bool warmup = true;
  bool prepare_only = false;
};

Args parse_args(int argc, char **argv) {
  Args a;
  const char *env_models = std::getenv("HP_MODELS");  // as hyperparse_server; --models wins; empty = HF cache
  if (env_models) a.engine.models.dir = env_models;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "-h" || k == "--help") {
      std::fputs(kUsage, stdout);
      std::exit(0);
    } else if (k == "--out") a.out = next();
    else if (k == "--md") a.md = next();
    else if (k == "--dpi") a.parse.ocr_dpi = to_int(k, next());
    else if (k == "--det-side") a.engine.models.det_max_side = to_int(k, next());
    else if (k == "--workers") a.engine.workers = to_int(k, next());
    else if (k == "--models") a.engine.models.dir = next();
    else if (k == "--no-formula") a.engine.models.formula = a.parse.formulas = false;
    else if (k == "--no-warmup") a.warmup = false;
    else if (k == "--prepare-models") a.prepare_only = true;
    else if (k == "--rec") {
      const std::string v = next();
      if (v != "budget" && v != "bucket") throw std::runtime_error("--rec budget|bucket");
      a.engine.models.rec = v == "bucket" ? hp::RecMode::kWidthBucket : hp::RecMode::kColumnBudget;
    } else if (k == "--ocr") {
      const std::string v = next();
      if (v == "auto") a.parse.ocr = hp::OcrMode::kAuto;
      else if (v == "all") a.parse.ocr = hp::OcrMode::kAll;
      else if (v == "none") a.parse.ocr = hp::OcrMode::kNone;
      else throw std::runtime_error("--ocr auto|all|none");
    } else if (k.size() > 1 && k[0] == '-') throw std::runtime_error("unknown option " + k + " (see hyperparse --help)");
    else if (a.pdf.empty()) a.pdf = k;
    else throw std::runtime_error("more than one input PDF: " + a.pdf + ", " + k + " (see hyperparse --help)");
  }
  if (a.pdf.empty() && !a.prepare_only)
    throw std::runtime_error(std::string("no input PDF\n\n") + kUsage);
  a.engine.models.ocr = a.parse.ocr != hp::OcrMode::kNone;  // don't load det/rec that cannot run
  return a;
}

void print_stats(const hp::ParseStats &s) {
  const int n = std::max(1, s.pages - s.failed);
  const int ocr_pages = s.routes[hp::kPartial] + s.routes[hp::kFullOcr];
  auto per_ocr = [&](double v) { return ocr_pages ? v / ocr_pages : 0.0; };
  const double parse_ms = s.wall_ms - s.load_ms;
  std::fprintf(stderr,
               "%d pages in %.0f ms = %.1f pages/s  (%d failed)%s\n"
               "  routes: %d native, %d partial, %d full OCR | %d lines; OCR recognised %d of %d detected boxes\n"
               "  per page (in workers): text layer + routing %.1f ms, render %.1f ms\n"
               "  GPU thread: OCR %.1f ms per OCR page (det+filter %.1f, rec %.1f), host wait on layout %.1f ms/page, "
               "assembly %.2f ms/page\n"
               "  region-only det on %d partial pages (mean canvas %.2f MP); %d text-layer boxes recognised without det\n"
               "  formulas: %d recognised in %d batches, %.1f ms each on the assembly thread (%d decode steps)\n",
               s.pages, parse_ms, s.pages * 1000.0 / std::max(parse_ms, 1e-3), s.failed,
               s.load_ms > 0 ? (" + " + std::to_string(static_cast<int>(s.load_ms)) + " ms model load").c_str() : "",
               s.routes[0], s.routes[1], s.routes[2], s.lines, s.rec_boxes, s.det_boxes, s.text_ms / n, s.render_ms / n,
               per_ocr(s.ocr_ms), per_ocr(s.det_ms), per_ocr(s.rec_ms), s.layout_wait_ms / n, s.assemble_ms / n,
               s.region_det_pages, s.region_det_pages ? s.region_det_canvas_px / s.region_det_pages / 1e6 : 0.0,
               s.direct_rec_boxes, s.formulas, s.formula_batches, s.formulas ? s.formula_ms / s.formulas : 0.0,
               s.formula_steps);
}

}  // namespace

int main(int argc, char **argv) try {
  std::setlocale(LC_CTYPE, "C.UTF-8");
  Args a = parse_args(argc, argv);
  a.engine.models.dir = hp::prepare_models(a.engine.models.dir, {a.engine.models.ocr, a.engine.models.formula});
  if (a.prepare_only) {
    std::printf("%s\n", a.engine.models.dir.c_str());
    return 0;
  }
  std::ifstream f(a.pdf, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + a.pdf);
  const std::vector<uint8_t> pdf((std::istreambuf_iterator<char>(f)), {});

  hp::Engine engine(a.engine);  // forks the page workers first
  if (a.warmup) {
    engine.warmup();
    const auto life = engine.lifecycle();
    std::fprintf(stderr, "models loaded in %.0f ms, warm-up %.0f ms; %d page workers\n", life.last_load_ms,
                 life.last_warmup_ms, engine.config().workers);
  }

  std::ofstream out;
  if (!a.out.empty()) out.open(a.out);
  std::vector<std::string> md_pages;
  const hp::ParseStats stats = engine.parse(pdf, a.parse, [&](hp::PageResult &&p) {
    if (!p.ok) {
      std::fprintf(stderr, "page %d: %s\n", p.page + 1, p.error.c_str());
      return;
    }
    if (!a.md.empty()) {
      if (md_pages.size() <= static_cast<size_t>(p.page)) md_pages.resize(p.page + 1);
      md_pages[p.page] = std::move(p.markdown);
    }
    if (out.is_open()) out << hp::page_to_json(p, false).dump() << '\n';
  });
  if (!a.md.empty()) {
    std::ofstream md(a.md);
    for (int p = 0; p < stats.pages; ++p)
      md << "<!-- page " << p + 1 << " -->\n\n" << (p < static_cast<int>(md_pages.size()) ? md_pages[p] : "") << "\n\n";
  }
  print_stats(stats);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
