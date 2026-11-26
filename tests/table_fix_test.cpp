// Tests for the table clean-up (src/table_fix.{h,cpp}, a port of liteparse
// markdown_layout/table_fix.rs) and the geometry grid that uses it (src/table_grid.cpp).
//
//   table_fix_test          unit tests (the Rust tests of table_fix.rs, plus header band, line
//                           building, HTML writer and geometry-grid cases); exit 1 on failure
//   table_fix_test --cli    JSON lines on stdin -> JSON lines on stdout, one per request, running
//                           the real code on saved tables (tools/table_fix/rescore.py):
//     {"kind":"model","grids":[{"cells":[{"r0","c0","r1","c1","text","bx":[x0,y0,x1,y1]}],
//       "head_rows":[..],"n_rows","n_cols"}, ..],"spans":[[text,x,y,w,h], ..],"taken":[0|1, ..],
//       "band":true,"clean_up":true,"headless":true}
//        -> {"html": <the region's tables, as assemble_page writes them>, "absorbed": <spans>}
//     {"kind":"grid","spans":[[text,x0,y0,x1,y1], ..],"stack_header":true,"clean_up":true}
//        -> {"text": <table_markdown>}
//     {"kind":"cells","rows":[[text, ..], ..],"boxes":[[[x,y,w,h]|null, ..], ..],"bands":[[y0,y1], ..],
//       "h":<median span height>,"stack_header":true,"clean_up":true}
//        -> {"text": <write_grid>}   (a saved grid table's cells, when its spans are not available)
//   table_fix_test --page <pdf> <page>...   the page's text layer as the server reads it (CPU)

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pdf_render.h"
#include "table_fix.h"
#include "table_grid.h"
#include "tablenet/decode.h"
#include "tablenet/units.h"
#include "text_layer.h"

using namespace hp;
using nlohmann::json;

namespace {

int failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

template <class A, class B>
void check_eq(const A &got, const B &want, const char *what, int line) {
  if (!(got == want)) {
    std::cerr << "  FAIL line " << line << ": " << what << "\n    got:  " << json(got).dump()
              << "\n    want: " << json(want).dump() << "\n";
    ++failures;
  }
}
#define CHECK_EQ(got, want) check_eq((got), (want), #got, __LINE__)

tfix::SpanCell sc(const std::string &t) { return {t, 1, 1, std::nullopt}; }

std::vector<std::vector<std::string>> texts(const tfix::SpanTable &t) {
  std::vector<std::vector<std::string>> out;
  for (const auto &r : t.rows) {
    std::vector<std::string> row;
    for (const auto &c : r) row.push_back(c.text);
    out.push_back(row);
  }
  return out;
}

using Rows = std::vector<std::vector<std::string>>;

// ── the tests of table_fix.rs ───────────────────────────────────────────────

void currency_cells_move_and_empty_column_goes() {
  tfix::SpanTable b{{{sc("Item"), sc(""), sc("2024"), sc("Pct")},
                     {sc("Appliances"), sc("$"), sc("14,046"), sc("8.8 %")},
                     {sc("Bath"), sc(""), sc("6,485"), sc("4.1")}},
                    1};
  tfix::fix_merged(b);
  CHECK_EQ(texts(b), (Rows{{"Item", "2024", "Pct"}, {"Appliances", "$ 14,046", "8.8 %"}, {"Bath", "6,485", "4.1"}}));
}

void trailing_symbol_moves_right() {
  tfix::SpanTable b{{{sc("A"), sc("8.8 % $"), sc("13,917")}, {sc("B"), sc("4.3"), sc("6,630")}}, 0};
  tfix::fix_merged(b);
  CHECK_EQ(texts(b)[0], (std::vector<std::string>{"A", "8.8 %", "$ 13,917"}));
}

void wrapped_cell_row_folds() {
  tfix::SpanTable b{{{sc("Title"), sc("Page"), sc("Dated")},
                     {sc("Independent Distributors"), sc("CW-CGL"), sc("06/01/18")},
                     {sc(""), sc("FOOD.EXCLPRODUCTS"), sc("")},
                     {sc("Independent Distributors"), sc("CW-CGL"), sc("06/01/18")},
                     {sc("Other"), sc("CW-X"), sc("06/01/18")}},
                    1};
  tfix::fix_merged(b);
  const Rows t = texts(b);
  CHECK_EQ(t.size(), size_t{4});
  CHECK_EQ(t[1][1], std::string("CW-CGL FOOD.EXCLPRODUCTS"));
}

void numeric_sub_rows_do_not_fold() {
  tfix::SpanTable b{{{sc("Year"), sc("Q"), sc("Value")},
                     {sc("2022"), sc("Q1"), sc("5")},
                     {sc(""), sc("Q2"), sc("6")},
                     {sc("2023"), sc("Q1"), sc("7")}},
                    1};
  tfix::fix_merged(b);
  CHECK_EQ(b.rows.size(), size_t{4});
}

void leaders_stripped() {
  CHECK_EQ(tfix::strip_leader_runs("2022 ......................"), std::string("2022"));
  CHECK_EQ(tfix::strip_leader_runs("Total . . . . . ."), std::string("Total"));
  CHECK_EQ(tfix::strip_leader_runs("v1.2.3"), std::string("v1.2.3"));
}

void group_header_spans_when_centred() {
  auto r = [](float x, float w) { return std::optional<tfix::Rect>(tfix::Rect{x, 0, w, 10}); };
  auto cell = [](const std::string &t, std::optional<tfix::Rect> b) { return tfix::SpanCell{t, 1, 1, b}; };
  tfix::SpanTable b{{{cell("", std::nullopt), cell("Fiscal 2024", r(120, 60)), cell("", std::nullopt)},
                     {cell("Item", r(0, 50)), cell("Net Sales", r(100, 40)), cell("% of Net Sales", r(150, 50))},
                     {cell("Bath", std::nullopt), cell("6,485", std::nullopt), cell("4.1", std::nullopt)}},
                    2};
  tfix::fix_merged(b);
  CHECK_EQ(b.rows[0].size(), size_t{2});
  CHECK_EQ(b.rows[0][1].colspan, 2);
}

// ── more cases ──────────────────────────────────────────────────────────────

// A plain table stays plain, and its header row stays the first row.
void plain_table_keeps_header() {
  tfix::SpanTable b{{{sc("Item"), sc(""), sc("2024")}, {sc("Appliances"), sc("$"), sc("14,046")}, {sc("Bath"), sc(""), sc("6,485")}},
                    1};
  tfix::fix_plain(b);
  CHECK_EQ(texts(b), (Rows{{"Item", "2024"}, {"Appliances", "$ 14,046"}, {"Bath", "6,485"}}));
  CHECK_EQ(b.header_rows, 1);
  CHECK(tfix::is_plain(b));
}

tn::GridCell gc(int r0, int c0, int r1, int c1, const std::string &t, std::array<double, 4> bx) {
  tn::GridCell c;
  c.r0 = r0, c.c0 = c0, c.r1 = r1, c.c1 = c1, c.text = t, c.bx = bx;
  return c;
}

// span_rows + to_html write a grid exactly as tn::Grid::to_html when its header rows lead.
void html_matches_grid_writer() {
  tn::Grid g;
  g.n_rows = 3, g.n_cols = 3;
  g.head_rows = {0};
  g.cells = {gc(0, 0, 0, 0, "Name", {0, 0, 40, 10}), gc(0, 1, 0, 2, "Q1 & \"Q2\"", {50, 0, 140, 10}),
             gc(1, 0, 2, 0, "A<b>", {0, 12, 40, 32}), gc(1, 1, 1, 1, "1", {50, 12, 60, 22}),
             gc(2, 2, 2, 2, "4", {130, 24, 140, 34})};
  CHECK_EQ(tfix::to_html(tfix::span_rows(g)), g.to_html());
  const tfix::SpanTable t = tfix::span_rows(g);
  CHECK_EQ(t.header_rows, 1);
  CHECK_EQ(t.rows[2].size(), size_t{2});  // (2,0) is covered by the rowspan, (2,1) is an empty slot
  // a head row below the top (a sub-header) stays <th>, also through the clean-up
  g.head_rows = {0, 2};
  CHECK_EQ(tfix::to_html(tfix::span_rows(g)), g.to_html());
  tfix::SpanTable f = tfix::span_rows(g);
  tfix::fix_merged(f);
  CHECK_EQ(tfix::to_html(f), g.to_html());
}

// A 3-column table whose top row has an empty first slot; a two-phrase line just above it
// becomes the header row.
tn::Grid band_grid() {
  tn::Grid g;
  g.n_rows = 3, g.n_cols = 3;
  g.cells = {gc(0, 1, 0, 1, "2024", {200, 100, 230, 110}), gc(0, 2, 0, 2, "2023", {300, 100, 330, 110}),
             gc(1, 0, 1, 0, "Sales", {50, 112, 90, 122}),  gc(1, 1, 1, 1, "1,200", {200, 112, 230, 122}),
             gc(1, 2, 1, 2, "1,100", {300, 112, 330, 122}), gc(2, 0, 2, 0, "Costs", {50, 124, 90, 134}),
             gc(2, 1, 2, 1, "800", {205, 124, 230, 134}),  gc(2, 2, 2, 2, "700", {305, 124, 330, 134})};
  return g;
}

void header_band_joins() {
  tn::Grid g = band_grid();
  const std::vector<tfix::BandSpan> spans = {{"Segment", 50, 86, 45, 10}, {"Fiscal", 195, 86, 22, 10},
                                             {"year", 219, 86, 16, 10},    {"Prior", 300, 86, 30, 10},
                                             {"Results", 50, 40, 40, 10}};  // too far above
  std::vector<char> taken(spans.size(), 0);
  const std::string html = tfix::model_table_html(g, spans, taken);
  CHECK_EQ(html, std::string("<table>\n<tr><th>Segment</th><th>Fiscal year</th><th>Prior</th></tr>\n"
                             "<tr><td></td><td>2024</td><td>2023</td></tr>\n<tr><td>Sales</td><td>1,200</td><td>1,100</td></tr>\n"
                             "<tr><td>Costs</td><td>800</td><td>700</td></tr>\n</table>"));
  CHECK_EQ(taken, (std::vector<char>{1, 1, 1, 1, 0}));
  // the same band with the spans already taken (another table's), or with every slot of the top
  // row filled, stays out
  tn::Grid g2 = band_grid();
  std::vector<char> all(spans.size(), 1);
  CHECK(tfix::model_table_html(g2, spans, all).find("Segment") == std::string::npos);
  tn::Grid g3 = band_grid();
  g3.cells.push_back(gc(0, 0, 0, 0, "Item", {50, 100, 70, 110}));
  g3.head_rows = {0};
  std::vector<char> none(spans.size(), 0);
  CHECK(tfix::model_table_html(g3, spans, none).find("Segment") == std::string::npos);
}

// hp: a grid without any header row takes the band over a full top row (the header row LayoutNet
// left out); with a header row, or with the switch off, it does not (liteparse's rule).
void header_band_headless() {
  const std::vector<tfix::BandSpan> spans = {{"Segment", 50, 86, 45, 10}, {"Fiscal year", 195, 86, 40, 10},
                                             {"Prior", 300, 86, 30, 10}};
  tn::Grid g = band_grid();
  g.cells.push_back(gc(0, 0, 0, 0, "Item", {50, 100, 70, 110}));
  std::vector<char> taken(spans.size(), 0);
  const std::string want = "<table>\n<tr><th>Segment</th><th>Fiscal year</th><th>Prior</th></tr>\n<tr><td>Item";
  CHECK_EQ(tfix::model_table_html(g, spans, taken).substr(0, want.size()), want);
  std::vector<char> t2(spans.size(), 0);
  CHECK(tfix::model_table_html(g, spans, t2, {.band = true, .clean_up = true, .headless = false}).find("Segment") ==
        std::string::npos);
  // a data row left above the table is no header
  const std::vector<tfix::BandSpan> row = {{"Rent", 50, 86, 30, 10}, {"1,300", 200, 86, 30, 10}, {"1,250", 300, 86, 30, 10}};
  tn::Grid g2 = band_grid();
  g2.cells.push_back(gc(0, 0, 0, 0, "Item", {50, 100, 70, 110}));
  std::vector<char> t3(row.size(), 0);
  CHECK(tfix::model_table_html(g2, row, t3).find("Rent") == std::string::npos);
}

// Prose above a table is not a header band.
void prose_is_no_band() {
  tn::Grid g = band_grid();
  const std::vector<tfix::BandSpan> spans = {
      {"The table below shows the results for the segment over the year and the year before.", 50, 86, 200, 10},
      {"Prior", 300, 86, 30, 10}};
  std::vector<char> taken(spans.size(), 0);
  CHECK(tfix::model_table_html(g, spans, taken).find("Prior") == std::string::npos);
}

void band_lines_join_phrases() {
  const std::vector<tfix::BandSpan> spans = {{"meeting", 64, 1, 30, 10}, {"CVMP", 40, 0, 22, 10},
                                             {"Product", 0, 0, 30, 10},  {"date", 96, 0, 18, 10},
                                             {"next", 0, 30, 20, 10}};
  const std::vector<tfix::BandLine> l = tfix::band_lines(spans);
  CHECK_EQ(l.size(), size_t{2});
  CHECK_EQ(l[0].spans.size(), size_t{2});
  CHECK_EQ(l[0].spans[0].text, std::string("Product"));
  CHECK_EQ(l[0].spans[1].text, std::string("CVMP meeting date"));
  CHECK_EQ(l[0].text, std::string("Product CVMP meeting date"));
  CHECK_EQ(l[1].members, (std::vector<int>{4}));
}

PageLine pl(const std::string &t, float x0, float y0, float x1, float y1) { return PageLine{t, x0, y0, x1, y1}; }

// A wrapped header written one line per row (SERFF rate tables): one header row in the pipe table.
void grid_header_lines_stack() {
  std::vector<PageLine> s = {
      pl("Advantage Co-", 300, 0, 360, 10),  // line 1
      pl("Choice High", 100, 6, 150, 16),    pl("Choice Low", 200, 6, 250, 16), pl("Advantage", 400, 6, 450, 16),
      pl("Age Range", 0, 12, 50, 22),        pl("Pay", 320, 12, 340, 22),
  };
  const char *ages[] = {"0-20", "21-24", "25-29"};
  for (int r = 0; r < 3; ++r) {
    const float y = 30 + 12.f * r;
    s.push_back(pl(ages[r], 0, y, 30, y + 10));
    for (int c = 1; c < 5; ++c) s.push_back(pl(std::to_string(10 * r + c) + ".50", 100.f * c, y, 100.f * c + 30, y + 10));
  }
  const std::string md = table_markdown(s);
  CHECK_EQ(md.substr(0, md.find('\n')), std::string("| Age Range | Choice High | Choice Low | Advantage Co-Pay | Advantage |"));
  CHECK_EQ(std::count(md.begin(), md.end(), '\n'), 4L);  // header, rule, 3 rows
  // without stacking: the old one-line-per-row header
  const std::string old = table_markdown(s, {.stack_header = false, .clean_up = false});
  CHECK_EQ(std::count(old.begin(), old.end(), '\n'), 6L);
  // a first data row without a number (it repeats a body cell in its column) is no header line
  std::vector<PageLine> k = {pl("Title of each class", 0, 0, 90, 10), pl("Exchange", 150, 0, 190, 10),
                             pl("Common Stock", 0, 12, 60, 22),        pl("New York", 150, 12, 190, 22)};
  const char *notes[] = {"1.875% Notes 2026", "0.750% Notes 2026", "1.125% Notes 2027"};
  for (int r = 0; r < 3; ++r)
    k.push_back(pl(notes[r], 0, 24.f + 12 * r, 80, 34.f + 12 * r)), k.push_back(pl("New York", 150, 24.f + 12 * r, 190, 34.f + 12 * r));
  const std::string km = table_markdown(k);
  CHECK_EQ(km.substr(0, km.find('\n')), std::string("| Title of each class | Exchange |"));
}

// A text-only table (no value row) keeps its rows; a currency column is folded into its numbers.
void grid_text_table_and_currency() {
  std::vector<PageLine> s = {pl("Name", 0, 0, 30, 10),   pl("Role", 100, 0, 130, 10),  pl("Alice", 0, 12, 30, 22),
                             pl("Chair", 100, 12, 130, 22), pl("Bob", 0, 24, 20, 34), pl("Member", 100, 24, 140, 34)};
  const std::string md = table_markdown(s);
  CHECK_EQ(std::count(md.begin(), md.end(), '\n'), 3L);
  std::vector<PageLine> c = {pl("Item", 0, 0, 30, 10),       pl("2024", 140, 0, 170, 10),     pl("Appliances", 0, 20, 60, 30),
                             pl("$", 110, 20, 116, 30),       pl("14,046", 140, 20, 170, 30), pl("Bath", 0, 40, 30, 50),
                             pl("$", 110, 40, 116, 50),       pl("6,485", 145, 40, 170, 50)};
  CHECK_EQ(table_markdown(c), std::string("| Item | 2024 |\n| --- | --- |\n| Appliances | $ 14,046 |\n| Bath | $ 6,485 |"));
}

int run_tests() {
  struct T {
    const char *name;
    void (*fn)();
  };
  const T tests[] = {{"currency_cells_move_and_empty_column_goes", currency_cells_move_and_empty_column_goes},
                     {"trailing_symbol_moves_right", trailing_symbol_moves_right},
                     {"wrapped_cell_row_folds", wrapped_cell_row_folds},
                     {"numeric_sub_rows_do_not_fold", numeric_sub_rows_do_not_fold},
                     {"leaders_stripped", leaders_stripped},
                     {"group_header_spans_when_centred", group_header_spans_when_centred},
                     {"plain_table_keeps_header", plain_table_keeps_header},
                     {"html_matches_grid_writer", html_matches_grid_writer},
                     {"header_band_joins", header_band_joins},
                     {"header_band_headless", header_band_headless},
                     {"prose_is_no_band", prose_is_no_band},
                     {"band_lines_join_phrases", band_lines_join_phrases},
                     {"grid_header_lines_stack", grid_header_lines_stack},
                     {"grid_text_table_and_currency", grid_text_table_and_currency}};
  int failed = 0;
  for (const T &t : tests) {
    const int before = failures;
    t.fn();
    const bool ok = failures == before;
    failed += !ok;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", t.name);
  }
  std::printf("%zu tests, %d failed\n", std::size(tests), failed);
  return failed ? 1 : 0;
}

// ── --cli ───────────────────────────────────────────────────────────────────

tn::Grid grid_of(const json &j) {
  tn::Grid g;
  for (const json &c : j.at("cells")) {
    tn::GridCell x;
    x.r0 = c.at("r0"), x.c0 = c.at("c0"), x.r1 = c.at("r1"), x.c1 = c.at("c1");
    x.text = c.at("text").get<std::string>();
    for (int k = 0; k < 4; ++k) x.bx[k] = c.at("bx")[k].get<double>();
    g.cells.push_back(std::move(x));
  }
  g.head_rows = j.at("head_rows").get<std::vector<int>>();
  g.n_rows = j.at("n_rows"), g.n_cols = j.at("n_cols");
  return g;
}

int run_cli() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    const json req = json::parse(line);
    json out;
    if (req.at("kind") == "model") {
      std::vector<tfix::BandSpan> spans;
      for (const json &s : req.at("spans"))
        spans.push_back({s[0].get<std::string>(), s[1].get<float>(), s[2].get<float>(), s[3].get<float>(), s[4].get<float>()});
      std::vector<char> taken;
      for (const json &t : req.at("taken")) taken.push_back(static_cast<char>(t.get<int>()));
      const tfix::FixOptions opt{req.value("band", true), req.value("clean_up", true), req.value("headless", true)};
      std::string html;
      for (const json &g : req.at("grids")) html += (html.empty() ? "" : "\n") + tfix::model_table_html(grid_of(g), spans, taken, opt);
      int absorbed = 0;
      for (size_t i = 0; i < taken.size(); ++i) absorbed += taken[i] && !req.at("taken")[i].get<int>();
      out = {{"html", html}, {"absorbed", absorbed}};
    } else if (req.at("kind") == "cells") {
      std::vector<std::vector<std::string>> text;
      std::vector<std::vector<std::optional<tfix::Rect>>> box;
      std::vector<std::pair<float, float>> band;
      for (const json &r : req.at("rows")) text.push_back(r.get<std::vector<std::string>>());
      for (const json &r : req.at("boxes")) {
        std::vector<std::optional<tfix::Rect>> row;
        for (const json &b : r)
          row.push_back(b.is_null() ? std::nullopt
                                    : std::optional(tfix::Rect{b[0].get<float>(), b[1].get<float>(), b[2].get<float>(),
                                                               b[3].get<float>()}));
        box.push_back(std::move(row));
      }
      for (const json &b : req.at("bands")) band.push_back({b[0].get<float>(), b[1].get<float>()});
      const GridOptions opt{req.value("stack_header", true), req.value("clean_up", true)};
      out = {{"text", write_grid(std::move(text), std::move(box), band, req.at("h").get<float>(), opt)}};
    } else {
      std::vector<PageLine> spans;
      for (const json &s : req.at("spans"))
        spans.push_back(pl(s[0].get<std::string>(), s[1].get<float>(), s[2].get<float>(), s[3].get<float>(), s[4].get<float>()));
      const GridOptions opt{req.value("stack_header", true), req.value("clean_up", true)};
      out = {{"text", table_markdown(spans, opt)}};
    }
    std::cout << out.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
  }
  std::cout.flush();
  return 0;
}

// --page <pdf> <page index>...: the text layer as the server reads it (CPU only), one JSON line
// per page: size in points, route, the native items it keeps ([text, x0, y0, x1, y1], points)
// and LayoutNet's native units' words ([text, x0, y0, x1, y1], points).
int run_page_dump(int argc, char **argv) {
  std::ifstream f(argv[2], std::ios::binary);
  const std::vector<char> pdf((std::istreambuf_iterator<char>(f)), {});
  PdfRaster raster(reinterpret_cast<const uint8_t *>(pdf.data()), pdf.size());
  if (!raster.ok()) return 1;
  for (int a = 3; a < argc; ++a) {
    const int page = std::atoi(argv[a]);
    const tn::PageUnits units = tn::native_units_doc(raster.doc(), page);  // before extraction, as the worker
    PageText pt = extract_page_text(raster.doc(), page);
    const Route rt = route_page(pt);
    json items = json::array(), words = json::array();
    for (const TextItem &it : pt.items)
      if (!it.dropped) items.push_back({it.text, it.box.x0, it.box.y0, it.box.x1, it.box.y1});
    for (const tn::Unit &u : units.units) {
      if (u.q != 0) continue;  // rotated text: reading frame != page frame
      for (const tn::Word &w : u.words) words.push_back({w.text, w.x0, u.pbox[1], w.x1, u.pbox[3]});
    }
    std::cout << json{{"page", page}, {"w_pt", pt.w_pt}, {"h_pt", pt.h_pt}, {"route", rt.kind}, {"items", items},
                      {"words", words}}
                     .dump(-1, ' ', false, json::error_handler_t::replace)
              << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--cli") return run_cli();
  if (argc > 3 && std::string(argv[1]) == "--page") return run_page_dump(argc, argv);
  return run_tests();
}
