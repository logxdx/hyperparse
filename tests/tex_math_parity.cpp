// Math spans of PDF pages (hp::tex::page_math), one JSON line per span, for
// parity against LiteParse's tex_math (tests/tex_math_parity.py compares).
//   tex_math_parity [--all] [--selftest] <in.pdf>...
// --all prints every span page_math found (field "emit" says whether LiteParse
// would emit it); default prints only emitted spans. A last line per file gives
// the per-page CPU time of page_math (pdfium page/text-page loading excluded).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <fpdf_text.h>
#include <fpdfview.h>

#include "tex_math/detail.h"
#include "tex_math/tex_math.h"

namespace {

std::string json_str(const std::string &s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
    else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += (char)c;
  }
  return o + "\"";
}

int selftest() {
  using namespace hp::tex::detail;
  int bad = 0;
  auto expect = [&](bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); ++bad; }
  };
  expect(decode(Enc::Oms, 0) == "-", "oms 0");
  expect(decode(Enc::Oms, 48) == "'", "oms 48");
  expect(decode(Enc::Oms, 64) == R"(\aleph)", "oms 64");
  expect(!decode(Enc::Oms, 65), "oms 65");
  expect(decode(Enc::Oms, 91) == R"(\cup)", "oms 91");
  expect(decode(Enc::Oms, 112) == R"(\surd)", "oms 112");
  expect(decode(Enc::Oms, 127) == R"(\spadesuit)", "oms 127");
  expect(decode(Enc::Omx, 15) == R"(\backslash)", "omx 15");
  expect(decode(Enc::Omx, 53) == "", "omx 53");
  expect(decode(Enc::Omx, 99) == "ACC:widehat", "omx 99");
  expect(decode(Enc::Oml, 39) == R"(\varphi)", "oml 39");
  expect(decode(Enc::Oml, 126) == "ACC:vec", "oml 126");
  expect(is_tex_math_font("ABCDEF+CMMI10") && is_tex_math_font("MSBM10") && is_tex_math_font("CMEX10") &&
             !is_tex_math_font("CMR10") && !is_tex_math_font("Times-Roman"),
         "is_tex_math_font");
  auto glyph = [](const char *t, Role role, const char *fam, float size, float base, float x0, float x1) {
    G g;
    g.t = t;
    g.role = role;
    g.fam = fam;
    g.size = size;
    g.base = base;
    g.x0 = x0;
    g.x1 = x1;
    g.top = 800.0f - (base + size * 0.75f);
    g.bottom = 800.0f - (base - size * 0.25f);
    g.loose_top = g.top;
    g.loose_bottom = g.bottom;
    return g;
  };
  std::vector<G> all = {glyph("x", Role::Mi, "CMMI", 10, 100, 0, 5), glyph("i", Role::Mi, "CMMI", 7, 98, 5, 8),
                        glyph("2", Role::None, "CMR", 7, 104, 8, 11)};
  expect(build(all, {0, 1, 2}, 100, 10) == "x_{i}^{2}", "build scripts");
  std::vector<G> pr = {glyph("f", Role::Mi, "CMMI", 10, 100, 0, 5)};
  G prime = glyph("'", Role::Sy, "CMSY", 7, 104, 5, 7);
  prime.t = std::string(1, '\0') + "'";
  pr.push_back(prime);
  expect(build(pr, {0, 1}, 100, 10) == "f'", "primes");
  expect(!hp::tex::has_structure(R"(\pm )") && !hp::tex::has_structure(R"(\alpha )") &&
             hp::tex::has_structure("x_{i}") && hp::tex::has_structure(R"(\mathcal{O}(2^{M}\cdot K))"),
         "has_structure");
  std::printf("selftest: %s\n", bad ? "FAILED" : "ok");
  return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char **argv) {
  bool all = false;
  std::vector<const char *> files;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--all")) all = true;
    else if (!std::strcmp(argv[i], "--selftest")) return selftest();
    else files.push_back(argv[i]);
  }
  if (files.empty()) {
    std::fprintf(stderr, "usage: tex_math_parity [--all] [--selftest] <in.pdf>...\n");
    return 2;
  }
  FPDF_InitLibrary();
  for (const char *path : files) {
    std::ifstream f(path, std::ios::binary);
    const std::vector<char> pdf((std::istreambuf_iterator<char>(f)), {});
    FPDF_DOCUMENT doc = FPDF_LoadMemDocument(pdf.data(), (int)pdf.size(), nullptr);
    if (!doc) {
      std::fprintf(stderr, "cannot open %s\n", path);
      continue;
    }
    const int pages = FPDF_GetPageCount(doc);
    double ms_total = 0;
    for (int p = 0; p < pages; ++p) {
      FPDF_PAGE page = FPDF_LoadPage(doc, p);
      if (!page) continue;
      FPDF_TEXTPAGE tp = FPDFText_LoadPage(page);
      const auto t0 = std::chrono::steady_clock::now();
      const auto spans = hp::tex::page_math(page, tp, false, all);
      ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      for (const auto &s : spans) {
        std::printf("{\"file\":%s,\"page\":%d,\"tex\":%s,\"x0\":%.4f,\"x1\":%.4f,\"top\":%.4f,\"bottom\":%.4f,"
                    "\"size\":%.4f,\"emit\":%s,\"display\":%s,\"chars\":[",
                    json_str(path).c_str(), p + 1, json_str(s.tex).c_str(), s.x0, s.x1, s.top, s.bottom, s.size,
                    hp::tex::has_structure(s.tex) ? "true" : "false", s.display ? "true" : "false");
        for (size_t k = 0; k < s.chars.size(); ++k) std::printf("%s%d", k ? "," : "", s.chars[k]);
        std::printf("]}\n");
      }
      FPDFText_ClosePage(tp);
      FPDF_ClosePage(page);
    }
    std::printf("{\"file\":%s,\"pages\":%d,\"ms\":%.4f}\n", json_str(path).c_str(), pages, ms_total);
    FPDF_CloseDocument(doc);
  }
  FPDF_DestroyLibrary();
  return 0;
}
