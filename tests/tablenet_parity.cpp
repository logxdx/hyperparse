// Stage-by-stage parity of hp's LayoutNet port against liteparse's table model, from per-page
// dumps written by a reference build (LP_TN_DUMP=<dir>, projection signals off: LP_TN_NOPROJ=1).
//
//   tablenet_parity <doc.pdf> <dump_dir> <tablenet.plan> [max_pages]
//
// Per dumped page (p####.json, 1-based page; p####_img.u8 the image-branch page):
//   units     native_units() on the PDF page vs the dump's units (text-layer pages only)
//   rules     detect_rules() on a 150 dpi render (reference flags) vs the dump's ruling lines
//   image     page_image() of that render vs the dumped image-branch pixels
//   feats     unit_features() / pair_rule_features() from the dump's units and rules
//   logits    TensorRT on the dump's own inputs vs the dump's (ONNX Runtime) logits
//   tables    decode of the TRT logits vs the dump's tables (HTML)
//   e2e       run_page() from hp's own units and render vs the dump's tables

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <fpdfview.h>
#include <nlohmann/json.hpp>

#include "tablenet/decode.h"
#include "tablenet/features.h"
#include "tablenet/page.h"
#include "tablenet/tablenet.h"
#include "tablenet/units.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace hp::tn;

namespace {

std::vector<Unit> units_from(const json &j) {
  std::vector<Unit> us;
  for (const json &u : j) {
    Unit x;
    for (int k = 0; k < 4; ++k) x.bx[k] = u["bx"][k].get<double>(), x.pbox[k] = u["pbox"][k].get<double>();
    x.text = u["text"].get<std::string>();
    x.h = u["h"].get<double>();
    x.size = u.value("size", 0.0);
    x.bold = u["bold"].is_null() ? -1 : (u["bold"].get<bool>() ? 1 : 0);
    x.q = static_cast<uint8_t>(u.value("q", 0));
    x.minor = u.value("minor", false);
    us.push_back(std::move(x));
  }
  return us;
}

Rules rules_from(const json &j) {
  Rules r;
  if (j.is_null()) return r;
  for (const json &s : j["h"]) r.h.push_back({s[0].get<float>(), s[1].get<float>(), s[2].get<float>()});
  for (const json &s : j["v"]) r.v.push_back({s[0].get<float>(), s[1].get<float>(), s[2].get<float>()});
  return r;
}

float max_abs_diff(const std::vector<float> &a, const json &b) {
  if (b.is_null() || b.size() != a.size()) return std::numeric_limits<float>::infinity();
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i].get<float>()));
  return m;
}

// Render at 150 dpi as the reference does (size rounded, white fill, ANNOT | PRINTING), BGR.
std::vector<uint8_t> render_grey(FPDF_PAGE page, int &w, int &h) {
  const double s = kRulesDpi / 72.0;
  w = static_cast<int>(std::round(FPDF_GetPageWidthF(page) * s));
  h = static_cast<int>(std::round(FPDF_GetPageHeightF(page) * s));
  std::vector<uint8_t> bgr(static_cast<size_t>(w) * h * 3);
  FPDF_BITMAP bmp = FPDFBitmap_CreateEx(w, h, FPDFBitmap_BGR, bgr.data(), w * 3);
  FPDFBitmap_FillRect(bmp, 0, 0, w, h, 0xFFFFFFFF);
  FPDF_RenderPageBitmap(bmp, page, 0, 0, w, h, 0, FPDF_ANNOT | FPDF_PRINTING);
  FPDFBitmap_Destroy(bmp);
  return grey_from_bgr(bgr.data(), w, h, static_cast<size_t>(w) * 3);
}

std::vector<std::string> html_of(const std::vector<Table> &ts) {
  std::vector<std::string> v;
  for (const Table &t : ts) v.push_back(t.grid.to_html());
  return v;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: tablenet_parity <doc.pdf> <dump_dir> <tablenet.plan> [max_pages]\n");
    return 2;
  }
  const size_t max_pages = argc > 4 ? std::stoul(argv[4]) : 1u << 30;
  FPDF_InitLibrary();
  FPDF_DOCUMENT doc = FPDF_LoadDocument(argv[1], nullptr);
  if (!doc) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  TableNet net;
  if (!net.load(argv[3])) {
    std::fprintf(stderr, "cannot load %s\n", argv[3]);
    return 2;
  }
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<fs::path> files;
  for (const auto &e : fs::directory_iterator(argv[2]))
    if (e.path().extension() == ".json") files.push_back(e.path());
  std::sort(files.begin(), files.end());
  if (files.size() > max_pages) files.resize(max_pages);

  struct {
    int pages = 0, native = 0, units_ok = 0, rules_ok = 0, img_ok = 0, feats_ok = 0, pairf_ok = 0, logit_pages = 0,
        tables_ok = 0, e2e_ok = 0, ref_tables = 0, e2e_tables = 0;
    float logit_max = 0, feat_max = 0;
    long img_bytes_off = 0;
  } S;
  for (const fs::path &f : files) {
    std::ifstream in(f);
    const json j = json::parse(in);
    const int pno = j["page"].get<int>();
    const bool ocr = j["ocr"].get<bool>();
    const std::vector<Unit> ref_units = units_from(j["units"]);
    const int n = static_cast<int>(ref_units.size());
    const double pw = j["wh"][0].get<double>(), ph = j["wh"][1].get<double>();
    std::vector<std::string> ref_html;
    for (const json &t : j["tables"]) ref_html.push_back(t["html"].get<std::string>());
    S.ref_tables += static_cast<int>(ref_html.size());
    ++S.pages;
    std::string notes;

    FPDF_PAGE page = FPDF_LoadPage(doc, pno - 1);
    // units (text-layer pages)
    PageUnits mine;
    if (!ocr) {
      ++S.native;
      mine = native_units(page);
      bool ok = mine.units.size() == ref_units.size();
      for (size_t i = 0; ok && i < mine.units.size(); ++i) {
        const Unit &a = mine.units[i], &b = ref_units[i];
        ok = a.text == b.text && a.q == b.q && a.minor == b.minor && a.bold == b.bold;
        for (int k = 0; ok && k < 4; ++k) ok = std::abs(a.bx[k] - b.bx[k]) < 1e-3 && std::abs(a.pbox[k] - b.pbox[k]) < 1e-3;
      }
      S.units_ok += ok;
      if (!ok) notes += " units(" + std::to_string(mine.units.size()) + " vs " + std::to_string(ref_units.size()) + ")";
    }
    // render -> rules + image
    int rw = 0, rh = 0;
    const std::vector<uint8_t> grey = render_grey(page, rw, rh);
    const Rules rules = detect_rules(grey, rw, rh, kRulesDpi / 72.f);
    const Rules ref_rules = rules_from(j["rules"]);
    const bool rules_ok = rules.h == ref_rules.h && rules.v == ref_rules.v;
    S.rules_ok += rules_ok;
    if (!rules_ok)
      notes += " rules(h " + std::to_string(rules.h.size()) + "/" + std::to_string(ref_rules.h.size()) + " v " +
               std::to_string(rules.v.size()) + "/" + std::to_string(ref_rules.v.size()) + ")";
    const PageImage img = page_image(grey, rw, rh, kRulesDpi / 72.f);
    PageImage ref_img;
    ref_img.w = j["img"]["w"].get<int>();
    ref_img.h = j["img"]["h"].get<int>();
    ref_img.k = j["img"]["k"].get<float>();
    {
      fs::path ip = f;
      ip.replace_filename(f.stem().string() + "_img.u8");
      std::ifstream is(ip, std::ios::binary);
      ref_img.px.assign(std::istreambuf_iterator<char>(is), {});
    }
    long off = 0;
    if (img.px.size() == ref_img.px.size())
      for (size_t i = 0; i < img.px.size(); ++i) off += img.px[i] != ref_img.px[i];
    else
      off = static_cast<long>(ref_img.px.size());
    S.img_ok += off == 0;
    S.img_bytes_off += off;
    if (off) notes += " img(" + std::to_string(off) + " px differ)";

    // features from the reference units and rules
    const std::vector<float> feats = unit_features(ref_units, pw, ph, &ref_rules);
    const float fd = max_abs_diff(feats, j["feats"]);
    S.feat_max = std::max(S.feat_max, fd);
    S.feats_ok += fd < 1e-4f;
    if (fd >= 1e-4f) {
      notes += " feats(" + std::to_string(fd) + ")";
      int shown = 0;
      for (size_t k = 0; k < feats.size() && shown < 3; ++k) {
        const float r = j["feats"][k].get<float>();
        if (std::abs(feats[k] - r) < 1e-4f) continue;
        const size_t u = k / kFeatDim;
        notes += " [u" + std::to_string(u) + " d" + std::to_string(k % kFeatDim) + " " + std::to_string(feats[k]) + "/" +
                 std::to_string(r) + " '" + ref_units[u].text + "']";
        ++shown;
      }
    }
    std::vector<float> pairf = pair_rule_features(ref_units, ref_rules);
    if (!j["pairf"].is_null()) {
      const float pd = max_abs_diff(pairf, j["pairf"]);
      S.pairf_ok += pd < 1e-6f;
      if (pd >= 1e-6f) notes += " pairf(" + std::to_string(pd) + ")";
    } else {
      ++S.pairf_ok;
    }

    // network on the reference inputs
    const ImgInput ii = image_input(ref_img, ref_units);
    Logits l = net.forward(feats, pairf, n, ii, stream);
    const float ld = std::max({max_abs_diff(l.node, j["node"]), max_abs_diff(l.cls, j["cls"]),
                               j["pair"].is_null() ? 0.f : max_abs_diff(l.pair, j["pair"])});
    S.logit_max = std::max(S.logit_max, ld);
    ++S.logit_pages;
    if (ld > 1e-2f) notes += " logits(" + std::to_string(ld) + ")";

    // decode of the TRT logits
    {
      std::vector<bool> minor(n);
      for (int i = 0; i < n; ++i) minor[i] = ref_units[i].minor;
      mask_minor(l, minor);
      std::vector<DUnit> du;
      for (const Unit &u : ref_units) du.push_back({u.bx, u.text, u.h});
      std::vector<std::string> html;
      for (std::vector<int> g : decode_tables(du, l)) {
        std::erase_if(g, [&](int i) { return minor[i]; });
        if (g.size() < 2) continue;
        const Grid grid = decode_grid(du, g, l);
        if (grid.n_cols < 2 || grid.is_toc() || grid.is_chart() || grid.is_prose()) continue;
        html.push_back(grid.to_html());
      }
      const bool ok = html == ref_html;
      S.tables_ok += ok;
      if (!ok) notes += " tables(" + std::to_string(html.size()) + " vs " + std::to_string(ref_html.size()) + ")";
    }

    // end to end from hp's own units and render
    {
      PageInput pi;
      pi.units = ocr ? ref_units : mine.units;
      pi.w = ocr ? pw : mine.w;
      pi.h = ocr ? ph : mine.h;
      pi.ocr = ocr;
      pi.grey = grey;
      pi.px_w = rw;
      pi.px_h = rh;
      pi.px_per_pt = kRulesDpi / 72.f;
      const auto tables = run_page(net, pi, stream);
      const auto html = html_of(tables);
      S.e2e_tables += static_cast<int>(html.size());
      const bool ok = html == ref_html;
      S.e2e_ok += ok;
      if (!ok) notes += " e2e(" + std::to_string(html.size()) + " vs " + std::to_string(ref_html.size()) + ")";
    }
    FPDF_ClosePage(page);
    std::printf("p%04d n=%4d %s%s\n", pno, n, notes.empty() ? "OK" : "DIFF", notes.c_str());
  }
  std::printf(
      "\n%d pages (%d text-layer): units %d/%d, rules %d/%d, image %d/%d (%ld px off), feats %d/%d (max %.2g), "
      "pairf %d/%d, logits max |d| %.3g over %d pages, tables %d/%d, e2e %d/%d (tables: ref %d, e2e %d)\n",
      S.pages, S.native, S.units_ok, S.native, S.rules_ok, S.pages, S.img_ok, S.pages, S.img_bytes_off, S.feats_ok,
      S.pages, S.feat_max, S.pairf_ok, S.pages, S.logit_max, S.logit_pages, S.tables_ok, S.pages, S.e2e_ok, S.pages,
      S.ref_tables, S.e2e_tables);
  FPDF_CloseDocument(doc);
  FPDF_DestroyLibrary();
  return 0;
}
