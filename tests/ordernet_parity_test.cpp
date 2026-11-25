// Parity of hp's OrderNet post-processing (src/ordernet/order.cpp) with indra_ocr's
// order_post.py / layout_post.py, from per-page dumps of the Python reference (CPU only: the
// network outputs are in the dumps).
//
//   ordernet_parity_test <dump_dir>...
//
// Per page (p####.json: page, wh, boxes, texts, feats, order, blocks, markdown, and a / cls /
// same when N <= 500):
//   feats     line_features() vs feats (max abs diff <= 1e-5)
//   order     decode_blocks_order() vs order. numpy's argsort is unstable, so a mismatch is
//             re-tried with random tie-breaks in the XY-cut / top-down argsorts (a test-side
//             copy of the in-block ordering); one that reproduces the reference is "tie-only"
//   blocks    split_blocks(reference order) vs blocks
//   markdown  page_markdown(reference order) vs markdown (byte-exact); also with our order
// Exit status 1 on any feats / real order / blocks / markdown failure.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ordernet/order.h"

namespace hp::on::detail {
std::vector<std::vector<int>> cluster(const std::vector<float> &logits, int n);
}

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace hp::on;
using Box = std::array<float, 4>;

namespace {

// ---- tie oracle: XY-cut / top-down with a given tie-break rank in every argsort ----------

struct Oracle {
  std::mt19937 rng;
  bool randomize;

  std::vector<int> argsort(const std::vector<float> &v) {
    std::vector<int> o(v.size()), rank(v.size());
    std::iota(o.begin(), o.end(), 0);
    std::iota(rank.begin(), rank.end(), 0);
    if (randomize) std::shuffle(rank.begin(), rank.end(), rng);
    std::sort(o.begin(), o.end(), [&](int i, int j) { return v[i] != v[j] ? v[i] < v[j] : rank[i] < rank[j]; });
    return o;
  }

  std::vector<int> xycut(const std::vector<Box> &boxes) {
    std::vector<int> ids(boxes.size()), out;
    std::iota(ids.begin(), ids.end(), 0);
    rec(boxes, ids, out);
    return out;
  }

  void rec(const std::vector<Box> &boxes, const std::vector<int> &ids, std::vector<int> &out) {
    if (ids.size() <= 1) {
      out.insert(out.end(), ids.begin(), ids.end());
      return;
    }
    std::vector<Box> b;
    for (int i : ids) b.push_back(boxes[i]);
    float g[2];
    int kk[2];
    std::vector<int> o[2];
    for (int axis = 1; axis >= 0; --axis) {
      const int lo_i = axis ? 1 : 0, hi_i = axis ? 3 : 2;
      std::vector<float> lo;
      for (const Box &x : b) lo.push_back(x[lo_i]);
      auto &oo = o[axis];
      oo = argsort(lo);
      float end = b[oo[0]][hi_i], best = 0.0f;
      int bk = -1;
      for (size_t k = 1; k < oo.size(); ++k) {
        const float gap = lo[oo[k]] - end;
        if (gap > best) best = gap, bk = static_cast<int>(k);
        end = std::max(end, b[oo[k]][hi_i]);
      }
      g[axis] = best, kk[axis] = bk;
    }
    if (kk[1] < 0 && kk[0] < 0) {
      for (int i : order_top_down(b)) out.push_back(ids[i]);
      return;
    }
    const int ax = g[1] >= g[0] ? 1 : 0;
    std::vector<int> first, second;
    for (int k = 0; k < static_cast<int>(o[ax].size()); ++k) (k < kk[ax] ? first : second).push_back(ids[o[ax][k]]);
    rec(boxes, first, out);
    rec(boxes, second, out);
  }
};

// decode_blocks_order with the oracle's in-block ordering (blocks and block order as the
// library computes them: those have no ties).
std::vector<int> oracle_order(const Heads &h, const std::vector<Line> &lines, Oracle &orc) {
  const auto blocks = detail::cluster(h.same, h.n);
  std::vector<std::vector<int>> inner;
  for (const auto &b : blocks) {
    std::vector<Box> bx;
    for (int i : b) bx.push_back(lines[i].box);
    std::array<float, kClasses> s{};
    for (int i : b)
      for (int c = 0; c < kClasses; ++c) s[c] += h.cls[i * kClasses + c];
    const int cls = static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin());
    const bool grid = cls == 5 || cls == 8 || cls == 9;
    std::vector<int> in;
    for (int k : grid ? order_top_down(bx) : orc.xycut(bx)) in.push_back(b[k]);
    inner.push_back(std::move(in));
  }
  // Block order: blocks are contiguous runs of the library's order (it has no ties there).
  const auto lib = decode_blocks_order(h, lines);
  std::vector<int> block_of(h.n);
  for (size_t k = 0; k < blocks.size(); ++k)
    for (int i : blocks[k]) block_of[i] = static_cast<int>(k);
  std::vector<int> out;
  for (size_t p = 0; p < lib.size(); ++p)
    if (p == 0 || block_of[lib[p]] != block_of[lib[p - 1]])
      out.insert(out.end(), inner[block_of[lib[p]]].begin(), inner[block_of[lib[p]]].end());
  return out;
}

std::string short_diff(const std::string &a, const std::string &b) {
  size_t i = 0;
  while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
  const size_t from = i > 40 ? i - 40 : 0;
  auto cut = [&](const std::string &s) {
    std::string t = s.substr(from, 120);
    for (char &c : t)
      if (c == '\n') c = '|';
    return t;
  };
  return "at byte " + std::to_string(i) + "\n      ours: " + cut(a) + "\n      ref:  " + cut(b);
}

std::string ints(const std::vector<int> &v, size_t from, size_t n) {
  std::string s;
  for (size_t i = from; i < v.size() && i < from + n; ++i) s += std::to_string(v[i]) + " ";
  return s;
}

struct Summary {
  int pages = 0, with_heads = 0, feats_ok = 0, order_exact = 0, order_tie = 0, order_real = 0, blocks_ok = 0,
      md_ok = 0, md_own_ok = 0;
  double feats_max = 0;
};

struct Timing {
  int n;
  double ms;
  std::string page;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <dump_dir>...\n", argv[0]);
    return 2;
  }
  bool fail = false;
  std::vector<Timing> timings;
  std::vector<std::pair<std::string, Summary>> sums;
  for (int ai = 1; ai < argc; ++ai) {
    const fs::path dir = argv[ai];
    std::vector<fs::path> files;
    for (const auto &e : fs::directory_iterator(dir))
      if (e.path().extension() == ".json") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    Summary S;
    for (const auto &f : files) {
      std::ifstream in(f);
      const json j = json::parse(in);
      const std::string name = dir.filename().string() + "/" + f.filename().string();
      ++S.pages;
      std::vector<Line> lines;
      for (size_t i = 0; i < j["boxes"].size(); ++i) {
        Line l;
        for (int k = 0; k < 4; ++k) l.box[k] = j["boxes"][i][k].get<float>();
        l.text = j["texts"][i].get<std::string>();
        lines.push_back(std::move(l));
      }
      const int n = static_cast<int>(lines.size());
      const float W = j["wh"][0].get<float>(), H = j["wh"][1].get<float>();
      std::vector<std::string> notes;

      const auto t0 = std::chrono::steady_clock::now();
      const auto feats = line_features(lines, W, H);
      const auto t1 = std::chrono::steady_clock::now();
      const auto ref_feats = j["feats"].get<std::vector<float>>();
      double fmax = ref_feats.size() == feats.size() ? 0.0 : INFINITY;
      int fat = -1;
      for (size_t i = 0; i < feats.size() && i < ref_feats.size(); ++i) {
        const double d = std::fabs(static_cast<double>(feats[i]) - ref_feats[i]);
        if (d > fmax) fmax = d, fat = static_cast<int>(i);
      }
      S.feats_max = std::max(S.feats_max, fmax);
      const bool feats_ok = fmax <= 1e-5;
      S.feats_ok += feats_ok;
      if (!feats_ok) {
        fail = true;
        notes.push_back("feats max diff " + std::to_string(fmax) + " at line " + std::to_string(fat / kFeats) +
                        " col " + std::to_string(fat % kFeats));
      }

      if (!j.contains("a")) {
        std::printf("%-26s n=%4d %s (no heads)%s\n", name.c_str(), n, feats_ok ? "PASS" : "DIFF",
                    feats_ok ? "" : (" " + notes[0]).c_str());
        continue;
      }
      ++S.with_heads;
      Heads h;
      h.n = n;
      h.a = j["a"].get<std::vector<float>>();
      h.cls = j["cls"].get<std::vector<float>>();
      h.same = j["same"].get<std::vector<float>>();
      const auto ref_order = j["order"].get<std::vector<int>>();

      const auto t2 = std::chrono::steady_clock::now();
      const auto order = decode_blocks_order(h, lines);
      const auto t3 = std::chrono::steady_clock::now();
      const auto md_own = page_markdown(lines, order, h, false);
      const auto t4 = std::chrono::steady_clock::now();
      timings.push_back({n,
                         std::chrono::duration<double, std::milli>((t1 - t0) + (t3 - t2) + (t4 - t3)).count(),
                         name});

      // The oracle without random tie-breaks must reproduce the library (else it proves nothing).
      Oracle orc{std::mt19937(12345), false};
      const bool mirror = oracle_order(h, lines, orc) == order;
      if (!mirror) {
        fail = true;
        notes.push_back("tie oracle does not mirror the library");
      }
      const char *order_tag = "exact";
      if (order == ref_order) {
        ++S.order_exact;
      } else {
        bool tie = false;
        orc.randomize = true;
        for (int t = 0; mirror && t < 300 && !tie; ++t) tie = oracle_order(h, lines, orc) == ref_order;
        size_t p = 0;
        while (p < order.size() && p < ref_order.size() && order[p] == ref_order[p]) ++p;
        const std::string where = "order differs at " + std::to_string(p) + ": ours " + ints(order, p, 8) +
                                  "/ ref " + ints(ref_order, p, 8);
        if (tie) {
          ++S.order_tie;
          order_tag = "tie-only";
          notes.push_back(where + "(reproduced with a random tie-break)");
        } else {
          ++S.order_real;
          order_tag = "REAL";
          fail = true;
          notes.push_back(where);
        }
      }

      const auto blocks = split_blocks(ref_order, h);
      const bool blocks_ok = blocks == j["blocks"].get<std::vector<std::vector<int>>>();
      S.blocks_ok += blocks_ok;
      if (!blocks_ok) {
        fail = true;
        notes.push_back("blocks differ");
      }

      const std::string ref_md = j["markdown"].get<std::string>();
      const auto md = page_markdown(lines, ref_order, h, false);
      const bool md_ok = md == ref_md;
      S.md_ok += md_ok;
      if (!md_ok) {
        fail = true;
        notes.push_back("markdown " + short_diff(md, ref_md));
      }
      S.md_own_ok += md_own == ref_md;

      const bool pass = feats_ok && order == ref_order && blocks_ok && md_ok && mirror;
      std::printf("%-26s n=%4d %s order=%s%s\n", name.c_str(), n, pass ? "PASS" : "DIFF", order_tag,
                  md_own == ref_md ? "" : " (markdown from our order differs)");
      for (const auto &s : notes) std::printf("    %s\n", s.c_str());
    }
    sums.emplace_back(dir.filename().string(), S);
  }

  std::printf("\n%-12s %6s %6s %9s %9s %6s %6s %7s %7s %8s %10s\n", "dir", "pages", "heads", "feats_ok",
              "max_diff", "exact", "tie", "real", "blocks", "md_ok", "md_own_ok");
  for (const auto &[name, S] : sums)
    std::printf("%-12s %6d %6d %9d %9.2g %6d %6d %7d %7d %8d %10d\n", name.c_str(), S.pages, S.with_heads,
                S.feats_ok, S.feats_max, S.order_exact, S.order_tie, S.order_real, S.blocks_ok, S.md_ok,
                S.md_own_ok);
  std::sort(timings.begin(), timings.end(), [](const Timing &a, const Timing &b) { return a.n > b.n; });
  std::printf("\nfeatures + decode + markdown, largest pages:\n");
  for (size_t i = 0; i < timings.size() && i < 5; ++i)
    std::printf("  %-26s n=%4d %7.2f ms\n", timings[i].page.c_str(), timings[i].n, timings[i].ms);
  std::printf("\n%s\n", fail ? "FAIL" : "OK");
  return fail ? 1 : 0;
}
