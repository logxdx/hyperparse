// Parity test for the TableNet decoder (src/tablenet/decode.cpp) against page dumps written by
// the Rust reference (liteparse table_model, LP_TN_DUMP).
//
//   tablenet_decode_test <dump_dir> [<dump_dir> ...]
//
// Each dump dir holds p####.json files with the page's units, the RAW logits (before
// mask_minor) and the tables the reference emitted. Per page this replays mod.rs
// run_page_framed: mask_minor, decode_tables, the minor filter, decode_grid and the grid filters
// (n_cols < 2, is_toc, is_chart, is_prose), and compares units, HTML, n_rows, n_cols and
// head_rows exactly. Pages whose pair logits were not dumped (n > 600) are skipped.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tablenet/decode.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

struct Table {
  std::vector<int> units;
  std::string html;
  int n_rows = 0, n_cols = 0;
  std::vector<int> head_rows;
};

std::string join(const std::vector<int> &v) {
  std::string s = "[";
  for (size_t k = 0; k < v.size(); ++k) s += (k ? "," : "") + std::to_string(v[k]);
  return s + "]";
}

// The first line where two strings differ, both sides.
std::string first_diff(const std::string &a, const std::string &b) {
  std::istringstream sa(a), sb(b);
  std::string la, lb;
  int line = 0;
  while (true) {
    const bool ga = static_cast<bool>(std::getline(sa, la));
    const bool gb = static_cast<bool>(std::getline(sb, lb));
    if (!ga && !gb) return "(identical)";
    ++line;
    if (!ga || !gb || la != lb) {
      return "line " + std::to_string(line) + "\n      want: " + (gb ? lb : "<eof>") +
             "\n      got:  " + (ga ? la : "<eof>");
    }
  }
}

std::vector<Table> run_page(const json &d) {
  using namespace hp::tn;
  const int n = d["n"].get<int>();
  Logits l;
  l.n = n;
  l.node = d["node"].get<std::vector<float>>();
  l.pair = d["pair"].get<std::vector<float>>();
  if (d.contains("cls") && d["cls"].is_array()) l.cls = d["cls"].get<std::vector<float>>();
  l.n_pair = static_cast<int>(l.pair.size() / (static_cast<size_t>(n) * n));
  l.n_cls = n > 0 ? static_cast<int>(l.cls.size() / n) : 0;
  std::vector<bool> minor;
  std::vector<DUnit> du;
  for (const auto &u : d["units"]) {
    minor.push_back(u["minor"].get<bool>());
    const auto b = u["bx"].get<std::vector<double>>();
    du.push_back({{b[0], b[1], b[2], b[3]}, u["text"].get<std::string>(), u["h"].get<double>()});
  }
  mask_minor(l, minor);
  const DecodeOptions opt{};
  std::vector<Table> out;
  for (auto g : decode_tables(du, l, opt)) {
    std::erase_if(g, [&](int i) { return minor[i]; });
    if (g.size() < 2) continue;
    const Grid grid = decode_grid(du, g, l, opt);
    if (grid.n_cols < 2 || grid.is_toc() || grid.is_chart() || grid.is_prose()) continue;
    out.push_back({g, grid.to_html(), grid.n_rows, grid.n_cols, grid.head_rows});
  }
  return out;
}

// Empty when equal, else a short description of the first difference.
std::string compare(const std::vector<Table> &want, const std::vector<Table> &got) {
  std::string diff;
  if (want.size() != got.size())
    diff += "  table count: want " + std::to_string(want.size()) + ", got " +
            std::to_string(got.size()) + "\n";
  for (size_t t = 0; t < std::min(want.size(), got.size()); ++t) {
    const Table &w = want[t], &g = got[t];
    const std::string at = "  table " + std::to_string(t) + ": ";
    if (w.units != g.units) diff += at + "units want " + join(w.units) + " got " + join(g.units) + "\n";
    if (w.n_rows != g.n_rows || w.n_cols != g.n_cols)
      diff += at + "size want " + std::to_string(w.n_rows) + "x" + std::to_string(w.n_cols) +
              " got " + std::to_string(g.n_rows) + "x" + std::to_string(g.n_cols) + "\n";
    if (w.head_rows != g.head_rows)
      diff += at + "head_rows want " + join(w.head_rows) + " got " + join(g.head_rows) + "\n";
    if (w.html != g.html) diff += at + "html differs at " + first_diff(g.html, w.html) + "\n";
  }
  if (want.size() != got.size()) {
    for (size_t t = std::min(want.size(), got.size()); t < std::max(want.size(), got.size()); ++t) {
      const Table &x = t < want.size() ? want[t] : got[t];
      diff += std::string("  extra table in ") + (t < want.size() ? "want" : "got") + ": units " +
              join(x.units) + "\n";
    }
  }
  return diff;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <dump_dir> [<dump_dir> ...]\n", argv[0]);
    return 2;
  }
  size_t total = 0, passed = 0, skipped = 0;
  for (int a = 1; a < argc; ++a) {
    std::vector<fs::path> files;
    for (const auto &e : fs::directory_iterator(argv[a])) {
      const std::string name = e.path().filename().string();
      if (name.size() > 5 && name[0] == 'p' && name.ends_with(".json")) files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    size_t dir_total = 0, dir_passed = 0;
    for (const auto &f : files) {
      std::ifstream in(f);
      const json d = json::parse(in);
      if (d["pair"].is_null()) {
        ++skipped;
        std::printf("SKIP %s (no pair logits)\n", f.string().c_str());
        continue;
      }
      std::vector<Table> want;
      for (const auto &t : d["tables"])
        want.push_back({t["units"].get<std::vector<int>>(), t["html"].get<std::string>(),
                        t["n_rows"].get<int>(), t["n_cols"].get<int>(),
                        t["head_rows"].get<std::vector<int>>()});
      const std::string diff = compare(want, run_page(d));
      ++dir_total;
      if (diff.empty()) {
        ++dir_passed;
        std::printf("PASS %s (%zu tables)\n", f.string().c_str(), want.size());
      } else {
        std::printf("FAIL %s\n%s", f.string().c_str(), diff.c_str());
      }
    }
    std::printf("== %s: %zu pages, %zu passed\n", argv[a], dir_total, dir_passed);
    total += dir_total;
    passed += dir_passed;
  }
  std::printf("%zu pages, %zu passed (%zu skipped)\n", total, passed, skipped);
  return passed == total ? 0 : 1;
}
