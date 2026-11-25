// OrderNet on TensorRT against the Python reference (ONNX Runtime) dumps.
//
//   ordernet_trt_check <order.plan> <dump_dir>...
//
// Per dumped page with network outputs: hp's features (line_features) -> TensorRT -> heads, compared
// with the dump's ONNX Runtime heads (max |diff| of a, cls, same); then the order and Markdown
// decoded from the TensorRT heads against the dump's (how often the GPU path gives the same result).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "ordernet/net.h"
#include "ordernet/order.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace hp::on;

namespace {
float max_diff(const std::vector<float> &a, const json &b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i].get<float>()));
  return m;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <order.plan> <dump_dir>...\n", argv[0]);
    return 2;
  }
  OrderNet net;
  if (!net.load(argv[1])) return 2;
  cudaStream_t s;
  cudaStreamCreate(&s);
  int pages = 0, order_same = 0, md_same = 0;
  float worst = 0;
  for (int ai = 2; ai < argc; ++ai) {
    std::vector<fs::path> files;
    for (const auto &e : fs::directory_iterator(argv[ai]))
      if (e.path().extension() == ".json") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const auto &f : files) {
      std::ifstream in(f);
      const json j = json::parse(in);
      if (!j.contains("a")) continue;
      std::vector<Line> lines;
      for (size_t i = 0; i < j["boxes"].size(); ++i) {
        Line l;
        for (int k = 0; k < 4; ++k) l.box[k] = j["boxes"][i][k].get<float>();
        l.text = j["texts"][i].get<std::string>();
        lines.push_back(std::move(l));
      }
      const int n = static_cast<int>(lines.size());
      const Heads h = net.forward(line_features(lines, j["wh"][0].get<float>(), j["wh"][1].get<float>()), n, s);
      const float d = std::max({max_diff(h.a, j["a"]), max_diff(h.cls, j["cls"]), max_diff(h.same, j["same"])});
      worst = std::max(worst, d);
      const std::vector<int> order = decode_blocks_order(h, lines);
      const bool os = order == j["order"].get<std::vector<int>>();
      const bool ms = page_markdown(lines, order, h) == j["markdown"].get<std::string>();
      ++pages, order_same += os, md_same += ms;
      if (!os || !ms || d > 1e-3f)
        std::printf("%s n=%d max|d|=%.2g order %s markdown %s\n", f.filename().c_str(), n, d, os ? "same" : "DIFF",
                    ms ? "same" : "DIFF");
    }
  }
  std::printf("%d pages: heads max |diff| %.3g; order identical %d/%d, Markdown identical %d/%d\n", pages, worst,
              order_same, pages, md_same, pages);
  return 0;
}
