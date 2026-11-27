// Formula recognition on crops dumped by HP_DUMP_FORMULA (c_<n>.bgr): timing and LaTeX per crop.
//   formula_bench <plan dir>/formula <dump dir> <out.txt> [split] [shuffle seed]
// The crops run in groups of `split` (default: all at once), as the assembly thread's batches do;
// with a seed they are shuffled first (results are written back in dump order).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "formula.h"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: formula_bench <formula plan dir> <dump dir> <out.txt> [split] [seed]\n");
    return 2;
  }
  const std::string dir = argv[2];
  std::vector<std::vector<uint8_t>> px;
  std::vector<hp::FormulaCrop> crops;
  for (int n = 1;; ++n) {  // c_0 is the warm-up crop
    std::ifstream f(dir + "/c_" + std::to_string(n) + ".bgr", std::ios::binary);
    if (!f) {
      break;
    }
    int32_t wh[2];
    f.read(reinterpret_cast<char *>(wh), sizeof wh);
    px.emplace_back((size_t)wh[0] * wh[1] * 3);
    f.read(reinterpret_cast<char *>(px.back().data()), (std::streamsize)px.back().size());
    crops.push_back({nullptr, (size_t)wh[0] * 3, wh[0], wh[1]});
  }
  for (size_t i = 0; i < crops.size(); ++i) crops[i].bgr = px[i].data();
  const size_t split = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : crops.size();
  std::vector<size_t> order(crops.size());
  std::iota(order.begin(), order.end(), 0);
  if (argc > 5) std::shuffle(order.begin(), order.end(), std::mt19937(std::atoi(argv[5])));

  hp::FormulaRecognizer fr;
  if (!fr.load(argv[1])) return 1;
  if (const char *d = std::getenv("HP_DUMP_FORMULA")) fr.dump_dir = d;
  cudaStream_t s;
  int lo = 0, hi = 0;
  cudaDeviceGetStreamPriorityRange(&lo, &hi);
  cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, std::getenv("HP_PRIO") ? hi : lo);
  (void)fr.run({crops[0]}, s);  // warm-up
  fr.steps_run = 0;
  std::vector<std::string> res(crops.size());
  const auto t = std::chrono::steady_clock::now();
  for (size_t g = 0; g < order.size(); g += split) {
    std::vector<hp::FormulaCrop> part;
    for (size_t i = g; i < std::min(order.size(), g + split); ++i) part.push_back(crops[order[i]]);
    auto r = fr.run(part, s);
    for (size_t i = 0; i < r.size(); ++i) res[order[g + i]] = std::move(r[i]);
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
  size_t free_b, total_b;
  cudaMemGetInfo(&free_b, &total_b);
  std::printf("%zu crops in %.0f ms (%.2f ms each), %d decode steps (%.3f ms/step), VRAM used %.0f MB\n", crops.size(),
              ms, ms / crops.size(), fr.steps_run, ms / std::max(1, fr.steps_run), (total_b - free_b) / 1048576.0);
  std::ofstream out(argv[3]);
  for (auto &l : res) out << l << '\n';
  return 0;
}
