// formula_preprocess vs a frozen golden vector.
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "formula.h"

int main() {
  const int W = 18, H = 11, S = hp::kFormulaS;
  std::vector<uint8_t> bgr(W * H * 3);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      uint8_t *p = &bgr[(y * W + x) * 3];
      p[0] = (uint8_t)((x * 17 + y * 3) & 0xFF);
      p[1] = (uint8_t)((x * 5 + y * 29) & 0xFF);
      p[2] = (uint8_t)((x * 11 + y * 7) & 0xFF);
    }
  std::vector<float> out(S * S);
  hp::formula_preprocess(bgr.data(), W * 3, W, H, out.data());
  double sum = 0;
  float mn = 1e9f, mx = -1e9f;
  for (float v : out) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
  struct { int i; float v; } g[] = {{0, -4.563291f}, {383, -4.563291f}, {147072, -4.563291f}, {147455, -4.563291f},
      {73920, -0.778142f}, {44220, -3.492012f}, {50020, -2.851790f}, {57800, -1.580575f}, {69310, -1.047327f},
      {77050, 0.068695f}, {88620, -2.422900f}, {54143, -0.917225f}, {72960, -2.406451f}, {36960, -3.618392f},
      {96130, -2.920880f}, {104060, -1.599934f}};
  float worst = 0;
  for (auto &s : g) worst = std::max(worst, std::fabs(out[s.i] - s.v));
  const bool ok = std::fabs(sum - (-433490.671810)) < 50.0 && std::fabs(mn - (-4.563291f)) < 1e-4f &&
                  std::fabs(mx - 0.610179f) < 1e-3f && worst < 1e-3f;
  std::printf("sum %.6f (golden -433490.671810)  min %.6f  max %.6f  worst sample |diff| %.2e -> %s\n", sum, mn, mx,
              worst, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
