// Parity + timing: hp::CropBatch (one launch) vs the core/kernels launches it replaces
// (cuda_batch_roi_warp per rec batch, cuda_fused_slanext_pre per table).
//   crop_batch_test [n_lines=90] [n_tables=2]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "crop_batch.h"
#include "core/common/cuda/cuda_check.h"
#include "core/common/geometry/perspective.h"
#include "core/kernels/kernels.h"
#include "core/recognition/rec_geometry.h"

using hp::core::Box;
using hp::core::CudaPtr;

int main(int argc, char **argv) {
  const int n_lines = argc > 1 ? std::atoi(argv[1]) : 90;
  const int n_tables = argc > 2 ? std::atoi(argv[2]) : 2;
  const int W = 1275, H = 1650;  // Letter @150 DPI
  std::mt19937 rng(7);
  std::vector<uint8_t> page(static_cast<size_t>(W) * H * 3);
  for (auto &v : page) v = static_cast<uint8_t>(rng() & 255);
  CudaPtr<uint8_t> d_page(page.size());
  CUDA_CHECK(cudaMemcpy(d_page.get(), page.data(), page.size(), cudaMemcpyHostToDevice));
  hp::core::GpuImage img{d_page.get(), static_cast<size_t>(W) * 3, H, W};

  // Text lines: slightly rotated quads, widths 20..1100 px, heights 14..40 px.
  std::uniform_real_distribution<float> U(0.f, 1.f);
  std::vector<Box> boxes(n_lines);
  std::vector<int> bucket(n_lines);
  for (int i = 0; i < n_lines; ++i) {
    const float w = 20 + 1080 * U(rng) * U(rng), h = 14 + 26 * U(rng);
    const float cx = w / 2 + (W - w) * U(rng), cy = h / 2 + (H - h) * U(rng);
    const float a = (U(rng) - 0.5f) * 0.06f, ca = std::cos(a), sa = std::sin(a);
    const float dx[4] = {-w / 2, w / 2, w / 2, -w / 2}, dy[4] = {-h / 2, -h / 2, h / 2, h / 2};
    for (int k = 0; k < 4; ++k)
      boxes[i][k] = {static_cast<int>(std::lround(cx + dx[k] * ca - dy[k] * sa)),
                     static_cast<int>(std::lround(cy + dx[k] * sa + dy[k] * ca))};
    bucket[i] = hp::core::recognition::snap_width_bucket(hp::core::recognition::rec_input_width(boxes[i], 48));
  }
  // Group into batches (same bucket, <= 32) like PaddleRec.
  std::vector<int> order(n_lines);
  for (int i = 0; i < n_lines; ++i) order[i] = i;
  std::ranges::sort(order, {}, [&](int i) { return bucket[i]; });
  struct B { int beg, end, w; size_t off; };
  std::vector<B> batches;
  size_t total = 0;
  for (int beg = 0; beg < n_lines;) {
    int end = beg;
    while (end < n_lines && end - beg < 32 && bucket[order[end]] == bucket[order[beg]]) ++end;
    batches.push_back({beg, end, bucket[order[beg]], total});
    total += static_cast<size_t>(end - beg) * 3 * 48 * bucket[order[beg]];
    beg = end;
  }
  struct T { int x, y, w, h; };
  std::vector<T> tables;
  for (int i = 0; i < n_tables; ++i) {
    const int tw = 200 + rng() % 900, th = 100 + rng() % 700;
    tables.push_back({static_cast<int>(rng() % (W - tw)), static_cast<int>(rng() % (H - th)), tw, th});
  }
  const size_t table_elems = 3ull * 488 * 488;
  CudaPtr<float> ref(total + n_tables * table_elems), got(total + n_tables * table_elems);
  CudaPtr<float> d_M(n_lines * 9);
  CudaPtr<int> d_w(n_lines);
  hp::core::CudaHostPtr<float> hM(n_lines * 9);
  hp::core::CudaHostPtr<int> hw(n_lines);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  auto reference = [&] {
    // As PaddleRec: per batch, 2 H2D (transforms, widths) + 1 warp launch, no sync.
    for (const auto &b : batches) {
      for (int j = b.beg; j < b.end; ++j) {
        const auto ct = hp::core::compute_crop_transform(boxes[order[j]], 48, b.w);
        std::copy(ct.M_inv, ct.M_inv + 9, hM.get() + 9 * j);
        hw.get()[j] = ct.crop_width;
      }
      CUDA_CHECK(cudaMemcpyAsync(d_M.get() + 9 * b.beg, hM.get() + 9 * b.beg, (b.end - b.beg) * 9 * 4, cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(d_w.get() + b.beg, hw.get() + b.beg, (b.end - b.beg) * 4, cudaMemcpyHostToDevice, s));
      hp::core::kernels::cuda_batch_roi_warp(img, d_M.get() + 9 * b.beg, d_w.get() + b.beg, ref.get() + b.off,
                                              b.end - b.beg, 48, b.w, s);
    }
    for (int i = 0; i < n_tables; ++i)
      hp::core::kernels::cuda_fused_slanext_pre(img, tables[i].x, tables[i].y, tables[i].w, tables[i].h,
                                                  ref.get() + total + i * table_elems, s);
  };
  hp::CropBatch cb;
  auto fused = [&] {
    cb.clear();
    for (const auto &b : batches)
      for (int j = b.beg; j < b.end; ++j)
        cb.add_rec_line(boxes[order[j]], got.get() + b.off + static_cast<size_t>(j - b.beg) * 3 * 48 * b.w, b.w);
    for (int i = 0; i < n_tables; ++i)
      cb.add_table(tables[i].x, tables[i].y, tables[i].w, tables[i].h, got.get() + total + i * table_elems);
    cb.launch(d_page.get(), static_cast<size_t>(W) * 3, W, H, s);
  };

  CUDA_CHECK(cudaMemset(ref.get(), 0x7f, (total + n_tables * table_elems) * 4));
  CUDA_CHECK(cudaMemset(got.get(), 0x7f, (total + n_tables * table_elems) * 4));
  reference();
  fused();
  CUDA_CHECK(cudaStreamSynchronize(s));
  const size_t N = total + n_tables * table_elems;
  std::vector<float> a(N), b(N);
  CUDA_CHECK(cudaMemcpy(a.data(), ref.get(), N * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b.data(), got.get(), N * 4, cudaMemcpyDeviceToHost));
  size_t diff = 0;
  float maxd = 0;
  for (size_t i = 0; i < N; ++i) {
    const float d = std::fabs(a[i] - b[i]);
    if (d > 0) ++diff;
    maxd = std::max(maxd, d);
  }
  std::printf("%d lines in %zu rec batches + %d tables: %zu floats, %zu differ, max |diff| %.3g -> %s\n", n_lines,
              batches.size(), n_tables, N, diff, maxd, diff == 0 ? "BIT-EXACT" : (maxd < 1e-5 ? "ULP-LEVEL" : "MISMATCH"));

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  auto time = [&](auto fn, const char *name) {
    for (int i = 0; i < 5; ++i) fn();
    CUDA_CHECK(cudaStreamSynchronize(s));
    const auto t0 = std::chrono::steady_clock::now();
    cudaEventRecord(e0, s);
    for (int i = 0; i < 50; ++i) fn();
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 50;
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    std::printf("  %-34s GPU %.3f ms/page   host wall %.3f ms/page\n", name, ms / 50, wall);
  };
  time([&] { reference(); CUDA_CHECK(cudaStreamSynchronize(s)); }, "core/kernels: launch per batch/table");
  time([&] { fused(); CUDA_CHECK(cudaStreamSynchronize(s)); }, "hp CropBatch: one launch");
  return diff == 0 || maxd < 1e-5 ? 0 : 1;
}
