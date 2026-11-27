// RenderPool fault tolerance: every submitted page must yield exactly one result
// even when workers are killed (crash) or overrun the page timeout, and the pool
// must keep working afterwards; pages of two documents in flight at once each
// render from their own document.
//
//   render_pool_test <doc.pdf>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <unistd.h>

#include <fpdf_edit.h>
#include <fpdf_ppo.h>
#include <fpdf_save.h>
#include <fpdfview.h>

#include "pdf_render.h"
#include "render_pool.h"

namespace {

int failures = 0;
#define CHECK(c)                                                         \
  do {                                                                   \
    if (!(c)) {                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                                        \
    }                                                                    \
  } while (0)

// A new PDF made of the first `n` pages of `pdf` in reverse order (empty on failure).
std::vector<uint8_t> reversed_copy(const std::vector<uint8_t> &pdf, int n) {
  struct Writer : FPDF_FILEWRITE {
    std::vector<uint8_t> out;
  } w;
  w.version = 1;
  w.WriteBlock = [](FPDF_FILEWRITE *self, const void *data, unsigned long size) -> int {
    auto *p = static_cast<const uint8_t *>(data);
    static_cast<Writer *>(self)->out.insert(static_cast<Writer *>(self)->out.end(), p, p + size);
    return 1;
  };
  hp::PdfRaster src(pdf.data(), pdf.size());  // initializes pdfium
  FPDF_DOCUMENT dst = FPDF_CreateNewDocument();
  std::string range;
  for (int i = n; i >= 1; --i) range += std::to_string(i) + (i > 1 ? "," : "");
  const bool ok = dst && src.ok() && FPDF_ImportPages(dst, static_cast<FPDF_DOCUMENT>(src.doc()), range.c_str(), 0) &&
                  FPDF_SaveAsCopy(dst, &w, 0);
  if (dst) FPDF_CloseDocument(dst);
  return ok ? w.out : std::vector<uint8_t>{};
}

// Grandchildren of this process = the pool's workers (children of the supervisor).
std::vector<pid_t> worker_pids() {
  std::vector<pid_t> out;
  auto children_of = [](pid_t p) {
    std::vector<pid_t> c;
    std::ifstream f("/proc/" + std::to_string(p) + "/task/" + std::to_string(p) + "/children");
    for (pid_t x; f >> x;) c.push_back(x);
    return c;
  };
  for (pid_t sup : children_of(getpid()))
    for (pid_t w : children_of(sup)) out.push_back(w);
  return out;
}

struct Counts {
  int ok = 0, crashed = 0, timed_out = 0, other = 0;
  std::set<int> pages;
};

// Submit every page through the slot ring; optionally kill a busy worker after
// `kill_after` results.
Counts run_document(hp::RenderPool &pool, const std::vector<uint8_t> &pdf, int n_pages, int kill_after) {
  const auto doc = pool.set_document(0, pdf);
  const uint32_t gen = doc.gen;
  std::vector<int> free_slots;
  for (int s = 0; s < pool.slots(); ++s) free_slots.push_back(s);
  int next = 0, received = 0;
  Counts c;
  auto fill = [&] {
    while (next < n_pages && !free_slots.empty()) {
      pool.submit(next++, free_slots.back(), doc, {});
      free_slots.pop_back();
    }
  };
  fill();
  while (received < n_pages) {
    const auto d = pool.wait_done();
    if (d.gen != gen || c.pages.count(d.page)) continue;  // duplicate report
    c.pages.insert(d.page);
    ++received;
    switch (d.status) {
      case hp::RenderPool::kOk: ++c.ok; break;
      case hp::RenderPool::kWorkerCrashed: ++c.crashed; break;
      case hp::RenderPool::kTimedOut: ++c.timed_out; break;
      default: ++c.other;
    }
    free_slots.push_back(d.slot);
    if (received == kill_after)
      for (pid_t w : worker_pids()) kill(w, SIGKILL);  // all workers, mid-document
    fill();
  }
  return c;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: render_pool_test <doc.pdf>\n");
    return 2;
  }
  std::ifstream f(argv[1], std::ios::binary);
  const std::vector<uint8_t> pdf((std::istreambuf_iterator<char>(f)), {});
  int n_pages = 0;
  {
    hp::PdfRaster probe(pdf.data(), pdf.size());
    n_pages = probe.page_count();
  }
  CHECK(n_pages > 0);

  hp::RenderPool::Options o;
  o.workers = 3;
  o.slots = 8;
  o.pixel_bytes = static_cast<size_t>(2560) * 3328 * 3;
  o.meta_bytes = 8u << 20;
  {
    hp::RenderPool pool(o);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK(worker_pids().size() == 3);

    // 1. Crash: kill every worker after 10 results.
    const auto t = std::chrono::steady_clock::now();
    const Counts a = run_document(pool, pdf, n_pages, 10);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    std::printf("crash run: %d pages in %.0f ms: %d ok, %d crashed, %d timed out, %d other\n", n_pages, ms, a.ok,
                a.crashed, a.timed_out, a.other);
    CHECK(static_cast<int>(a.pages.size()) == n_pages);
    CHECK(a.crashed >= 1);  // workers were busy: their in-flight pages are reported
    CHECK(a.ok + a.crashed == n_pages);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CHECK(worker_pids().size() == 3);  // respawned

    // 2. Same pool, same document again: fully healthy.
    const Counts b = run_document(pool, pdf, n_pages, -1);
    std::printf("after respawn: %d ok of %d\n", b.ok, n_pages);
    CHECK(b.ok == n_pages);

    // 3. Idle recycling: workers that worked and then idle for 2 s are replaced.
    const auto before = worker_pids();
    std::this_thread::sleep_for(std::chrono::milliseconds(2600));
    const auto after = worker_pids();
    int replaced = 0;
    for (pid_t p : after) replaced += std::find(before.begin(), before.end(), p) == before.end();
    std::printf("idle recycling: %d of %zu workers replaced\n", replaced, after.size());
    CHECK(after.size() == 3 && replaced == 3);
  }
  CHECK(worker_pids().empty());  // destructor reaped supervisor + workers

  // 4. Timeout: a 1 ms budget kills workers mid-page; every page still reported once.
  o.page_timeout = std::chrono::milliseconds(1);
  {
    hp::RenderPool pool(o);
    const Counts c = run_document(pool, pdf, std::min(n_pages, 30), -1);
    std::printf("timeout run: %d ok, %d timed out, %d crashed\n", c.ok, c.timed_out, c.crashed);
    CHECK(static_cast<int>(c.pages.size()) == std::min(n_pages, 30));
    CHECK(c.timed_out >= 1);
  }

  // 5. Two documents in flight at once (two document slots), pages interleaved:
  //    every page renders from its own document. The second document is the
  //    first n pages of the first in reverse (page i of B = page n-1-i of A), so
  //    a worker that rendered from the wrong document would produce other pixels.
  o.page_timeout = std::chrono::milliseconds(120000);
  o.documents = 2;
  {
    const int n = std::min(n_pages, 12);
    const std::vector<uint8_t> reversed = reversed_copy(pdf, n);
    CHECK(!reversed.empty());
    hp::RenderPool pool(o);
    const auto a = pool.set_document(0, pdf);
    const auto b = pool.set_document(1, reversed);
    CHECK(a.gen != b.gen);
    // Checksum per (document, page) of the rendered pixels.
    std::vector<uint64_t> sum_a(n), sum_b(n);
    std::vector<int> free_slots;
    for (int s = 0; s < pool.slots(); ++s) free_slots.push_back(s);
    int next = 0, received = 0, ok = 0;
    auto fill = [&] {
      while (next < 2 * n && !free_slots.empty()) {
        const bool is_a = next % 2 == 0;
        pool.submit(next / 2, free_slots.back(), is_a ? a : b, {});
        free_slots.pop_back();
        ++next;
      }
    };
    fill();
    while (received < 2 * n) {
      const auto d = pool.wait_done();
      ++received;
      if (d.status == hp::RenderPool::kOk) {
        ++ok;
        uint64_t h = 1469598103934665603ull;
        const uint8_t *px = pool.pixels(d.slot);
        for (size_t i = 0; i < static_cast<size_t>(d.w) * d.h * 3; i += 61) h = (h ^ px[i]) * 1099511628211ull;
        if (d.gen == a.gen) sum_a[d.page] = h;
        else sum_b[n - 1 - d.page] = h;  // B's page i is A's page n-1-i
      }
      free_slots.push_back(d.slot);
      fill();
    }
    std::printf("two documents: %d ok of %d\n", ok, 2 * n);
    CHECK(ok == 2 * n);
    CHECK(sum_a == sum_b);
    pool.release_document(0);
    pool.release_document(1);
  }

  std::printf(failures ? "FAILED (%d)\n" : "PASS\n", failures);
  return failures ? 1 : 0;
}
