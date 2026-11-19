// hp::Engine lifecycle: warm-up, explicit unload, idle unload, reload on demand.
//
//   engine_lifecycle_test lifecycle <doc.pdf>
//       Asserts the state machine and prints device memory at each step.
//   engine_lifecycle_test latency <doc.pdf> cold|load|warmup
//       Fresh process: time the first parse after no preparation, after load(),
//       or after warmup(). Then time a second, steady-state parse. Run each mode in
//       its own process: CUDA lazy module loading is per process.
//   engine_lifecycle_test loadtimes [cycles]
//       load()/unload() cycles with per-model load times and unload split.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "engine.h"
#include "model_store.h"

namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int failures = 0;
#define CHECK(c)                                                         \
  do {                                                                   \
    if (!(c)) {                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                                        \
    }                                                                    \
  } while (0)

double device_used_mib() {
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  return static_cast<double>(total_b - free_b) / (1 << 20);
}

std::vector<uint8_t> read_file(const char *path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

hp::EngineConfig base_config() {
  hp::EngineConfig cfg;
  // As hyperparse: HP_MODELS, else the HF cache + engine cache. First call is before any Engine/thread.
  static const std::string dir = [] {
    const char *m = std::getenv("HP_MODELS");
    return hp::prepare_models(m ? m : "", {});
  }();
  cfg.models.dir = dir;
  return cfg;
}

struct Run {
  hp::ParseStats stats;
  std::vector<std::vector<hp::LayoutBox>> layout;     // per page
  std::vector<std::map<std::string, int>> words;      // per page: word multiset of all lines
  int ok = 0;
};

bool same_layout(const std::vector<hp::LayoutBox> &a, const std::vector<hp::LayoutBox> &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].cls != b[i].cls || a[i].x0 != b[i].x0 || a[i].y0 != b[i].y0 || a[i].x1 != b[i].x1 || a[i].y1 != b[i].y1)
      return false;
  return true;
}

// Word-multiset F1 between two parses of the same document.
double word_f1(const Run &a, const Run &b) {
  long tp = 0, na = 0, nb = 0;
  for (size_t p = 0; p < a.words.size() && p < b.words.size(); ++p) {
    for (const auto &[w, n] : a.words[p]) {
      na += n;
      if (auto it = b.words[p].find(w); it != b.words[p].end()) tp += std::min(n, it->second);
    }
    for (const auto &[w, n] : b.words[p]) nb += n;
  }
  return na + nb ? 2.0 * tp / (na + nb) : 1.0;
}

Run parse(hp::Engine &e, const std::vector<uint8_t> &pdf) {
  Run r;
  r.stats = e.parse(pdf, {}, [&](hp::PageResult &&p) {
    if (r.layout.size() <= static_cast<size_t>(p.page)) {
      r.layout.resize(p.page + 1);
      r.words.resize(p.page + 1);
    }
    r.layout[p.page] = p.layout;
    for (const auto &l : p.lines) {
      std::istringstream ws(l.text);
      for (std::string w; ws >> w;) ++r.words[p.page][w];
    }
    r.ok += p.ok;
  });
  return r;
}

int lifecycle(const std::vector<uint8_t> &pdf) {
  hp::EngineConfig cfg = base_config();
  cfg.idle_unload = milliseconds(1500);
  hp::Engine e(cfg);
  const double base = device_used_mib();  // CUDA context only
  std::printf("device memory, context only:   %7.0f MiB\n", base);
  CHECK(!e.loaded());

  // 1. parse() on an unloaded engine loads on demand.
  const Run cold = parse(e, pdf);
  CHECK(e.loaded());
  CHECK(cold.stats.load_ms > 0);
  CHECK(cold.ok == cold.stats.pages);
  const double loaded_mib = device_used_mib();
  std::printf("device memory, loaded:         %7.0f MiB  (cold parse: load %.0f ms + %d pages %.0f ms)\n", loaded_mib,
              cold.stats.load_ms, cold.stats.pages, cold.stats.wall_ms - cold.stats.load_ms);

  // 2. Idle unload after 1.5 s without activity.
  std::this_thread::sleep_for(milliseconds(2300));
  CHECK(!e.loaded());
  CHECK(e.lifecycle().idle_unloads == 1);
  const double idle_mib = device_used_mib();
  std::printf("device memory, idle-unloaded:  %7.0f MiB  (unload took %.0f ms)\n", idle_mib, e.lifecycle().last_unload_ms);
  CHECK(idle_mib < base + 64);  // everything but the context is back

  // 3. warmup() reloads; the next parse needs no load and matches the cold parse
  //    within run-to-run noise. The core det postprocess is not bit-reproducible:
  //    two parses in one loaded engine agree at word F1 0.998-0.999 (mix100.pdf),
  //    while layout is reproducible exactly.
  e.warmup();
  CHECK(e.loaded());
  const auto life = e.lifecycle();
  std::printf("warmup(): load %.0f ms + warm-up %.0f ms\n", life.last_load_ms, life.last_warmup_ms);
  const Run warm = parse(e, pdf);
  CHECK(warm.stats.load_ms == 0);
  int same = 0;
  for (size_t i = 0; i < warm.layout.size() && i < cold.layout.size(); ++i) same += same_layout(warm.layout[i], cold.layout[i]);
  const double f1 = word_f1(cold, warm);
  std::printf("warm parse: %d pages %.0f ms; vs cold parse: layout identical on %d/%zu pages, word F1 %.4f\n",
              warm.stats.pages, warm.stats.wall_ms, same, cold.layout.size(), f1);
  CHECK(warm.ok == warm.stats.pages);
  CHECK(same == static_cast<int>(cold.layout.size()));
  CHECK(f1 >= 0.995);

  // 4. Explicit unload.
  e.unload();
  CHECK(!e.loaded());
  CHECK(device_used_mib() < base + 64);
  CHECK(e.lifecycle().unloads == 2);

  // 5. A parse that outlasts a short idle timeout is never interrupted; the
  //    unload comes after it, once the timeout has elapsed from its end.
  e.set_idle_unload(milliseconds(100));
  const Run during = parse(e, pdf);
  CHECK(during.ok == during.stats.pages);
  std::this_thread::sleep_for(milliseconds(400));
  CHECK(!e.loaded());
  CHECK(e.lifecycle().idle_unloads == 2);

  // 6. Failures leave the engine usable: a non-PDF is rejected up front, and a
  //    sink that throws aborts the parse (outstanding pages drained) with its own
  //    exception.
  bool rejected = false;
  try {
    const std::vector<uint8_t> junk(1000, 'x');
    (void)e.parse(junk, {}, [](hp::PageResult &&) {});
  } catch (const hp::InvalidPdfError &) {
    rejected = true;
  }
  CHECK(rejected);
  // max_pages rejects before on_start and before any page is rendered.
  bool over_limit = false, started = false;
  try {
    hp::ParseOptions o;
    o.max_pages = 10;
    o.on_start = [&](int) { started = true; };
    (void)e.parse(pdf, o, [](hp::PageResult &&) {});
  } catch (const hp::PageLimitError &x) {
    over_limit = x.limit == 10 && x.pages > 10;
  }
  CHECK(over_limit && !started);
  // on_start sees the page count first; throwing from it aborts the parse.
  int announced = -1, pages_before_abort = 0;
  bool start_error = false;
  try {
    hp::ParseOptions o;
    o.on_start = [&](int n) {
      announced = n;
      throw std::runtime_error("start failure");
    };
    (void)e.parse(pdf, o, [&](hp::PageResult &&) { ++pages_before_abort; });
  } catch (const std::runtime_error &x) {
    start_error = std::string(x.what()) == "start failure";
  }
  CHECK(start_error && announced == cold.stats.pages && pages_before_abort == 0);
  bool sink_error = false;
  int delivered = 0;
  try {
    (void)e.parse(pdf, {}, [&](hp::PageResult &&) {
      if (++delivered == 3) throw std::runtime_error("sink failure");
    });
  } catch (const std::runtime_error &x) {
    sink_error = std::string(x.what()) == "sink failure";
  }
  CHECK(sink_error);
  const Run after_error = parse(e, pdf);
  std::printf("after errors: %d/%d pages ok\n", after_error.ok, after_error.stats.pages);
  CHECK(after_error.ok == after_error.stats.pages);

  // 7. Timeout 0 disables idle unloading.
  e.set_idle_unload(milliseconds(0));
  e.load();
  std::this_thread::sleep_for(milliseconds(400));
  CHECK(e.loaded());

  const auto l = e.lifecycle();
  std::printf("lifecycle: %d loads, %d unloads (%d idle)\n", l.loads, l.unloads, l.idle_unloads);
  std::printf(failures ? "FAILED (%d)\n" : "PASS\n", failures);
  return failures ? 1 : 0;
}

int latency(const std::vector<uint8_t> &pdf, const std::string &mode) {
  hp::Engine e(base_config());
  double prep_ms = 0;
  const auto t = Clock::now();
  if (mode == "load") e.load();
  else if (mode == "warmup") e.warmup();
  prep_ms = ms_since(t);
  const auto t1 = Clock::now();
  const Run first = parse(e, pdf);
  const double first_ms = ms_since(t1);
  const auto t2 = Clock::now();
  (void)parse(e, pdf);
  const double second_ms = ms_since(t2);
  std::printf("%-6s prep %6.0f ms | first parse %6.0f ms (of which load %4.0f) | second parse %5.0f ms\n", mode.c_str(),
              prep_ms, first_ms, first.stats.load_ms, second_ms);
  return 0;
}

// Load/unload timing: `cycles` load()+unload() pairs in one process. The first
// load also creates the CUDA context and loads CUDA modules; later ones show
// the steady cost of an idle-unload -> reload round trip.
int load_times(int cycles) {
  hp::Engine e(base_config());
  std::printf("%-5s %8s %8s %8s %8s %8s %6s %8s | %8s %8s %8s\n", "cycle", "layout", "det", "rec", "formula", "pin",
              "", "load", "free", "unpin", "unload");
  for (int i = 0; i < cycles; ++i) {
    const auto t = Clock::now();
    e.load();
    const double load_wall = ms_since(t);
    const auto t2 = Clock::now();
    e.unload();
    const double unload_wall = ms_since(t2);
    const auto l = e.lifecycle();
    std::printf("%-5d %8.0f %8.0f %8.0f %8.0f %8.0f %6s %8.0f | %8.0f %8.0f %8.0f\n", i + 1, l.last_load.layout_ms,
                l.last_load.det_ms, l.last_load.rec_ms, l.last_load.formula_ms, l.last_pin_ms, "", load_wall,
                l.last_unload_models_ms, l.last_unload_unpin_ms, unload_wall);
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc >= 3 && !std::strcmp(argv[1], "lifecycle")) return lifecycle(read_file(argv[2]));
  if (argc >= 4 && !std::strcmp(argv[1], "latency")) return latency(read_file(argv[2]), argv[3]);
  if (argc >= 2 && !std::strcmp(argv[1], "loadtimes")) return load_times(argc >= 3 ? std::atoi(argv[2]) : 5);
  std::fprintf(stderr, "usage: engine_lifecycle_test lifecycle <doc.pdf> | latency <doc.pdf> cold|load|warmup\n");
  return 2;
}
