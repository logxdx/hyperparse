#include "server_config.h"

#include "api_docs.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace hp::server {
namespace {

constexpr const char *kUsage =
    "usage: hyperparse_server [options]   (every --flag-name also reads env HP_FLAG_NAME)\n"
    "  --host ADDR            listen address (0.0.0.0)\n"
    "  --port N               listen port (8080)\n"
    "  --io-threads N         Drogon I/O threads (4)\n"
    "  --max-queue N          requests waiting for the engine before 503 (16)\n"
    "  --queue-timeout-ms N   answer 504 if a request waited longer (0 = off)\n"
    "  --max-body-mb N        largest accepted upload (100)\n"
    "  --max-pages N          largest accepted PDF in pages (2000; 0 = off)\n"
    "  --shutdown-grace N     seconds to finish running work on SIGTERM/SIGINT (30)\n"
    "  --idle-unload N        unload the models after N idle seconds (0 = never)\n"
    "  --no-warmup            don't load + warm up the models before listening\n"
    "  --models DIR           plans + dicts, or ONNX to build them from (default: HF cache + engine cache)\n"
    "  --workers N            page-worker processes (0 = auto)\n"
    "  --det-side N           text-det max side (1280)\n"
    "  --rec budget|bucket    recognizer batching (budget)\n"
    "  --no-ocr               don't load det/rec (text layer + layout only)\n"
    "  --no-formula           don't load the formula model\n"
    "  --no-tables            don't load LayoutNet (table cells from text geometry instead)\n";

int to_int(const std::string &k, const std::string &v) {
  char *end = nullptr;
  const long x = std::strtol(v.c_str(), &end, 10);
  if (v.empty() || *end != '\0' || x < 0 || x > 1'000'000'000) throw std::runtime_error(k + ": expected a number >= 0");
  return static_cast<int>(x);
}

// "--max-queue" -> "HP_MAX_QUEUE"
std::string env_name(const std::string &flag) {
  std::string e = "HP_";
  for (char c : flag.substr(2)) e += c == '-' ? '_' : static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return e;
}

const std::vector<std::string> kValueFlags = {
    "--host",       "--port",     "--io-threads", "--max-queue", "--queue-timeout-ms", "--max-body-mb", "--max-pages",
    "--shutdown-grace", "--idle-unload", "--models", "--workers",  "--det-side",         "--rec"};
const std::vector<std::string> kSwitches = {"--no-warmup", "--no-ocr", "--no-formula", "--no-tables"};

}  // namespace

ServerConfig parse_server_config(int argc, char **argv) {
  // Environment first, then argv, so flags override.
  std::vector<std::string> args;
  for (const auto &f : kValueFlags)
    if (const char *v = std::getenv(env_name(f).c_str())) args.insert(args.end(), {f, v});
  for (const auto &f : kSwitches)
    if (const char *v = std::getenv(env_name(f).c_str()); v && std::string(v) != "0") args.push_back(f);
  for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

  ServerConfig c;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &k = args[i];
    if (k == "-h" || k == "--help") {
      std::fputs((kUsage + endpoints_help()).c_str(), stdout);
      std::exit(0);
    }
    auto next = [&]() -> std::string {
      if (i + 1 >= args.size()) throw std::runtime_error("missing value for " + k + "\n" + kUsage);
      return args[++i];
    };
    if (k == "--host") c.host = next();
    else if (k == "--port") c.port = to_int(k, next());
    else if (k == "--io-threads") c.io_threads = std::max(1, to_int(k, next()));
    else if (k == "--max-queue") c.max_queue = to_int(k, next());
    else if (k == "--queue-timeout-ms") c.queue_timeout = std::chrono::milliseconds(to_int(k, next()));
    else if (k == "--max-body-mb") c.max_body_mb = std::max(1, to_int(k, next()));
    else if (k == "--max-pages") c.max_pages = to_int(k, next());
    else if (k == "--shutdown-grace") c.shutdown_grace_s = to_int(k, next());
    else if (k == "--idle-unload") c.engine.idle_unload = std::chrono::seconds(to_int(k, next()));
    else if (k == "--no-warmup") c.warmup = false;
    else if (k == "--models") c.engine.models.dir = next();
    else if (k == "--workers") c.engine.workers = to_int(k, next());
    else if (k == "--det-side") c.engine.models.det_max_side = to_int(k, next());
    else if (k == "--no-ocr") c.engine.models.ocr = false;
    else if (k == "--no-formula") c.engine.models.formula = false;
    else if (k == "--no-tables") c.engine.models.tables = false;
    else if (k == "--rec") {
      const std::string v = next();
      if (v != "budget" && v != "bucket") throw std::runtime_error("--rec budget|bucket");
      c.engine.models.rec = v == "bucket" ? RecMode::kWidthBucket : RecMode::kColumnBudget;
    } else {
      throw std::runtime_error("unknown option " + k + " (see hyperparse_server --help)");
    }
  }
  c.max_body_mem_mb = std::min(c.max_body_mem_mb, c.max_body_mb);
  return c;
}

}  // namespace hp::server
