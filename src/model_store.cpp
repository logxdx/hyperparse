#include "model_store.h"

#include <cuda_runtime.h>
#include <curl/curl.h>  // types and option constants only: libcurl is dlopen'ed when a download is needed
#include <dlfcn.h>
#include <fcntl.h>
#include <pwd.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine_build.h"

namespace hp {
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string env(const char *k) {
  const char *v = std::getenv(k);
  return v ? v : "";
}

std::string trim(std::string s) {
  const auto ws = " \t\r\n";
  s.erase(0, s.find_first_not_of(ws));
  s.erase(s.find_last_not_of(ws) + 1);
  return s;
}

// ---- locations ------------------------------------------------------------

fs::path home_dir() {
  if (std::string h = env("HOME"); !h.empty()) return h;
  if (const passwd *pw = getpwuid(getuid())) return pw->pw_dir;
  throw std::runtime_error("cannot find the home directory (HOME is unset)");
}
fs::path xdg_cache() {
  const std::string x = env("XDG_CACHE_HOME");
  return x.empty() ? home_dir() / ".cache" : fs::path(x);
}
// The same resolution as huggingface_hub, so `hf download` and hp share one cache.
fs::path hf_home() {
  const std::string h = env("HF_HOME");
  return h.empty() ? xdg_cache() / "huggingface" : fs::path(h);
}
fs::path hf_hub_cache() {
  const std::string c = env("HF_HUB_CACHE");
  return c.empty() ? hf_home() / "hub" : fs::path(c);
}
std::string hf_token() {
  if (std::string t = env("HF_TOKEN"); !t.empty()) return t;
  const std::string p = env("HF_TOKEN_PATH");
  std::ifstream in(p.empty() ? hf_home() / "token" : fs::path(p));
  std::string t;
  std::getline(in, t);
  return trim(t);
}
fs::path engine_cache() {
  const std::string c = env("HP_CACHE_DIR");
  return c.empty() ? xdg_cache() / "hyperparse" : fs::path(c);
}

bool needed(ModelGroup g, ModelNeeds n) {
  return g == ModelGroup::kLayout || (g == ModelGroup::kOcr && n.ocr) || (g == ModelGroup::kFormula && n.formula) ||
         (g == ModelGroup::kTables && n.tables) || (g == ModelGroup::kOrder && n.ordernet);
}

// A plan is needed when its group is, and (models with precision variants) it is the chosen one.
bool needed(const PlanSpec &s, const ModelNeeds &n) {
  if (!needed(s.group, n)) return false;
  if (s.variant.empty()) return true;
  if (s.group == ModelGroup::kOcr) return s.variant == (n.rec_probs ? "probs" : "logits");
  return s.variant == (s.group == ModelGroup::kTables ? n.tables_precision : n.order_precision);
}

bool plans_ready(const fs::path &dir, ModelNeeds n) {
  for (const PlanSpec &s : plan_specs())
    if (needed(s, n) && !fs::exists(dir / s.plan) && !(s.optional && !fs::exists(dir / s.onnx))) return false;
  for (const ModelAsset &a : model_assets())
    if (needed(a.group, n) && !fs::exists(dir / a.path)) return false;
  return true;
}

// ---- SHA-256 (download verification against manifest.json) ---------------

class Sha256 {
 public:
  void update(const uint8_t *p, size_t n) {
    total_ += n;
    while (n > 0) {
      const size_t k = std::min(n, size_t{64} - len_);
      std::memcpy(buf_ + len_, p, k);
      len_ += k, p += k, n -= k;
      if (len_ == 64) block(buf_), len_ = 0;
    }
  }
  std::string hex() {
    const uint64_t bits = total_ * 8;
    const uint8_t one = 0x80, zero = 0;
    update(&one, 1);
    while (len_ != 56) update(&zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    update(len, 8);
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h_[i]);
    return out;
  }

 private:
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const uint8_t *p) {
    static constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h_[0] += a, h_[1] += b, h_[2] += c, h_[3] += d, h_[4] += e, h_[5] += f, h_[6] += g, h_[7] += h;
  }
  uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buf_[64];
  size_t len_ = 0;
  uint64_t total_ = 0;
};

std::string sha256_file(const fs::path &p) {
  std::ifstream in(p, std::ios::binary);
  std::vector<uint8_t> buf(1 << 20);
  Sha256 h;
  while (in) {
    in.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
    h.update(buf.data(), static_cast<size_t>(in.gcount()));
  }
  return h.hex();
}

// ---- HTTP over the system libcurl, loaded only when downloading -----------

class Http {
 public:
  struct Reply {
    long status = 0;
    std::map<std::string, std::string> headers;  // final response only, lower-case names
    std::string body;
  };

  explicit Http(std::string token) : token_(std::move(token)) {
    lib_ = dlopen("libcurl.so.4", RTLD_NOW | RTLD_LOCAL);
    if (!lib_)
      throw std::runtime_error(std::string("downloading the models needs libcurl (") + dlerror() +
                               "); install curl, or fetch them with `hf download " + kModelRepo + "`");
    sym(global_init_, "curl_global_init");
    sym(init_, "curl_easy_init");
    sym(setopt_, "curl_easy_setopt");
    sym(perform_, "curl_easy_perform");
    sym(getinfo_, "curl_easy_getinfo");
    sym(cleanup_, "curl_easy_cleanup");
    sym(strerror_, "curl_easy_strerror");
    sym(slist_append_, "curl_slist_append");
    sym(slist_free_, "curl_slist_free_all");
    global_init_(CURL_GLOBAL_DEFAULT);
  }

  // GET (or HEAD) url. With `sink`, the body goes there instead of Reply::body.
  Reply fetch(const std::string &url, bool head, bool follow, std::FILE *sink = nullptr) {
    Reply r;
    Sink s{&r, sink};
    CURL *c = init_();
    curl_slist *hdrs = slist_append_(nullptr, "User-Agent: hyperparse");
    // libcurl drops this header when a redirect leaves huggingface.co (the CDN URL is pre-signed).
    if (!token_.empty()) hdrs = slist_append_(hdrs, ("Authorization: Bearer " + token_).c_str());
    setopt_(c, CURLOPT_URL, url.c_str());
    setopt_(c, CURLOPT_HTTPHEADER, hdrs);
    setopt_(c, CURLOPT_FOLLOWLOCATION, follow ? 1L : 0L);
    setopt_(c, CURLOPT_NOBODY, head ? 1L : 0L);
    setopt_(c, CURLOPT_CONNECTTIMEOUT, 30L);
    setopt_(c, CURLOPT_LOW_SPEED_LIMIT, 1024L);  // abort a stalled transfer after 60 s
    setopt_(c, CURLOPT_LOW_SPEED_TIME, 60L);
    setopt_(c, CURLOPT_HEADERFUNCTION, &Http::on_header);
    setopt_(c, CURLOPT_HEADERDATA, &s);
    setopt_(c, CURLOPT_WRITEFUNCTION, &Http::on_body);
    setopt_(c, CURLOPT_WRITEDATA, &s);
    const CURLcode rc = perform_(c);
    getinfo_(c, CURLINFO_RESPONSE_CODE, &r.status);
    cleanup_(c);
    slist_free_(hdrs);
    if (rc != CURLE_OK) throw std::runtime_error(url + ": " + strerror_(rc));
    return r;
  }

 private:
  struct Sink {
    Reply *reply;
    std::FILE *file;
  };
  static size_t on_header(char *b, size_t size, size_t n, void *ud) {
    auto *s = static_cast<Sink *>(ud);
    const std::string line(b, size * n);
    if (line.rfind("HTTP/", 0) == 0) s->reply->headers.clear();  // a new response (redirect hop)
    if (const size_t colon = line.find(':'); colon != std::string::npos) {
      std::string k = line.substr(0, colon);
      for (char &ch : k) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      s->reply->headers[k] = trim(line.substr(colon + 1));
    }
    return size * n;
  }
  static size_t on_body(char *b, size_t size, size_t n, void *ud) {
    auto *s = static_cast<Sink *>(ud);
    if (s->file) return std::fwrite(b, size, n, s->file) * size;
    s->reply->body.append(b, size * n);
    return size * n;
  }
  template <class F>
  void sym(F &f, const char *name) {
    f = reinterpret_cast<F>(dlsym(lib_, name));
    if (!f) throw std::runtime_error(std::string("libcurl lacks ") + name);
  }

  std::string token_;
  void *lib_ = nullptr;
  CURLcode (*global_init_)(long) = nullptr;
  CURL *(*init_)() = nullptr;
  CURLcode (*setopt_)(CURL *, CURLoption, ...) = nullptr;
  CURLcode (*perform_)(CURL *) = nullptr;
  CURLcode (*getinfo_)(CURL *, CURLINFO, ...) = nullptr;
  void (*cleanup_)(CURL *) = nullptr;
  const char *(*strerror_)(CURLcode) = nullptr;
  curl_slist *(*slist_append_)(curl_slist *, const char *) = nullptr;
  void (*slist_free_)(curl_slist *) = nullptr;
};

// ---- Hugging Face cache (huggingface_hub layout) --------------------------
//   models--<org>--<name>/refs/<revision>          commit sha
//   models--<org>--<name>/blobs/<etag>             file contents
//   models--<org>--<name>/snapshots/<sha>/<path>   -> ../../blobs/<etag>

fs::path repo_cache_dir() {
  std::string name = kModelRepo;
  for (size_t p; (p = name.find('/')) != std::string::npos;) name.replace(p, 1, "--");
  return hf_hub_cache() / ("models--" + name);
}

json read_manifest(const fs::path &snap) {
  std::ifstream in(snap / "manifest.json");
  if (!in) return json();
  return json::parse(in, nullptr, false);
}

// A snapshot is usable when its manifest and every file it lists are present.
std::optional<fs::path> local_snapshot(const fs::path &repo, const std::string &rev) {
  std::ifstream ref(repo / "refs" / rev);
  std::string sha;
  std::getline(ref, sha);
  sha = trim(sha);
  if (sha.empty() && fs::exists(repo / "snapshots" / rev)) sha = rev;  // revision given as a commit
  if (sha.empty()) return std::nullopt;
  const fs::path snap = repo / "snapshots" / sha;
  const json m = read_manifest(snap);
  if (!m.is_object() || !m.contains("files")) return std::nullopt;
  for (const json &f : m["files"])
    if (!fs::exists(snap / f.at("path").get<std::string>())) return std::nullopt;
  return snap;
}

// Whether the snapshot's manifest lists every model `needs` asks for (a snapshot cached before a
// model was added to the repo, e.g. tablenet.onnx, does not).
bool snapshot_covers(const fs::path &snap, ModelNeeds needs) {
  const json m = read_manifest(snap);
  if (!m.is_object() || !m.contains("files")) return false;
  auto listed = [&](const std::string &path) {
    for (const json &f : m["files"])
      if (f.at("path") == path) return true;
    return false;
  };
  for (const PlanSpec &s : plan_specs())
    // An optional model (doc_ori.onnx) counts too when online, so a snapshot cached before it was
    // added is fetched again; offline, such a snapshot runs without it.
    if (needed(s, needs) && (!s.optional || env("HF_HUB_OFFLINE") != "1") && !listed(s.onnx)) return false;
  for (const ModelAsset &a : model_assets())
    if (needed(a.group, needs) && !listed(a.path)) return false;
  return true;
}

std::string normalize_etag(std::string e) {
  if (e.rfind("W/", 0) == 0) e = e.substr(2);
  if (e.size() >= 2 && e.front() == '"' && e.back() == '"') e = e.substr(1, e.size() - 2);
  return e;
}

fs::path download_snapshot(const fs::path &repo, const std::string &rev) {
  Http http(hf_token());
  const std::string hub = "https://huggingface.co";
  const Http::Reply info = http.fetch(hub + "/api/models/" + kModelRepo + "/revision/" + rev, false, true);
  if (info.status == 401 || info.status == 403 || info.status == 404)
    throw std::runtime_error(std::string("cannot read ") + kModelRepo + " (HTTP " + std::to_string(info.status) +
                             "): a private repo needs HF_TOKEN or `hf auth login`");
  if (info.status != 200) throw std::runtime_error("Hugging Face API: HTTP " + std::to_string(info.status));
  const json j = json::parse(info.body);
  const std::string sha = j.at("sha");
  std::vector<std::string> files = {"manifest.json"};  // first: it verifies the others
  for (const json &s : j.at("siblings"))
    if (const std::string f = s.at("rfilename"); f != "manifest.json") files.push_back(f);

  const fs::path snap = repo / "snapshots" / sha;
  std::fprintf(stderr, "downloading %s@%.12s into %s\n", kModelRepo, sha.c_str(), repo.c_str());
  json manifest;
  for (const std::string &f : files) {
    const fs::path link = snap / f;
    if (!fs::exists(link)) {
      const std::string url = hub + "/" + kModelRepo + "/resolve/" + sha + "/" + f;
      Http::Reply h = http.fetch(url, true, false);
      if (h.status >= 400) throw std::runtime_error(f + ": HTTP " + std::to_string(h.status));
      std::string etag = normalize_etag(h.headers.count("x-linked-etag") ? h.headers["x-linked-etag"] : h.headers["etag"]);
      if (etag.empty() || etag.find('/') != std::string::npos) throw std::runtime_error(f + ": no usable ETag");
      const fs::path blob = repo / "blobs" / etag;
      if (!fs::exists(blob)) {
        fs::create_directories(blob.parent_path());
        const std::string size = h.headers.count("x-linked-size") ? h.headers["x-linked-size"] : h.headers["content-length"];
        std::fprintf(stderr, "  %s (%.1f MB)\n", f.c_str(), size.empty() ? 0.0 : std::stod(size) / 1e6);
        const fs::path tmp = blob.string() + ".incomplete";
        std::FILE *out = std::fopen(tmp.c_str(), "wb");
        if (!out) throw std::runtime_error("cannot write " + tmp.string());
        const Http::Reply r = http.fetch(url, false, true, out);
        if (std::fclose(out) != 0 || r.status != 200) {
          fs::remove(tmp);
          throw std::runtime_error(f + ": download failed (HTTP " + std::to_string(r.status) + ")");
        }
        if (manifest.is_object())
          for (const json &e : manifest["files"])
            if (e.at("path") == f && sha256_file(tmp) != e.at("sha256").get<std::string>()) {
              fs::remove(tmp);
              throw std::runtime_error(f + ": SHA-256 does not match manifest.json");
            }
        fs::rename(tmp, blob);
      }
      fs::create_directories(link.parent_path());
      std::string target;  // one "../" per path component, then out of snapshots/<sha>
      const fs::path rel(f);
      for (auto it = rel.begin(); it != rel.end(); ++it) target += "../";
      fs::create_symlink(target + "../blobs/" + etag, link);  // relative, as huggingface_hub does
    }
    if (f == "manifest.json") manifest = read_manifest(snap);
  }
  fs::create_directories(repo / "refs");
  std::ofstream(repo / "refs" / rev) << sha;
  return snap;
}

// ---- engines -------------------------------------------------------------

// Builds the missing plans of `needs` from onnx_dir into plan_dir and copies the
// assets across. A lock file serialises concurrent hp processes.
// What a plan was built from: its ONNX (SHA-256, from the manifest when there is one) and its
// optimization profiles. Stored next to the plan as <plan>.src.
std::string plan_source_key(const PlanSpec &s, const fs::path &onnx_dir) {
  std::string sha;
  const json m = read_manifest(onnx_dir);
  if (m.is_object() && m.contains("files"))
    for (const json &f : m["files"])
      if (f.at("path") == s.onnx) sha = f.at("sha256").get<std::string>();
  if (sha.empty() && fs::exists(onnx_dir / s.onnx)) sha = sha256_file(onnx_dir / s.onnx);
  std::string key = sha + " ws=" + std::to_string(s.workspace_gb) + (s.logits_output ? " logits" : "");
  if (s.host_shapes) key += " host_shapes";
  for (const Profile &p : s.profiles) {
    key += " |";
    for (const InputRange &r : p) {
      key += " " + r.name;
      for (const auto *v : {&r.min, &r.opt, &r.max}) {
        key += ":";
        for (int64_t d : *v) key += std::to_string(d) + "x";
      }
    }
  }
  return key;
}

std::string read_text(const fs::path &p) {
  std::ifstream in(p);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// A plan built from the same ONNX and profiles for this GPU and TensorRT in another engine-cache
// folder (the cache is keyed by the whole manifest, so adding one model to the repo starts a new
// folder): copy it instead of rebuilding.
bool reuse_plan(const PlanSpec &s, const fs::path &plan_dir, const std::string &key) {
  const std::string name = plan_dir.filename().string();
  const size_t dash = name.find('-');
  if (dash == std::string::npos || !fs::is_directory(plan_dir.parent_path())) return false;
  const std::string suffix = name.substr(dash);  // -sm<cc>-trt<version>
  for (const auto &e : fs::directory_iterator(plan_dir.parent_path())) {
    const std::string other = e.path().filename().string();
    if (e.path() == plan_dir || other.size() <= suffix.size() ||
        other.compare(other.size() - suffix.size(), suffix.size(), suffix) != 0)
      continue;
    const fs::path src = e.path() / s.plan;
    if (!fs::exists(src) || read_text(src.string() + ".src") != key) continue;
    fs::create_directories((plan_dir / s.plan).parent_path());
    const fs::path tmp = (plan_dir / s.plan).string() + ".tmp" + std::to_string(::getpid());
    fs::copy_file(src, tmp, fs::copy_options::overwrite_existing);
    fs::rename(tmp, plan_dir / s.plan);
    std::ofstream(plan_dir / (s.plan + ".src")) << key;
    return true;
  }
  return false;
}

void ensure_engines(const fs::path &onnx_dir, const fs::path &plan_dir, ModelNeeds needs) {
  fs::create_directories(plan_dir);
  const fs::path lock_path = plan_dir / ".build.lock";
  const int lock = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
  if (lock < 0) throw std::runtime_error("cannot create " + lock_path.string());
  if (::flock(lock, LOCK_EX | LOCK_NB) != 0) {
    std::fprintf(stderr, "waiting for another hp process building engines in %s\n", plan_dir.c_str());
    ::flock(lock, LOCK_EX);
  }
  std::vector<const PlanSpec *> todo;
  // Missing plans, and plans built from another ONNX or other profiles than hp's (per their .src).
  auto stale = [&](const PlanSpec &s) {
    const fs::path src = plan_dir / (s.plan + ".src");
    return fs::exists(src) && read_text(src) != plan_source_key(s, onnx_dir);
  };
  for (const PlanSpec &s : plan_specs())
    if (needed(s, needs) && (!fs::exists(plan_dir / s.plan) || stale(s)) && !(s.optional && !fs::exists(onnx_dir / s.onnx)))
      todo.push_back(&s);
  if (!todo.empty())
    std::fprintf(stderr, "preparing %zu TensorRT engine(s) with TensorRT %s in %s (one-time; minutes)\n", todo.size(),
                 trt_version().c_str(), plan_dir.c_str());
  for (size_t i = 0; i < todo.size(); ++i) {
    std::fprintf(stderr, "  [%zu/%zu] %s ...", i + 1, todo.size(), todo[i]->plan.c_str());
    const std::string key = plan_source_key(*todo[i], onnx_dir);
    if (reuse_plan(*todo[i], plan_dir, key)) {
      std::fprintf(stderr, " unchanged, copied from another engine cache\n");
      continue;
    }
    const auto t = std::chrono::steady_clock::now();
    build_plan(*todo[i], onnx_dir.string(), plan_dir.string());
    std::ofstream(plan_dir / (todo[i]->plan + ".src")) << key;
    std::fprintf(stderr, " %.0f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count());
  }
  for (const ModelAsset &a : model_assets()) {
    if (!needed(a.group, needs) || fs::exists(plan_dir / a.path)) continue;
    if (!fs::exists(onnx_dir / a.path)) throw std::runtime_error("missing " + (onnx_dir / a.path).string());
    fs::create_directories((plan_dir / a.path).parent_path());
    fs::copy_file(onnx_dir / a.path, plan_dir / a.path);
  }
  ::close(lock);
}

std::string gpu_key() {
  int dev = 0;
  cudaDeviceProp p{};
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess)
    throw std::runtime_error("no usable CUDA GPU");
  return "sm" + std::to_string(p.major) + std::to_string(p.minor);
}

// Runs fn in a forked child (CUDA, TensorRT and libcurl stay out of the caller)
// and returns its string result.
std::string run_in_child(const std::function<std::string()> &fn) {
  int fds[2];
  if (::pipe(fds) != 0) throw std::runtime_error("pipe failed");
  std::fflush(nullptr);
  const pid_t pid = ::fork();
  if (pid < 0) throw std::runtime_error("fork failed");
  if (pid == 0) {
    ::close(fds[0]);
    std::string msg;
    int code = 0;
    try {
      msg = "ok\n" + fn();
    } catch (const std::exception &e) {
      msg = std::string("error\n") + e.what();
      code = 1;
    }
    for (size_t off = 0; off < msg.size();) {
      const ssize_t w = ::write(fds[1], msg.data() + off, msg.size() - off);
      if (w <= 0) break;
      off += static_cast<size_t>(w);
    }
    std::fflush(nullptr);
    ::_exit(code);
  }
  ::close(fds[1]);
  std::string msg;
  char buf[4096];
  for (ssize_t r; (r = ::read(fds[0], buf, sizeof buf)) > 0;) msg.append(buf, static_cast<size_t>(r));
  ::close(fds[0]);
  int status = 0;
  ::waitpid(pid, &status, 0);
  if (WIFSIGNALED(status)) throw std::runtime_error("model preparation crashed (signal " + std::to_string(WTERMSIG(status)) + ")");
  const size_t nl = msg.find('\n');
  if (nl == std::string::npos) throw std::runtime_error("model preparation failed");
  if (msg.compare(0, nl, "ok") != 0) throw std::runtime_error(msg.substr(nl + 1));
  return msg.substr(nl + 1);
}

}  // namespace

std::string prepare_models(const std::string &dir, ModelNeeds needs) {
  if (!dir.empty()) {
    if (plans_ready(dir, needs)) return dir;
    return run_in_child([&] {
      ensure_engines(dir, dir, needs);
      return dir;
    });
  }
  return run_in_child([&] {
    const std::string rev = env("HP_MODEL_REVISION").empty() ? "main" : env("HP_MODEL_REVISION");
    const fs::path repo = repo_cache_dir();
    std::optional<fs::path> snap = local_snapshot(repo, rev);
    if (snap && !snapshot_covers(*snap, needs)) {
      // cached from before a model was added to the repo: fetch the revision again
      if (env("HF_HUB_OFFLINE") == "1")
        throw std::runtime_error("the cached " + std::string(kModelRepo) + " (" + snap->string() +
                                 ") lacks a model this hp needs, and HF_HUB_OFFLINE=1");
      snap.reset();
    }
    if (!snap) {
      if (env("HF_HUB_OFFLINE") == "1")
        throw std::runtime_error(std::string(kModelRepo) + " is not in " + hf_hub_cache().string() + " and HF_HUB_OFFLINE=1");
      snap = download_snapshot(repo, rev);
      if (!snapshot_covers(*snap, needs))
        throw std::runtime_error(std::string(kModelRepo) + "@" + rev + " lacks a model this hp needs (see its manifest.json)");
    }
    // Keyed by the manifest (model files + SHA-256), not the commit: a README
    // change must not trigger a rebuild, a model change must.
    const std::string models_id = sha256_file(*snap / "manifest.json").substr(0, 12);
    const fs::path plans = engine_cache() / "engines" / (models_id + "-" + gpu_key() + "-trt" + trt_version());
    ensure_engines(*snap, plans, needs);
    return plans.string();
  });
}

}  // namespace hp
