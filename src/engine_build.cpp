#include "engine_build.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <unistd.h>

namespace hp {
namespace fs = std::filesystem;

namespace {

class BuildLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kERROR) std::fprintf(stderr, "[TRT] %s\n", msg);
  }
};

nvinfer1::Dims to_dims(const std::vector<int64_t> &v) {
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int32_t>(v.size());
  for (size_t i = 0; i < v.size(); ++i) d.d[i] = v[i];
  return d;
}

InputRange fixed(const std::string &name, std::vector<int64_t> shape) { return {name, shape, shape, shape}; }

// torch's export of expand() sets one entry of a shape vector with ScatterElements(shape, [k],
// [v]). TensorRT has no shape-tensor scatter, so it treats the expand as data-dependent: each
// enqueueV3 computes the vector on the GPU, copies it back and synchronizes (the formula step:
// 4 of them per step, every step waiting for the GPU, and behind other streams' kernels when the
// GPU is shared). The same vector as shape[:k] ++ v ++ shape[k+1:] stays on the host. Only
// index arithmetic changes: the formula step's tokens are the same (arxiv100: 321/321).
void fold_shape_scatters(nvinfer1::INetworkDefinition &net) {
  for (int l = 0; l < net.getNbLayers(); ++l) {
    nvinfer1::ILayer *layer = net.getLayer(l);
    if (layer->getType() != nvinfer1::LayerType::kSCATTER) continue;
    auto *sc = static_cast<nvinfer1::IScatterLayer *>(layer);
    nvinfer1::ITensor *data = sc->getInput(0), *idx = sc->getInput(1), *upd = sc->getInput(2), *out = sc->getOutput(0);
    if (sc->getMode() != nvinfer1::ScatterMode::kELEMENT || !out->isShapeTensor() || data->getDimensions().nbDims != 1 ||
        idx->getDimensions().nbDims != 1 || idx->getDimensions().d[0] != 1 || upd->getDimensions().nbDims != 1)
      continue;
    const int64_t n = data->getDimensions().d[0];
    int64_t k = -1;
    for (int c = 0; c < net.getNbLayers(); ++c) {
      nvinfer1::ILayer *cl = net.getLayer(c);
      if (cl->getType() != nvinfer1::LayerType::kCONSTANT || cl->getOutput(0) != idx) continue;
      const nvinfer1::Weights w = static_cast<nvinfer1::IConstantLayer *>(cl)->getWeights();
      if (w.count == 1 && w.type == nvinfer1::DataType::kINT64) k = *static_cast<const int64_t *>(w.values);
      if (w.count == 1 && w.type == nvinfer1::DataType::kINT32) k = *static_cast<const int32_t *>(w.values);
    }
    if (k < 0) k += n;
    if (n <= 0 || k < 0 || k >= n) continue;
    std::vector<nvinfer1::ITensor *> parts;
    auto dims1 = [](int64_t v) { nvinfer1::Dims d{}; d.nbDims = 1; d.d[0] = v; return d; };
    if (k > 0) parts.push_back(net.addSlice(*data, dims1(0), dims1(k), dims1(1))->getOutput(0));
    parts.push_back(upd);
    if (k + 1 < n) parts.push_back(net.addSlice(*data, dims1(k + 1), dims1(n - k - 1), dims1(1))->getOutput(0));
    nvinfer1::IConcatenationLayer *cat = net.addConcatenation(parts.data(), static_cast<int32_t>(parts.size()));
    cat->setAxis(0);
    nvinfer1::ITensor *repl = cat->getOutput(0);
    for (int u = 0; u < net.getNbLayers(); ++u) {
      nvinfer1::ILayer *user = net.getLayer(u);
      if (user == cat) continue;
      for (int i = 0; i < user->getNbInputs(); ++i)
        if (user->getInput(i) == out) user->setInput(i, *repl);
    }
  }
}

std::vector<PlanSpec> make_specs() {
  std::vector<PlanSpec> s;
  // PP-OCR det: dynamic up to 1600 (multi_resize emits <= 1280 by default; 1600 leaves room).
  s.push_back({"det_small_fp16.onnx", "det_small_fp16.plan", ModelGroup::kOcr,
               {{{"x", {1, 3, 32, 32}, {1, 3, 1280, 992}, {2, 3, 1600, 1600}}}}});
  // PP-OCR rec: profile 0 dynamic, then one fixed profile per kRecGraphProfiles entry, in order.
  PlanSpec rec{"rec_small_fp16.onnx", "rec_small_fp16.plan", ModelGroup::kOcr,
               {{{"x", {1, 3, 48, 48}, {32, 3, 48, 320}, {32, 3, 48, 4000}}}}};
  const int64_t graphs[][2] = {{8, 320}, {16, 320}, {24, 320}, {32, 320}, {8, 480}, {16, 480},
                               {24, 480}, {32, 480}, {8, 800}, {16, 800}, {8, 1600}, {8, 4000}};
  for (const auto &g : graphs) rec.profiles.push_back({fixed("x", {g[0], 3, 48, g[1]})});
  s.push_back(rec);
  // D-FINE layout: batch 1 (fastest here, smallest arena), fixed 960 letterbox.
  s.push_back({"dfine_960_fp16.onnx", "dfine_960_fp16.plan", ModelGroup::kLayout,
               {{fixed("images", {1, 3, 960, 960})}}});
  // PP-FormulaNet-S: FP32 encoder + prep, FP16 decoder step (see src/formula.cu), batch <= 8.
  const int64_t B = 8;
  s.push_back({"formula/encoder.onnx", "formula/encoder.plan", ModelGroup::kFormula,
               {{{"x", {1, 1, 384, 384}, {1, 1, 384, 384}, {B, 1, 384, 384}}}}, 3.0});
  s.push_back({"formula/prep.onnx", "formula/prep.plan", ModelGroup::kFormula,
               {{{"memory", {1, 144, 2048}, {B, 144, 2048}, {B, 144, 2048}}}}});
  auto kv = [](int64_t b) { return std::vector<int64_t>{2, b, 16, 1056, 24}; };
  auto ckv = [](int64_t b) { return std::vector<int64_t>{2, b, 16, 144, 24}; };
  s.push_back({"formula/step_batched_fp16.onnx", "formula/step_fp16.plan", ModelGroup::kFormula,
               {{{"tokens", {1, 3}, {B, 3}, {B, 3}},
                 fixed("pos", {1}),
                 {"kb", kv(1), kv(B), kv(B)},
                 {"vb", kv(1), kv(B), kv(B)},
                 {"ck", ckv(1), ckv(B), ckv(B)},
                 {"cv", ckv(1), ckv(B), ckv(B)}}},
               3.0, {}, false, true});
  // LayoutNet, FP16 (autocast; attention masks -3e4 instead of -1e9 so they fit) or INT8 (Q/DQ,
  // rest FP16). Units 2..1500 (pages with more skip the model, as the reference does).
  // OmniDocBench table pages, CPU reference: TEDS FP32 0.511, FP16 0.511, INT8 0.498.
  auto units = [](int64_t n) { return std::vector<int64_t>{1, n}; };
  for (const char *v : {"fp16", "int8"})
    s.push_back({std::string("tablenet_") + v + ".onnx", table_plan(v), ModelGroup::kTables,
                 {{{"feats", {1, 2, 68}, {1, 256, 68}, {1, 1500, 68}},
                   {"mask", units(2), units(256), units(1500)},
                   {"pairf", {1, 2, 2, 2}, {1, 256, 256, 2}, {1, 1500, 1500, 2}},
                   fixed("image", {1, 1, 1024, 1024}),
                   {"boxes", {1, 2, 4}, {1, 256, 4}, {1, 1500, 4}},
                   fixed("valid", {1, 2})}},
                 2.0, v});
  // OrderNet, FP16 or INT8 (both scored like FP32 on olmOCR-bench / OmniDocBench order), lines
  // 2..2048 (longer pages keep XY-cut).
  for (const char *v : {"fp16", "int8"})
    s.push_back({std::string("order_") + v + ".onnx", order_plan(v), ModelGroup::kOrder,
                 {{{"feats", {1, 2, 16}, {1, 256, 16}, {1, 2048, 16}}, {"mask", {1, 2}, {1, 256}, {1, 2048}}}},
                 2.0, v});
  // Page orientation (PP-LCNet_x1_0_doc_ori, FP32, fixed 1x3x224x224; orient.h). Optional: only
  // when doc_ori.onnx is in the models dir.
  PlanSpec ori{"doc_ori.onnx", "doc_ori.plan", ModelGroup::kOcr, {{fixed("x", {1, 3, 224, 224})}}, 1.0};
  ori.optional = true;
  s.push_back(std::move(ori));
  return s;
}

}  // namespace

const std::vector<PlanSpec> &plan_specs() {
  static const std::vector<PlanSpec> specs = make_specs();
  return specs;
}

const std::vector<ModelAsset> &model_assets() {
  static const std::vector<ModelAsset> assets = {{"keys_small.txt", ModelGroup::kOcr},
                                                 {"formula/tokenizer.json", ModelGroup::kFormula}};
  return assets;
}

std::string trt_version() {
  const int v = getInferLibVersion();  // major * 10000 + minor * 100 + patch
  return std::to_string(v / 10000) + "." + std::to_string(v / 100 % 100) + "." + std::to_string(v % 100);
}

void build_plan(const PlanSpec &spec, const std::string &onnx_dir, const std::string &plan_dir) {
  const std::string onnx = onnx_dir + "/" + spec.onnx;
  if (!fs::exists(onnx)) throw std::runtime_error("missing ONNX model: " + onnx);
  BuildLogger logger;
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) throw std::runtime_error("TensorRT: cannot create a builder");
  std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));  // strongly typed (TRT 11)
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, logger));
  if (!parser->parseFromFile(onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
    std::string err = "cannot parse " + onnx;
    for (int i = 0; i < parser->getNbErrors(); ++i) err += std::string("\n  ") + parser->getError(i)->desc();
    throw std::runtime_error(err);
  }
  if (spec.host_shapes) fold_shape_scatters(*net);
  std::unique_ptr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
  cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<size_t>(spec.workspace_gb * (1ull << 30)));
  for (const Profile &p : spec.profiles) {
    nvinfer1::IOptimizationProfile *prof = builder->createOptimizationProfile();
    for (const InputRange &in : p) {
      using S = nvinfer1::OptProfileSelector;
      if (!prof->setDimensions(in.name.c_str(), S::kMIN, to_dims(in.min)) ||
          !prof->setDimensions(in.name.c_str(), S::kOPT, to_dims(in.opt)) ||
          !prof->setDimensions(in.name.c_str(), S::kMAX, to_dims(in.max)))
        throw std::runtime_error("bad profile for input " + in.name + " of " + spec.onnx);
    }
    cfg->addOptimizationProfile(prof);
  }
  std::unique_ptr<nvinfer1::IHostMemory> blob(builder->buildSerializedNetwork(*net, *cfg));
  if (!blob) throw std::runtime_error("TensorRT build failed: " + spec.onnx);

  const fs::path out = fs::path(plan_dir) / spec.plan;
  fs::create_directories(out.parent_path());
  const fs::path tmp = out.string() + ".tmp" + std::to_string(::getpid());
  {
    std::ofstream f(tmp, std::ios::binary);
    f.write(static_cast<const char *>(blob->data()), static_cast<std::streamsize>(blob->size()));
    if (!f) throw std::runtime_error("cannot write " + tmp.string());
  }
  fs::rename(tmp, out);
}

}  // namespace hp
