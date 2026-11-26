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
               3.0});
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
