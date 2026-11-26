#pragma once
// Builds hp's TensorRT engines from the ONNX models with the TensorRT that hp
// links, so a plan always matches the runtime that loads it. Profiles mirror
// the rec graph sizes in core/recognition (PaddleRec bakes one CUDA graph per
// fixed rec profile).

#include <cstdint>
#include <string>
#include <vector>

namespace hp {

struct InputRange {
  std::string name;
  std::vector<int64_t> min, opt, max;
};
using Profile = std::vector<InputRange>;

enum class ModelGroup { kLayout, kOcr, kFormula, kTables, kOrder };

struct PlanSpec {
  std::string onnx;  // relative to the ONNX dir
  std::string plan;  // relative to the plan dir
  ModelGroup group;
  std::vector<Profile> profiles;
  double workspace_gb = 2.0;
  std::string variant = {};  // precision variant ("fp16", "int8") for models that have several; "" = the only one
  // Built and loaded only when its ONNX is in the models dir (a model the HF repo may not carry yet).
  bool optional = false;
};

// LayoutNet (OCRia-nano-1's tablenet) and OrderNet (Indra OCR's order) come in reduced-precision
// variants: tablenet_<v>.onnx / order_<v>.onnx -> tablenet_<v>.plan / order_<v>.plan, v = fp16 | int8.
inline std::string table_plan(const std::string &variant) { return "tablenet_" + variant + ".plan"; }
inline std::string order_plan(const std::string &variant) { return "order_" + variant + ".plan"; }

// Every engine hp loads, in build order.
const std::vector<PlanSpec> &plan_specs();

// Non-plan files the models dir must also hold, per group.
struct ModelAsset {
  std::string path;
  ModelGroup group;
};
const std::vector<ModelAsset> &model_assets();

// Builds onnx_dir/spec.onnx into plan_dir/spec.plan (written to a temp file and
// renamed, so a crash never leaves a truncated plan). Throws std::runtime_error.
void build_plan(const PlanSpec &spec, const std::string &onnx_dir, const std::string &plan_dir);

// Version of the loaded libnvinfer, e.g. "11.5.0".
std::string trt_version();

}  // namespace hp
