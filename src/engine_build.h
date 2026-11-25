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

enum class ModelGroup { kLayout, kOcr, kFormula, kTables };

struct PlanSpec {
  std::string onnx;  // relative to the ONNX dir
  std::string plan;  // relative to the plan dir
  ModelGroup group;
  std::vector<Profile> profiles;
  double workspace_gb = 2.0;
};

// LayoutNet (table structure) is not in kModelRepo: it is OCRia-nano-1's tablenet.onnx, pinned
// to a commit of its Hugging Face repo and checked against this SHA-256 (model_store.cpp).
inline constexpr const char *kTableNetRepo = "llamaindex/OCRia-nano-1";
inline constexpr const char *kTableNetRevision = "53d45fe6803a568deedb081550c2fdb97cc99f97";
inline constexpr const char *kTableNetFile = "tablenet.onnx";
inline constexpr const char *kTableNetSha256 = "3b913866067d6e43f73a4e4c9c1eace955b7959447043cc39ea1d30e4c552a95";
inline constexpr const char *kTableNetPlan = "tablenet_3b913866.plan";

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
