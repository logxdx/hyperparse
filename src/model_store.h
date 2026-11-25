#pragma once
// Where hp's models come from. By default: the ONNX models of the Hugging Face
// repo kModelRepo, read from the standard Hugging Face cache (downloaded into
// it when missing), and TensorRT engines built from them on first use into
// hp's engine cache. Nothing is read from the source tree.

#include <string>

namespace hp {

inline constexpr const char *kModelRepo = "logxdx/hyperparse-models";

struct ModelNeeds {
  bool ocr = true;
  bool formula = true;
  bool tables = true;  // LayoutNet (tablenet.onnx)
};

// Returns a directory for ModelConfig::dir holding every plan and asset `needs`
// asks for, downloading and building what is missing.
//   dir set:   a models dir with the plans, or with the ONNX to build them from
//              (built in place).
//   dir empty: kModelRepo from $HF_HUB_CACHE, else $HF_HOME/hub, else
//              ~/.cache/huggingface/hub (token: $HF_TOKEN or the `hf auth login`
//              token file); plans in $HP_CACHE_DIR, else $XDG_CACHE_HOME/hyperparse,
//              else ~/.cache/hyperparse, under engines/<manifest hash>-sm<cc>-trt<version>.
// Downloads and builds run in a child process, so the caller stays free of CUDA
// and threads (Engine forks its page workers next). Throws std::runtime_error.
std::string prepare_models(const std::string &dir, ModelNeeds needs);

}  // namespace hp
