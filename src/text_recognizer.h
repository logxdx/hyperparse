#pragma once
// Text-line recognition behind one interface, so the pipeline does not care which
// batching strategy runs the PP-OCR CTC model.
//
//   kColumnBudget (default) hp::RecBatcher: exact widths packed under an
//                 8192-column budget. On 751 IDL scans it took rec from
//                 56.0 to 37.2 ms/page (12.7 -> 16.6 pages/s) at equal Textract
//                 word F1 (0.8944 vs 0.8946); see rec_batcher.h for the
//                 L2-cliff measurement behind the budget.
//   kWidthBucket  PaddleRec (core/recognition): up to 32 lines per
//                 width bucket, CUDA graphs per static profile. Kept for A/B runs.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/common/geometry/box.h"
#include "core/decode/gpu_image.h"

namespace hp {

enum class RecMode { kColumnBudget, kWidthBucket };

class TextRecognizer {
 public:
  virtual ~TextRecognizer() = default;
  // (text, confidence) per box, in input order. May synchronize `stream`.
  // no_cjk: decode with the CJK classes excluded (script.h).
  [[nodiscard]] virtual std::vector<std::pair<std::string, float>> run(const hp::core::GpuImage &img,
                                                                       const std::vector<hp::core::Box> &boxes,
                                                                       cudaStream_t stream, bool no_cjk) = 0;
};

// Throws std::runtime_error when the plan or dictionary cannot be loaded.
std::unique_ptr<TextRecognizer> load_text_recognizer(RecMode mode, const std::string &plan, const std::string &dict,
                                                     cudaStream_t stream);

}  // namespace hp
