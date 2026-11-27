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

#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/common/geometry/box.h"
#include "core/decode/gpu_image.h"

namespace hp {

enum class RecMode { kColumnBudget, kWidthBucket };

// One set of boxes recognised as a unit (batched on its own, as by one run()).
struct RecGroup {
  const std::vector<hp::core::Box> *boxes;
  bool no_cjk;
};

class TextRecognizer {
 public:
  using Texts = std::vector<std::pair<std::string, float>>;
  virtual ~TextRecognizer() = default;
  // (text, confidence) per box, in input order. May synchronize `stream`.
  // no_cjk: decode with the CJK classes excluded (script.h).
  [[nodiscard]] virtual Texts run(const hp::core::GpuImage &img, const std::vector<hp::core::Box> &boxes,
                                  cudaStream_t stream, bool no_cjk) = 0;

  // Split form of run() for pipelining: enqueue() queues every group on `stream`
  // (each group batched exactly as run() would batch it alone) and returns
  // without waiting; collect() waits and returns the texts of all groups,
  // concatenated in group order. One enqueue in flight at a time; `img` must stay
  // valid until collect(). The default runs synchronously inside enqueue().
  virtual void enqueue(const hp::core::GpuImage &img, std::span<const RecGroup> groups, cudaStream_t stream) {
    pending_.clear();
    for (const RecGroup &g : groups) {
      Texts t = run(img, *g.boxes, stream, g.no_cjk);
      pending_.insert(pending_.end(), std::make_move_iterator(t.begin()), std::make_move_iterator(t.end()));
    }
  }
  [[nodiscard]] virtual Texts collect() { return std::move(pending_); }

 protected:
  Texts pending_;
};

// Throws std::runtime_error when the plan or dictionary cannot be loaded.
std::unique_ptr<TextRecognizer> load_text_recognizer(RecMode mode, const std::string &plan, const std::string &dict,
                                                     cudaStream_t stream);

}  // namespace hp
