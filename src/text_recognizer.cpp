#include "text_recognizer.h"

#include <stdexcept>

#include "rec_batcher.h"
#include "core/recognition/paddle_rec.h"

namespace hp {
namespace {

// Adapter: the width-bucket PaddleRec (core/recognition) behind the TextRecognizer interface.
class WidthBucketRecognizer final : public TextRecognizer {
 public:
  WidthBucketRecognizer(const std::string &plan, const std::string &dict, cudaStream_t stream) {
    if (!rec_.load_model(plan) || !rec_.load_dict(dict)) throw std::runtime_error("rec load failed: " + plan);
    rec_.allocate_buffers();
    rec_.bake_graphs(stream);
  }
  std::vector<std::pair<std::string, float>> run(const hp::core::GpuImage &img, const std::vector<hp::core::Box> &boxes,
                                                 cudaStream_t stream, bool no_cjk) override {
    rec_.set_no_cjk(no_cjk, stream);
    return rec_.run(img, boxes, stream);
  }

 private:
  hp::core::recognition::PaddleRec rec_;
};

}  // namespace

std::unique_ptr<TextRecognizer> load_text_recognizer(RecMode mode, const std::string &plan, const std::string &dict,
                                                     cudaStream_t stream) {
  if (mode == RecMode::kWidthBucket) return std::make_unique<WidthBucketRecognizer>(plan, dict, stream);
  auto rec = std::make_unique<RecBatcher>();
  if (!rec->load(plan, dict)) throw std::runtime_error("rec load failed: " + plan);
  return rec;
}

}  // namespace hp
