#pragma once
// Every GPU resource the pipeline uses, owned by one object: TensorRT engines and
// contexts, their device buffers, streams and events. Loading the models means
// constructing a ModelSet, and unloading means destroying it. Nothing GPU-side
// lives anywhere else, so an unload cannot leak a buffer that some other module
// still holds. (The CUDA context itself stays; see Engine::unload.)
//
// Layout runs on two lanes, each with its own D-FINE context and stream, so
// page i+1's upload, resize and layout overlap page i's OCR on the shared OCR
// stream. Formula decoding gets its own stream on the assembly thread.

#include <array>
#include <memory>
#include <string>

#include "crop_batch.h"
#include "cuda_raii.h"
#include "dfine_layout.h"
#include "formula.h"
#include "text_det.h"
#include "text_recognizer.h"
#include "core/common/cuda/cuda_ptr.h"

namespace hp {

// D-FINE canvas. 960: 14.1 ms/page, class acc 0.917 (100 OmniDocBench pages);
// 800 was 9.3 ms/page but class acc 0.901, so only 960 is shipped.
inline constexpr int kLayoutSize = 960;

struct ModelConfig {
  std::string dir;               // holds the *.plan files, keys_small.txt, formula/
  // PP-OCR det long side. 960 instead of 1280: 16.6 -> 18.9 pages/s on IDL scans,
  // Textract word F1 0.8944 -> 0.8915 (small text is missed).
  int det_max_side = 1280;
  bool ocr = true;               // load det + rec; without them, pages without a text layer get layout only
  bool formula = true;           // load PP-FormulaNet-S
  RecMode rec = RecMode::kColumnBudget;
};

// Wall time per component while constructing a ModelSet: plan read +
// TensorRT deserialize + context creation + buffer allocation (and, for
// kWidthBucket rec, CUDA graph capture).
struct ModelLoadTimes {
  double layout_ms = 0;   // both D-FINE lanes
  double det_ms = 0;
  double rec_ms = 0;
  double formula_ms = 0;  // encoder + prep + decode step
};

// One layout lane: a D-FINE context, the page buffer it reads, and its stream.
struct LayoutLane {
  DfineLayout layout;
  hp::core::CudaPtr<uint8_t> page;  // current page, u8 BGR
  size_t page_cap = 0;
  CudaStream stream;
  CudaEvent resized;  // multi_resize done: the OCR stream may read the det input

  // Grow the page buffer to hold `bytes` (never shrinks while loaded).
  uint8_t *page_buffer(size_t bytes);
};

class ModelSet {
 public:
  static constexpr int kLanes = 2;

  // Loads every engine `cfg` asks for. Throws std::runtime_error naming the file
  // that failed; a partially built set is released before the throw.
  explicit ModelSet(const ModelConfig &cfg);
  // Synchronizes all streams, then frees everything.
  ~ModelSet();
  ModelSet(const ModelSet &) = delete;
  ModelSet &operator=(const ModelSet &) = delete;

  // Runs every engine and code path once on a synthetic page (see model_set.cpp).
  // It pays first-use costs before real traffic: lazy CUDA module loading, TRT
  // per-context setup, and buffer growth.
  void warmup();
  void synchronize();

  [[nodiscard]] const ModelConfig &config() const { return cfg_; }
  [[nodiscard]] const ModelLoadTimes &load_times() const { return load_times_; }
  [[nodiscard]] LayoutLane &lane(int i) { return *lanes_[i]; }
  [[nodiscard]] bool has_ocr() const { return det_ != nullptr; }
  [[nodiscard]] bool has_formula() const { return formula_ != nullptr; }
  [[nodiscard]] TextDet &det() { return *det_; }
  [[nodiscard]] CropBatch &det_crops() { return det_crops_; }
  [[nodiscard]] TextRecognizer &rec() { return *rec_; }
  [[nodiscard]] FormulaRecognizer &formula() { return *formula_; }
  [[nodiscard]] cudaStream_t ocr_stream() const { return ocr_stream_; }
  [[nodiscard]] cudaStream_t formula_stream() const { return formula_stream_; }

 private:
  ModelConfig cfg_;
  ModelLoadTimes load_times_;
  // Streams first: members are destroyed in reverse order, so the streams outlive
  // every engine and buffer that may still have work queued on them.
  CudaStream ocr_stream_, formula_stream_;
  std::array<std::unique_ptr<LayoutLane>, kLanes> lanes_;
  std::unique_ptr<TextDet> det_;
  CropBatch det_crops_;
  std::unique_ptr<TextRecognizer> rec_;
  std::unique_ptr<FormulaRecognizer> formula_;
};

}  // namespace hp
