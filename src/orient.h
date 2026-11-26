#pragma once
// Page orientation: PP-LCNet_x1_0_doc_ori (PaddleOCR, Apache-2.0; 0/90/180/270) on TensorRT.
// Picked over Indra OCR's orient model and OCRia-nano's router flip vote on 2,520 synthetic
// rotations + OmniParseBench rotation_skew (ocr-compare/nano-report/orient): with a 0.8
// confidence gate, 95.1% exact angle, 97.6% on real /Rotate pages, 0 of 614 upright pages turned.
// Without the gate it turns 2-4% of upright pages (plates, sparse pages), so the gate stays.
//
// Preprocessing is PaddleX's (inference.yml): short side to 256 (bilinear, as cv2 INTER_LINEAR),
// centre crop 224, RGB / 255, ImageNet mean/std, CHW. Output: probabilities over {0, 90, 180, 270};
// label L = the page content is turned L degrees clockwise.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/common/cuda/cuda_ptr.h"
#include "core/engine/trt/trt_engine.h"

namespace hp {

inline constexpr float kOrientMinConf = 0.8f;

struct OrientResult {
  int degrees_cw = 0;  // how far the content is turned clockwise (0, 90, 180, 270)
  float conf = 0.f;    // probability of that class
};

class DocOrient {
 public:
  bool load(const std::string &plan_path);
  // One page, u8 BGR (row stride w*3). Blocks until the result is on the host.
  OrientResult classify(const uint8_t *bgr, int w, int h, cudaStream_t stream);

 private:
  std::unique_ptr<hp::core::engine::TrtEngine> engine_;
  hp::core::CudaPtr<float> d_in_, d_out_;
  std::vector<float> h_in_;
};

// Turns a u8 BGR image (w x h, stride w*3) counter-clockwise by `degrees_cw` in place, so content
// turned that far clockwise reads upright. 90/270 swap w and h (same byte count).
void rotate_upright_bgr(uint8_t *bgr, int &w, int &h, int degrees_cw);

// Maps a box from the upright frame (after rotate_upright_bgr) back to the original image frame
// (w0 x h0, the size before rotation).
void box_to_original(float &x0, float &y0, float &x1, float &y1, int degrees_cw, float w0, float h0);

}  // namespace hp
