#pragma once
// LayoutNet inputs: unit features [N, 68], ruling lines and their unit / pair flags, and the
// image-branch page. A port of liteparse table_model/features.rs. LiteParse's grid-projection
// signals (features 53..60 and the block / leaf ids) are always absent here: on OmniDocBench
// tables and olmOCR-bench table tests LayoutNet scored the same or better without them
// (TEDS 0.511 vs 0.495; 68.0 vs 66.0), so hp does not reproduce LiteParse's projection.

#include <array>
#include <cstdint>
#include <vector>

#include "tablenet/units.h"

namespace hp::tn {

inline constexpr int kFeatDim = 68;
inline constexpr int kImgSide = 1024;  // the image branch's square canvas
inline constexpr float kRulesDpi = 150.f;  // training rendered pages at 150 dpi

// Ruling lines in page points: horizontal {y, x0, x1}, vertical {x, y0, y1}.
struct Rules {
  std::vector<std::array<float, 3>> h, v;
};

// Grey from a BGR render, as training did: OpenCV's RGB2GRAY integer weights applied to PDFium's
// BGR channel order (blue gets the red weight). `stride` in bytes.
std::vector<uint8_t> grey_from_bgr(const uint8_t *bgr, int w, int h, size_t stride);

// Dark pixels (< 170) in runs of at least 3% of the page side (15 px floor), each 8-connected
// component of runs one rule unless thicker than 2.5 pt. `scale` = pixels per point.
Rules detect_rules(const std::vector<uint8_t> &grey, int w, int h, float scale);

// The page for the image branch: grey with its longest side kImgSide (area-resampled), and its
// pixels per point.
struct PageImage {
  int w = 0, h = 0;
  std::vector<uint8_t> px;
  float k = 0;
};
PageImage page_image(const std::vector<uint8_t> &grey, int w, int h, float px_per_pt);

// The model's image inputs: canvas [kImgSide^2] ink in [0, 1] (white = 0), unit boxes as
// fractions of the canvas, and the page extent on it.
struct ImgInput {
  std::vector<float> canvas;
  std::vector<std::array<float, 4>> boxes;
  float valid[2] = {0, 0};
};
ImgInput image_input(const PageImage &im, const std::vector<Unit> &units);

// [N, kFeatDim] row-major. `rules` null = unknown (rule flags all 0, including "known").
std::vector<float> unit_features(const std::vector<Unit> &units, double page_w, double page_h, const Rules *rules);

// [N, N, 2]: rules strictly between two units that span both, counts capped at 3, scaled.
std::vector<float> pair_rule_features(const std::vector<Unit> &units, const Rules &rules);

}  // namespace hp::tn
