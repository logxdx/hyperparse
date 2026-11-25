#pragma once
// One page through LayoutNet: units -> features -> network -> decoded tables, as liteparse
// table_model/mod.rs run_page / run_page_framed (tables only; block classes are not used).

#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "tablenet/decode.h"
#include "tablenet/features.h"
#include "tablenet/tablenet.h"
#include "tablenet/units.h"

namespace hp::tn {

struct PageInput {
  std::vector<Unit> units;  // points, page frame (native units, OCR units, or both)
  double w = 0, h = 0;      // page size, points
  bool ocr = false;         // units come from OCR (never turned upright: no reading direction)
  // the page render: grey (grey_from_bgr), its size and pixels per point
  std::vector<uint8_t> grey;
  int px_w = 0, px_h = 0;
  float px_per_pt = 0;
};

struct Table {
  std::array<double, 4> bbox{};  // page points
  std::vector<int> units;        // indices into PageInput::units
  Grid grid;
};

struct PageStats {
  int units = 0;
  double features_ms = 0, net_ms = 0, decode_ms = 0;
};

// The page's tables (empty when it has < 2 or > kMaxUnits units, or under 20 characters).
std::vector<Table> run_page(TableNet &net, const PageInput &in, cudaStream_t stream, PageStats *stats = nullptr);

}  // namespace hp::tn
