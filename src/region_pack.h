#pragma once
// Shelf-pack a page's OCR regions into one det canvas, so detection runs once
// over just those regions instead of over the whole page.

#include <algorithm>
#include <vector>

namespace hp {

struct PackedRegion {
  int x, y, w, h;     // source rect, page px
  int ox, oy, pw, ph; // placement in the canvas (det px)
};

struct RegionPack {
  std::vector<PackedRegion> regions;
  int canvas_w = 0, canvas_h = 0;

  // rects: page px (already clipped); scale: det px per page px; max_w: canvas width cap.
  void pack(const std::vector<PackedRegion> &rects, float scale, int max_w, int gap = 8) {
    regions = rects;
    for (auto &r : regions) {
      r.pw = std::max(1, static_cast<int>(r.w * scale + 0.5f));
      r.ph = std::max(1, static_cast<int>(r.h * scale + 0.5f));
    }
    std::sort(regions.begin(), regions.end(), [](const auto &a, const auto &b) { return a.ph > b.ph; });
    int widest = 0;
    for (const auto &r : regions) widest = std::max(widest, r.pw);
    const int W = std::max(widest, max_w);
    int x = 0, y = 0, shelf_h = 0, used_w = 0;
    for (auto &r : regions) {
      if (x > 0 && x + r.pw > W) { y += shelf_h + gap; x = 0; shelf_h = 0; }
      r.ox = x;
      r.oy = y;
      x += r.pw + gap;
      shelf_h = std::max(shelf_h, r.ph);
      used_w = std::max(used_w, r.ox + r.pw);
    }
    canvas_w = std::max(32, (used_w + 31) / 32 * 32);
    canvas_h = std::max(32, (y + shelf_h + 31) / 32 * 32);
  }

  // Canvas point -> page px; false if it falls in no placed region.
  bool to_page(float cx, float cy, const PackedRegion *&hit) const {
    for (const auto &r : regions)
      if (cx >= r.ox && cx < r.ox + r.pw && cy >= r.oy && cy < r.oy + r.ph) { hit = &r; return true; }
    return false;
  }
};

}  // namespace hp
