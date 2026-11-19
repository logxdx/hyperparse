#include "core/docassembly/block_builder.h"

#include "core/common/text_flow.h"

#include <vector>

#include <algorithm>
#include <climits>
#include <cmath>
#include <utility>

#include "core/common/geometry/box.h"

namespace hp::core::docassembly {

namespace {

// Vertical centre of an item's (possibly rotated) box.
int centre_y(const OCRResultItem &it) {
  int sy = 0;
  for (int k = 0; k < 4; ++k) sy += it.box[k][1];
  return sy / 4;
}

// Right-most x of an item's box.
int right_x(const OCRResultItem &it) {
  int mx = INT_MIN;
  for (int k = 0; k < 4; ++k) mx = std::max(mx, it.box[k][0]);
  return mx;
}

// Join member texts into region content. Row grouping and the
// space-vs-newline rule live in text_flow.h, shared with the serializer path.
std::string join_content(const std::vector<OCRResultItem> &results,
                         const std::vector<int> &members,
                         const layout::LayoutBox &cell) {
  const auto [cx0, cy0, cx1, cy1] = hp::core::aabb(cell.box);
  (void)cx0;
  (void)cy0;
  (void)cy1;

  std::vector<hp::core::TextSpan> spans;
  spans.reserve(members.size());
  for (const int mi : members) {
    const auto &it = results[static_cast<std::size_t>(mi)];
    int x0 = INT_MAX, x1 = INT_MIN;
    for (int k = 0; k < 4; ++k) {
      x0 = std::min(x0, it.box[k][0]);
      x1 = std::max(x1, it.box[k][0]);
    }
    spans.push_back({centre_y(it), x0, x1});
  }
  const auto rows = hp::core::plan_rows(spans, cx1, cell.text_line_height);

  const auto text_of = [&](int i) -> const std::string & {
    return results[static_cast<std::size_t>(members[static_cast<std::size_t>(i)])]
        .text;
  };

  std::string out;
  bool first_row = true;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    if (rows[r].continuation) continue;  // rendered by its host row, below
    if (!first_row)
      out += (rows[r].join == hp::core::RowJoin::kSpace) ? ' ' : '\n';
    first_row = false;
    for (int i = rows[r].begin; i < rows[r].end; ++i) {
      if (i > rows[r].begin) out += ' ';
      out += text_of(i);
      // A wrapped continuation belongs immediately after the label it
      // continues, ahead of this row's remaining spans (the amount).
      if (i == rows[r].begin)
        for (std::size_t c = r + 1; c < rows.size() && rows[c].continuation;
             ++c)
          for (int m = rows[c].begin; m < rows[c].end; ++m) {
            out += ' ';
            out += text_of(m);
          }
    }
  }
  return out;
}

} // namespace


bool class_expects_text(int class_id) noexcept {
  // Everything except the purely pictorial / stamp classes. Formula classes
  // DO expect text: display_formula and inline_formula regions carry glyphs
  // the detector should find, and a formula region with no detections is a
  // real miss worth surfacing.
  switch (class_id) {
    case 3:   // chart
    case 9:   // footer_image
    case 13:  // header_image
    case 14:  // image
    case 20:  // seal
    case 21:  // table  — covered by its own cell recognition, not line boxes
      return false;
    default:
      return true;
  }
}

float region_coverage(const std::array<int, 4> &region,
                      const std::vector<std::array<int, 4>> &boxes) noexcept {
  const long long rw = static_cast<long long>(region[2]) - region[0];
  const long long rh = static_cast<long long>(region[3]) - region[1];
  if (rw <= 0 || rh <= 0) return -1.0f;
  const double region_area = static_cast<double>(rw) * static_cast<double>(rh);
  if (boxes.empty()) return 0.0f;

  // Clip every box to the region and drop the empties.
  std::vector<std::array<int, 4>> clipped;
  clipped.reserve(boxes.size());
  for (const auto &b : boxes) {
    const int x0 = std::max(b[0], region[0]);
    const int y0 = std::max(b[1], region[1]);
    const int x1 = std::min(b[2], region[2]);
    const int y1 = std::min(b[3], region[3]);
    if (x1 > x0 && y1 > y0) clipped.push_back({x0, y0, x1, y1});
  }
  if (clipped.empty()) return 0.0f;

  // Coordinate compression: the union area is the sum of the compressed grid
  // cells that at least one box covers.
  std::vector<int> xs, ys;
  xs.reserve(clipped.size() * 2);
  ys.reserve(clipped.size() * 2);
  for (const auto &b : clipped) {
    xs.push_back(b[0]);
    xs.push_back(b[2]);
    ys.push_back(b[1]);
    ys.push_back(b[3]);
  }
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
  std::sort(ys.begin(), ys.end());
  ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
  if (xs.size() < 2 || ys.size() < 2) return 0.0f;

  const std::size_t nx = xs.size() - 1, ny = ys.size() - 1;
  std::vector<char> filled(nx * ny, 0);
  for (const auto &b : clipped) {
    const auto xi0 = static_cast<std::size_t>(
        std::lower_bound(xs.begin(), xs.end(), b[0]) - xs.begin());
    const auto xi1 = static_cast<std::size_t>(
        std::lower_bound(xs.begin(), xs.end(), b[2]) - xs.begin());
    const auto yi0 = static_cast<std::size_t>(
        std::lower_bound(ys.begin(), ys.end(), b[1]) - ys.begin());
    const auto yi1 = static_cast<std::size_t>(
        std::lower_bound(ys.begin(), ys.end(), b[3]) - ys.begin());
    for (std::size_t yi = yi0; yi < yi1 && yi < ny; ++yi)
      for (std::size_t xi = xi0; xi < xi1 && xi < nx; ++xi)
        filled[yi * nx + xi] = 1;
  }

  double covered = 0.0;
  for (std::size_t yi = 0; yi < ny; ++yi) {
    const double h = static_cast<double>(ys[yi + 1] - ys[yi]);
    for (std::size_t xi = 0; xi < nx; ++xi) {
      if (filled[yi * nx + xi])
        covered += static_cast<double>(xs[xi + 1] - xs[xi]) * h;
    }
  }
  const double frac = covered / region_area;
  return static_cast<float>(frac < 0.0 ? 0.0 : (frac > 1.0 ? 1.0 : frac));
}

float page_coverage(const std::vector<ParsingBlock> &blocks) noexcept {
  double weighted = 0.0, weight = 0.0;
  for (const auto &b : blocks) {
    if (b.coverage < 0.0f) continue;  // text not expected here
    const double w =
        static_cast<double>(b.bbox[2] - b.bbox[0]) *
        static_cast<double>(b.bbox[3] - b.bbox[1]);
    if (w <= 0.0) continue;
    weighted += static_cast<double>(b.coverage) * w;
    weight += w;
  }
  if (weight <= 0.0) return -1.0f;
  return static_cast<float>(weighted / weight);
}

float aggregate_confidence(const std::vector<OCRResultItem> &results,
                           const std::vector<int> &members) {
  double weighted = 0.0;
  double weight = 0.0;
  for (int mi : members) {
    const auto &it = results[static_cast<std::size_t>(mi)];
    if (it.confidence < 0.0f) continue;  // never scored — skip, don't zero
    // Weight by text length so a one-character fragment does not count as
    // much as a full line.
    const double w = static_cast<double>(std::max<std::size_t>(it.text.size(), 1));
    weighted += static_cast<double>(it.confidence) * w;
    weight += w;
  }
  if (weight <= 0.0) return -1.0f;
  return static_cast<float>(weighted / weight);
}

std::string aggregate_source(const std::vector<OCRResultItem> &results,
                             const std::vector<int> &members) {
  std::string acc;
  bool seen = false;
  for (int mi : members) {
    const std::string &s = results[static_cast<std::size_t>(mi)].source;
    if (!seen) {
      acc = s;
      seen = true;
      continue;
    }
    acc = merge_source(acc, s);
  }
  return acc;
}

std::vector<ParsingBlock> build_page_blocks(
    const std::vector<OCRResultItem> &results,
    const std::vector<layout::LayoutBox> &layout,
    const std::vector<int> &reading_order, int page_index,
    const std::vector<std::pair<int, std::string>> &table_html) {
  std::vector<ParsingBlock> blocks;
  if (layout.empty() || reading_order.empty()) return blocks;

  // Bucket results by layout region, remembering the order regions were first
  // touched so the emitted blocks come out in reading order.
  std::vector<int> rank_of_layout(layout.size(), -1);
  std::vector<std::vector<int>> grouped(layout.size());
  int next_rank = 0;
  for (int ri : reading_order) {
    if (ri < 0 || static_cast<std::size_t>(ri) >= results.size()) continue;
    const int lid = results[static_cast<std::size_t>(ri)].layout_id;
    if (lid < 0 || static_cast<std::size_t>(lid) >= layout.size()) continue;
    if (rank_of_layout[static_cast<std::size_t>(lid)] < 0)
      rank_of_layout[static_cast<std::size_t>(lid)] = next_rank++;
    grouped[static_cast<std::size_t>(lid)].push_back(ri);
  }

  std::vector<int> layout_of_rank(static_cast<std::size_t>(next_rank), -1);
  for (std::size_t li = 0; li < layout.size(); ++li)
    if (rank_of_layout[li] >= 0)
      layout_of_rank[static_cast<std::size_t>(rank_of_layout[li])] =
          static_cast<int>(li);

  blocks.reserve(static_cast<std::size_t>(next_rank));
  for (int r = 0; r < next_rank; ++r) {
    const int li = layout_of_rank[static_cast<std::size_t>(r)];
    if (li < 0) continue;
    const auto &cell = layout[static_cast<std::size_t>(li)];
    const auto &members = grouped[static_cast<std::size_t>(li)];
    if (members.empty()) continue;

    ParsingBlock b;
    b.label = std::string(layout::label_name(cell.class_id));
    const auto [x0, y0, x1, y1] = hp::core::aabb(cell.box);
    b.bbox = {x0, y0, x1, y1};
    b.page_index = page_index;
    b.confidence = aggregate_confidence(results, members);
    b.source = aggregate_source(results, members);
    if (class_expects_text(cell.class_id)) {
      std::vector<std::array<int, 4>> boxes;
      boxes.reserve(members.size());
      for (int mi : members) {
        const auto [bx0, by0, bx1, by1] =
            hp::core::aabb(results[static_cast<std::size_t>(mi)].box);
        boxes.push_back({bx0, by0, bx1, by1});
      }
      b.coverage = region_coverage(b.bbox, boxes);
    }

    // A recognized table's structure is its content, so cross-page merge has
    // HTML to splice rather than the flat text of its cells.
    const std::string *html = nullptr;
    for (const auto &kv : table_html)
      if (kv.first == li) {
        html = &kv.second;
        break;
      }
    b.content = (b.label == "table" && html && !html->empty())
                    ? *html
                    : join_content(results, members, cell);

    blocks.push_back(std::move(b));
  }

  // ---- text-free regions ----
  //
  // A figure, photograph or chart has no OCR results, so it never appears in
  // `reading_order` (which indexes *results*), never receives a rank above,
  // and was silently dropped. Measured on an architectural cover sheet: layout
  // found 4 `image` regions and the block list contained none of them.
  //
  // That is not a cosmetic omission. A region absent from the IR cannot be
  // cited, cannot be counted, and cannot be routed to a stronger reader — so
  // "send charts to a VLM" is impossible for exactly the regions that need it.
  //
  // They are emitted with an empty `content`, `confidence` taken from the
  // layout detector's own score, and `source` = "layout" so no consumer
  // mistakes a detection score for a recognition score.
  std::vector<ParsingBlock> textfree;
  for (std::size_t li = 0; li < layout.size(); ++li) {
    if (rank_of_layout[li] >= 0) continue;  // had text, already emitted
    const auto &cell = layout[li];
    ParsingBlock b;
    b.label = std::string(layout::label_name(cell.class_id));
    const auto [x0, y0, x1, y1] = hp::core::aabb(cell.box);
    b.bbox = {x0, y0, x1, y1};
    b.page_index = page_index;
    b.confidence = cell.score;
    b.source = std::string(kSourceLayout);
    // The load-bearing case for this signal. A region the layout model called
    // text, in which detection found nothing at all, is coverage 0.0 — not
    // "not applicable". Its `confidence` here is the layout detector's own
    // score, which can be high and says nothing about whether the text was
    // read, so coverage is the only field that reports the miss.
    b.coverage = class_expects_text(cell.class_id) ? 0.0f : -1.0f;
    textfree.push_back(std::move(b));
  }
  if (textfree.empty()) return blocks;

  // Place them by position rather than appending: a figure halfway down a page
  // belongs halfway down the block list, not after the footer. Insertion is by
  // the region's centre against each placed block's centre, top-to-bottom then
  // left-to-right. This deliberately does not attempt to re-run XY-cut — it
  // approximates where the region sits in the flow the text blocks already
  // established.
  const auto centre = [](const ParsingBlock &b) {
    return std::pair<int, int>{(b.bbox[1] + b.bbox[3]) / 2,
                               (b.bbox[0] + b.bbox[2]) / 2};
  };
  // Sorted first so multiple insertions keep their relative order.
  std::sort(textfree.begin(), textfree.end(),
            [&](const ParsingBlock &a, const ParsingBlock &b) {
              return centre(a) < centre(b);
            });
  for (auto &tf : textfree) {
    const auto c = centre(tf);
    std::size_t at = blocks.size();
    for (std::size_t i = 0; i < blocks.size(); ++i)
      if (c < centre(blocks[i])) {
        at = i;
        break;
      }
    blocks.insert(blocks.begin() + static_cast<std::ptrdiff_t>(at),
                  std::move(tf));
  }
  return blocks;
}

} // namespace hp::core::docassembly
