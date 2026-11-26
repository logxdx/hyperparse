#pragma once
// OrderNet (Indra OCR's order.onnx): reading order and block structure over text lines.
// A C++ port of indra_ocr's order_post.py (line features, block-order decoding) and
// layout_post.py (blocks -> Markdown), indra_ocr 1.1.2 (Apache-2.0). The network itself runs
// on TensorRT (ordernet/net.h); everything here is CPU and deterministic.
//
// Python semantics are kept where they decide the output (str.isdigit / isupper / strip,
// East Asian width, round-half-even, stable sorts); tests/ordernet_parity_test checks against
// dumps from the Python reference.

#include <array>
#include <string>
#include <vector>

namespace hp::on {

inline constexpr int kFeats = 16;   // N_GEOM 10 + N_TEXT 6
inline constexpr int kClasses = 10;  // BLOCK_CLASSES

// BLOCK_CLASSES, in the cls head's order.
inline constexpr std::array<const char *, kClasses> kBlockClasses = {
    "Text", "Section-Header", "Page-Header", "Page-Footer", "List-Group",
    "Table", "Caption", "Footnote", "Table-Of-Contents", "Form"};

struct Line {
  std::array<float, 4> box;  // x0, y0, x1, y1 (page px)
  std::string text;          // UTF-8
};

// OrderNet outputs for N lines: a [N, N] (a[i, j] > 0: i before j), cls [N, kClasses],
// same [N, N] (same-block logits), row-major.
struct Heads {
  int n = 0;
  std::vector<float> a, cls, same;
};

// order_post.line_features: [N, kFeats] row-major. W, H: page size (px).
std::vector<float> line_features(const std::vector<Line> &lines, float W, float H);

// order_post.decode_order: order from a pairwise logit matrix [n, n].
std::vector<int> decode_order(const std::vector<float> &a, int n, int sweeps = 4);

// order_post.order_xycut / order_top_down over boxes.
std::vector<int> order_xycut(const std::vector<std::array<float, 4>> &boxes);
std::vector<int> order_top_down(const std::vector<std::array<float, 4>> &boxes);

// order_post.decode_blocks_order: blocks by average linkage of `same`, lines inside a block by
// XY-cut (row by row in Table / Form / Table-Of-Contents blocks), blocks by mean order logit.
std::vector<int> decode_blocks_order(const Heads &h, const std::vector<Line> &lines);

// layout_post.split_blocks: the reading order cut into blocks where consecutive lines'
// same-block logit is negative.
std::vector<std::vector<int>> split_blocks(const std::vector<int> &order, const Heads &h);

// layout_post.page_markdown (keep_furniture, no cells, no table heads, no pictures): blocks
// rendered by class (## headings, list items, display math lines, geometric tables, paragraphs
// with hyphenated words joined), furniture blocks dropped unless keep_furniture. `md`, when given,
// holds each line's styled Markdown (style_md.h), rendered in place of its plain text outside
// headings and tables.
std::string page_markdown(const std::vector<Line> &lines, const std::vector<int> &order, const Heads &h,
                          bool keep_furniture = false, const std::vector<std::string> *md = nullptr);

}  // namespace hp::on
