#pragma once
// LayoutNet logits -> tables: a C++ port of liteparse table_model/decode.rs (itself a port of
// tn_core.py) plus the Grid filters of table_model/mod.rs (is_toc / is_prose / is_chart).
// Order-sensitive steps (argmax ties, stable sorts, insertion order) follow the Rust exactly so
// tests/tablenet_parity compares against the reference dumps byte for byte.

#include <array>
#include <string>
#include <vector>

namespace hp::tn {

// Node logits [n, 2] (in_table, header), pair logits [n_pair, n, n] row-major (same_table,
// same_row, same_col, same_cell[, same_block]), class logits [n, n_cls] (may be empty).
struct Logits {
  int n = 0;
  std::vector<float> node, pair, cls;
  int n_cls = 0;
  int n_pair = 4;
  [[nodiscard]] float node_at(int i, int k) const { return node[static_cast<size_t>(i) * 2 + k]; }
  [[nodiscard]] float pair_at(int k, int i, int j) const {
    return pair[(static_cast<size_t>(k) * n + i) * n + j];
  }
};

// Units running against the page's main direction take part in the forward pass but are never
// cells: all their node and pair logits (first 4 pair heads) become -9 (mod.rs mask_minor).
void mask_minor(Logits &l, const std::vector<bool> &minor);

// A text unit as the decoder sees it: box in the text's reading frame (points, top-left origin).
struct DUnit {
  std::array<double, 4> bx;
  std::string text;
  double h;
};

struct DecodeOptions {
  float table_thresh = 0.f;
  bool head_band = true;
  bool forced_link = true;
  bool squeeze = true;
  bool fold_wraps = true;
};

// Average-linkage clusters of a symmetric n x n matrix (merge while mean > thresh), as decode.rs.
std::vector<std::vector<int>> avg_link(const std::vector<double> &p, int n, double thresh);

// Unit groups, one per table (decode.rs decode_tables).
std::vector<std::vector<int>> decode_tables(const std::vector<DUnit> &units, const Logits &l,
                                            const DecodeOptions &opt = {});

struct GridCell {
  int r0 = 0, c0 = 0, r1 = 0, c1 = 0;
  std::string text;
  std::array<double, 4> bx{};
};

struct Grid {
  std::vector<GridCell> cells;  // placement order
  std::vector<int> head_rows;
  int n_rows = 0, n_cols = 0;

  [[nodiscard]] std::string to_html() const;  // decode.rs Grid::to_html, byte for byte
  [[nodiscard]] bool is_toc() const;          // mod.rs
  [[nodiscard]] bool is_prose() const;        // mod.rs
  [[nodiscard]] bool is_chart() const;        // mod.rs
};

// One table's grid from its unit indices (decode.rs decode_grid).
Grid decode_grid(const std::vector<DUnit> &units, const std::vector<int> &idx, const Logits &l,
                 const DecodeOptions &opt = {});

}  // namespace hp::tn
