#pragma once

#include <climits>
#include <cstdlib>
#include <vector>

// How the text spans inside one layout region are joined back into a string.
//
// This lived twice — once in serialization_blocks.h for the image/blocks
// routes and once in docassembly/block_builder.cpp for the document routes —
// with the same rule written out longhand in both. They agreed by hand, which
// is to say they agreed until someone fixed one of them: a fix to the
// docassembly copy changed nothing observable, because `/ocr/raw?as_blocks=1`
// runs the serializer copy. One definition, rendered by both call sites.

namespace hp::core {

// Geometry of one span, in the order reading_order.cpp already sorted them.
struct TextSpan {
  int cy;  // centre y
  int x0;  // left edge
  int x1;  // right edge
};

// How a row attaches to the row before it.
enum class RowJoin { kNewline, kSpace };

struct TextRow {
  int begin = 0;  // [begin, end) into the span list
  int end = 0;
  RowJoin join = RowJoin::kNewline;
  // A wrapped continuation of the PREVIOUS row's label: its text belongs
  // immediately after that label, not on a line of its own and not after that
  // row's amount. Rows so marked are rendered by their host, not in sequence.
  bool continuation = false;
};

// Group spans into rows and say how each attaches. Callers render; nothing
// here knows about JSON escaping or string building.
[[nodiscard]] inline std::vector<TextRow>
plan_rows(const std::vector<TextSpan> &spans, int cell_x1,
          int text_line_height) {
  std::vector<TextRow> rows;
  if (spans.empty()) return rows;

  const int line_h = text_line_height > 0 ? text_line_height : 8;
  const int wrap_tol = line_h > 8 ? line_h : 8;

  // 1. Band into rows against the row's FIRST member, matching the ordering
  //    in reading_order.cpp. Chaining from the previous member instead lets a
  //    skewed or gradually descending row drift arbitrarily far from where it
  //    began and swallow the next one.
  int anchor = spans.front().cy;
  TextRow cur;
  cur.begin = 0;
  for (int i = 0; i < static_cast<int>(spans.size()); ++i) {
    if (i > 0 && std::abs(spans[static_cast<size_t>(i)].cy - anchor) > line_h / 2) {
      cur.end = i;
      rows.push_back(cur);
      cur = TextRow{};
      cur.begin = i;
      anchor = spans[static_cast<size_t>(i)].cy;
    }
  }
  cur.end = static_cast<int>(spans.size());
  rows.push_back(cur);

  // 2. Classify each row against the last row that was not a continuation.
  int host = -1;  // index of the current host row
  for (size_t r = 1; r < rows.size(); ++r) {
    const TextRow &prev = rows[host >= 0 ? static_cast<size_t>(host) : r - 1];
    const int prev_boxes = prev.end - prev.begin;
    const int prev_last_x1 = spans[static_cast<size_t>(prev.end - 1)].x1;
    const int prev_first_x0 = spans[static_cast<size_t>(prev.begin)].x0;
    const int x0 = spans[static_cast<size_t>(rows[r].begin)].x0;
    const int x1 = spans[static_cast<size_t>(rows[r].end - 1)].x1;

    // A wrapped label continuation: one box, starting at the same left margin
    // as the row above, in a row of its own, on a row that DID carry a
    // right-hand value. Measured on CORD-v2 test_0017, whose item names wrap:
    // "Mozarella Hot Dog" splits into "Mozarella Hiot" (x0 277) + "Dog"
    // (x0 277), and "Chili Pepper Croquette" into two spans at x0 275.
    // Giving "Dog" its own line is honest but leaves nothing able to tell
    // whether it belongs to the item above or below.
    //
    // Prose cannot trip this: every line of a paragraph is a single box, so
    // the `prev_boxes >= 2` test never holds for the line above.
    // ATTEMPTED AND REVERTED 2026-09-05. Detecting a wrapped label
    // continuation geometrically — one box, same left margin as the row
    // above, that row having carried a right-hand value — reads test_0017
    // correctly ("Mozarella Hiot Dog 38,000") and reads test_0046 WRONGLY:
    // there the CHANGE line's amount was never detected, so a genuine label
    // row of its own looked exactly like a continuation and was glued to the
    // row above as "Cash: CHANGE 40,000".
    //
    // Geometry cannot separate the two cases: "a label whose value is
    // missing" and "the tail of a label that wrapped" have the same shape.
    // Measured cost of guessing: CORD-v2 line-item F1 52.4% -> 47.3%,
    // spurious items 32 -> 49, scalar exact 84.3% -> 81.3%. Leaving each row
    // on its own line is less informative but it is not wrong.
    (void)prev_first_x0;
    (void)x0;
    (void)x1;

    // Otherwise: a new row, unless the row above was a soft-wrapped line of
    // prose. "Previous line reached the right margin, therefore it wrapped"
    // is true of prose and true of EVERY row of every table — a receipt row
    // is `label ....  amount` and the amount reaches the right margin by
    // construction — so as an unconditional rule it flattened whole regions
    // into one line. A recogniser emits one box per text line, so only a
    // one-box row can be a wrapped line of prose.
    rows[r].join = (prev_boxes == 1 && prev_last_x1 >= cell_x1 - wrap_tol)
                       ? RowJoin::kSpace
                       : RowJoin::kNewline;
    host = static_cast<int>(r);
  }
  if (!rows.empty() && host < 0) host = 0;
  return rows;
}

}  // namespace hp::core
