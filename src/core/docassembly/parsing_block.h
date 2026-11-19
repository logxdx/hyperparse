#pragma once

#include <array>
#include <string>
#include <string_view>

namespace hp::core::docassembly {

// Provenance values. Named so the merge helpers and their tests cannot drift
// apart on a string literal.
inline constexpr std::string_view kSourceOcr = "ocr";
inline constexpr std::string_view kSourceMixed = "mixed";
// A region that carries no text at all — a figure, photograph or chart. Its
// existence and confidence come from the layout detector rather than from
// recognition, and `content` is necessarily empty. Tagged distinctly so a
// consumer never mistakes a layout score for a recognition score.
inline constexpr std::string_view kSourceLayout = "layout";

// CUDA-free, per-region parsing block — the unit the document-assembly
// stage operates on. Mirrors PaddleX's PaddleOCRVLBlock so the ported
// cross-page / title-level / repetition logic stays faithful, but carries
// only the fields those algorithms actually read.
//
// `bbox` is axis-aligned [x0, y0, x1, y1] in page pixels (use hp::core::aabb
// to derive it from a rotated Box). `content` is the recognized payload:
// plain text for text-bound labels, an HTML table for `table`, LaTeX for
// formulas. `label` is a PP-DocLayoutV3 class name (see layout_types.h).
struct ParsingBlock {
  std::string label;
  std::array<int, 4> bbox{0, 0, 0, 0};
  std::string content;

  int page_index = 0;

  // Identity used by cross-page table merge: every block gets a unique
  // global_block_id; global_group_id points at the block it was folded
  // into (== global_block_id when it stands alone). Mirrors PaddleX.
  int global_block_id = 0;
  int global_group_id = 0;

  // Heading depth assigned by title leveling. 0 == doc_title, 1..4 ==
  // paragraph_title depth, -1 == not a heading / unset.
  int title_level = -1;

  // --- provenance, carried through assembly ---
  //
  // `OCRResultItem` already knows a region's confidence and where its text
  // came from, but that was lost the moment items were folded into blocks.
  // Downstream routing needs it on the block: "is this region trustworthy,
  // and did a model read it or did we take the PDF's word for it" is what
  // decides whether a region gets escalated to a stronger (and far more
  // expensive) reader.
  //
  // Recognition confidence for the whole block, in [0, 1].
  // -1 == unset, matching the `-1`-means-unset idiom used by `title_level`
  // and `OCRResultItem::layout_id`. A sentinel rather than 0.0f because
  // 0.0f is a legitimate score meaning "read it, trust it not at all",
  // which must not be confused with "never scored".
  float confidence = -1.0f;

  // Where `content` came from. Mirrors `OCRResultItem::source`, with the
  // same convention that empty means the implicit default ("ocr").
  // Values: "" (implicit ocr) | "pdf" (PDFium text layer) | "geometric"
  // | "mixed" (set by cross-page merge when a host and its follower
  // disagree — a table straddling a mode boundary, which is exactly the
  // sort of region worth reviewing) | "layout" (a text-free region such as
  // a figure; `confidence` is then the layout detector's score, not a
  // recognition score, and `content` is empty).
  std::string source;

  // Fraction of this region's area that detection actually covered, in
  // [0, 1]. -1 == not applicable (a region the layout detector does not
  // expect text in, e.g. an image or a chart).
  //
  // This exists because `confidence` is blind to the failure that matters
  // most. Confidence is a mean over the text that WAS read, so a region whose
  // text was never detected has nothing to average and reports a perfectly
  // ordinary score.
  //
  // The two signals separate the two ways a region fails, and neither
  // substitutes for the other. Measured on the local fixtures, area-weighted
  // per page (recognition confidence / page coverage / worst block):
  //
  //   healthy      book_page      0.981 / 0.768 / 0.727
  //                business_letter 0.978 / 0.861 / 0.744
  //                receipt        0.976 / 0.832 / 0.563
  //   misread      handwritten    0.768 / 0.851 / 0.712  <- coverage is FINE
  //   never read   dense_text     0.706 / 0.307 / 0.027  <- confidence is fine
  //                whiteboard     0.839 / 0.148 / 0.148
  //
  // The handwritten note is the case that proves the pair: its text was found
  // and attempted everywhere (coverage 0.851, as healthy as a clean letter),
  // and only confidence reveals the glyphs were wrong. `dense_text` is the
  // mirror image — 0.706 is unremarkable next to a merely-difficult page,
  // while coverage collapses.
  //
  // A healthy region sits around 0.55-0.90 — never 1.0, because line boxes do
  // not tile a paragraph's bounding box (leading, indents and ragged right
  // margins are real gaps). So this is a *relative* signal: compare against
  // the page and against peers, and treat a collapse rather than an absolute
  // value as the escalation trigger. On these fixtures every healthy page sits
  // above 0.76 and every failing one below 0.68, but that gap is not a
  // promise — calibrate on your own traffic.
  float coverage = -1.0f;
};

// How `confidence` combines when cross-page merge folds `follower` into
// `host`: the host ends up representing both halves, so it can only be as
// trustworthy as its weaker half. Deliberately the minimum rather than a
// length-weighted mean — a mean lets a short, badly-read continuation hide
// behind a long, well-read host, which defeats the point of the signal.
// An unset (-1) operand is ignored rather than treated as zero.
[[nodiscard]] inline float merge_confidence(float host, float follower) noexcept {
  if (host < 0.0f) return follower;
  if (follower < 0.0f) return host;
  return host < follower ? host : follower;
}

// How `source` combines on the same merge. Equal sources survive; anything
// else becomes "mixed". Empty is normalized to "ocr" first so that an
// implicit-default block merging with an explicit "ocr" block is not
// mislabelled as mixed.
[[nodiscard]] inline std::string merge_source(const std::string &host,
                                              const std::string &follower) {
  const std::string_view a = host.empty() ? kSourceOcr : std::string_view(host);
  const std::string_view b =
      follower.empty() ? kSourceOcr : std::string_view(follower);
  return a == b ? std::string(a) : std::string(kSourceMixed);
}

} // namespace hp::core::docassembly
