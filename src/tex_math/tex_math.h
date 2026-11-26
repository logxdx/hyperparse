#pragma once
// LaTeX from the glyphs of born-digital TeX PDF pages, without a model.
//
// C++ port of LiteParse's tex_math module (crates/liteparse/src/tex_math/{mod,
// encodings,structures}.rs and its caller page_math_spans / tex_math::emitted in
// extract.rs; Apache-2.0). Math glyphs are recognised by their font (Computer
// Modern / Latin Modern / tx / px / AMS / Euler / rsfs math fonts); glyphs with no
// Unicode are read through TeX's fixed encodings (OML / OMS / OMX / msam);
// sub/superscripts come from glyph size and baseline shift; fractions, overlines
// and square roots come from thin rules (paths, and 1x1 images stretched into
// rules). Thresholds and behaviour are LiteParse's; comments marked "Note:"
// carry its traps.
//
// Only TeX pages get math: a page needs a TeX-encoded math font (or an AMS /
// Euler / rsfs alphabet) before anything is emitted, and that font check runs
// first, so other pages cost one pass over pdfium's text objects.

#include <string>
#include <vector>

namespace hp::tex {

struct MathSpan {
  std::string tex;  // LaTeX body without delimiters, e.g. "x_{i}^{2}"; LiteParse emits "$" + tex + "$"
  // Item box, viewport points (origin top-left, y down; the convention of hp's
  // text layer, RectF in text_layer.h). x0/x1 span every glyph of the span;
  // top/bottom are the font (loose) box of the glyphs on the span's baseline
  // (else of its line), so the item sits in its text line like a word does.
  float x0 = 0, x1 = 0, top = 0, bottom = 0;
  float size = 0;          // font size of the span's base glyphs (LiteParse font_size / font_height)
  std::vector<int> chars;  // pdfium text-page char indices the span replaces (sorted, unique)
  // Not a LiteParse notion (it emits every span inline, as "$...$"): true when
  // nothing but math glyphs sits on the span's line within 2 em on either side,
  // i.e. the formula stands on its own line (display math). Informational.
  bool display = false;
};

// Math spans LiteParse would emit on this page (tex_math::page_math filtered by
// tex_math::emitted: spans with scripts, fractions, roots, accents or a math
// alphabet; a lone "\pm" or Greek letter stays text). Empty when the page has no
// TeX math font. In LiteParse reading order is not decided here: the spans are
// appended after the page's text items and placed by the layout pass.
//
// fpdf_page / fpdf_textpage: FPDF_PAGE and its FPDF_TEXTPAGE. Must run where pdfium
// is serialised (same rules as text_layer.h). keep_clipped_text mirrors LiteParse's
// --keep-clipped-text (default off: glyphs hidden by rectangular clips are not
// math). all_spans = true returns every span page_math found, emitted or not
// (debugging / parity).
std::vector<MathSpan> page_math(void *fpdf_page, void *fpdf_textpage, bool keep_clipped_text = false,
                                bool all_spans = false);

// LiteParse MathSpan::has_structure: whether a span is emitted.
bool has_structure(const std::string &tex);

}  // namespace hp::tex
