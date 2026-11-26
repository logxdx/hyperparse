#pragma once
// tex_math internals shared by tex_math.cpp, structures.cpp, encodings.cpp and
// collect.cpp. Ported from LiteParse's tex_math (Rust). Arithmetic is f32 like
// the original (float literals carry an f suffix: a double literal would widen
// the comparison), and every sort is stable with Rust's f32::total_cmp order.

#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tex_math.h"

namespace hp::tex::detail {

// The fixed encoding a TeX math font follows.
enum class Enc : uint8_t { None, Oml, Oms, Omx, Msam };
enum class Role : uint8_t { None, Mi, Sy, Bb, Ex, Frak, Scr, Cal };

// LaTeX for the raw char code of a glyph pdfium gave no Unicode for, in a font
// following `enc`; nullopt when the encoding has no entry (encodings.cpp).
std::optional<std::string_view> decode(Enc enc, uint32_t code);

// One page glyph, as collected by collect.cpp. Coordinates are viewport space
// (y down); origin_y is the baseline.
struct RawChar {
  int index = 0;
  // What pdfium reported: a clean Unicode char, or (is_code) no Unicode for the
  // glyph (pdfium's map error): the raw char code.
  bool is_code = false;
  uint32_t value = 0;
  const std::string *font_name = nullptr;  // font base name as in the PDF
  float size = 0;                          // effective size (font size x text-matrix scale)
  float descent = 0;                       // font descent at `size` (negative, points)
  float origin_y = 0, x0 = 0, x1 = 0;
  bool has_ink = false;
  float ink_top = 0, ink_bottom = 0;      // tight glyph box, viewport y
  float loose_top = 0, loose_bottom = 0;  // font box: the box text items use
};

// A thin horizontal rule. `y` is y-up page space (page height - y).
struct Rule {
  float x0 = 0, x1 = 0, y = 0, w = 0;
  // A rule from a viewport-space box (y down), when the box is thin and wide enough to be one.
  static std::optional<Rule> from_box(float x0, float x1, float top, float bottom, float page_h) {
    if (!(bottom - top <= 1.2f && x1 - x0 >= 1.5f)) return std::nullopt;
    return Rule{x0, x1, page_h - (top + bottom) / 2.0f, x1 - x0};
  }
};

// One glyph in the math pass. top / bottom are viewport y (down), base is the
// baseline in y-up page space (page height - y), as in the prototype.
struct G {
  std::string t;
  Role role = Role::None;
  Enc enc = Enc::None;  // font_info(fam).second
  std::string fam;
  float size = 0, base = 0, x0 = 0, x1 = 0, top = 0, bottom = 0;
  float loose_top = 0, loose_bottom = 0;  // font box, for the span's item box
  std::string pre, post;                  // accent commands opened before / closed after this glyph
  std::vector<int> src;                   // source char indices (several for a folded structure)
  float cx() const { return (x0 + x1) / 2.0f; }
};

using Idx = std::vector<int>;

// Prefix marking a glyph text that is already LaTeX (decoded from a TeX
// encoding, or a folded structure).
inline constexpr char PRE = '\0';
// Family of a folded structure's stand-in glyph.
inline constexpr std::string_view STAND_IN_FAM = "SYN";

// Rust f32::total_cmp as a strict weak order.
inline int32_t total_key(float f) {
  const int32_t i = std::bit_cast<int32_t>(f);
  return i ^ (int32_t)((uint32_t)(i >> 31) >> 1);
}
inline bool total_less(float a, float b) { return total_key(a) < total_key(b); }

// Python statistics.median (sorted with total_cmp).
float median(std::vector<float> v);
float max_size(const std::vector<G> &gs, const Idx &idx);
bool is_alpha(std::string_view t);  // non-empty, every char Unicode Alphabetic (Rust char::is_alphabetic)
bool is_operator(std::string_view w);
bool is_tex_math_font(std::string_view font_name);

Idx attach_accents(std::vector<G> &all, const Idx &sp);
std::string build(const std::vector<G> &all, const Idx &gs, float base, float size);

// structures.cpp: fold fractions, square roots and overlines into stand-in
// glyphs. Returns the live glyph list; stand-ins are appended to `all`.
Idx fold(std::vector<G> &all, Idx live, const std::vector<Rule> &rules, float h);

// Math spans of one page, or nullopt when the page isn't set in TeX.
std::optional<std::vector<MathSpan>> page_math(const std::vector<RawChar> &chars, const std::vector<Rule> &rules,
                                               float page_h);

// UTF-8 helpers.
std::u32string utf8_to_u32(std::string_view s);
void append_utf8(std::string &s, char32_t c);

}  // namespace hp::tex::detail
