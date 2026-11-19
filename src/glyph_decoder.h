#pragma once
// Glyph-level text recovery for chars whose PDF text mapping is missing or
// untrustworthy (no /ToUnicode with a custom encoding, buggy subset fonts, garbage
// maps). C++ port of liteparse's GlyphDecoder (crates/liteparse/src/extract.rs,
// glyph_names.rs, font_cmap.rs; Apache-2.0):
//   1. glyph name (PDF /Differences or the embedded program) -> Adobe Glyph List
//   2. glyph index -> the embedded sfnt's own unicode cmap, inverted
// Needs run-llama's pdfium fork (fetched by CMakeLists.txt): FPDFText_GetCharCode,
// FPDFFont_GetCharGlyphName / GetCharGlyphIndex / HasToUnicode / GetEncoding / GetType.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <fpdf_text.h>
#include <fpdfview.h>

namespace hp {

// AGL reverse mapping: strip ".suffix", split "_", per component uniXXXX(XXXX..),
// uXXXX[XX], or table lookup; presentation-form ligatures expanded to ASCII.
std::optional<std::string> resolve_glyph_name(std::string_view name);

// Mathematical Alphanumeric Symbols (U+1D400-1D7FF, U+210E) -> plain Latin /
// Greek / digits, so recovered math variables stay searchable ("𝑧" -> "z").
uint32_t fold_math_alnum(uint32_t c);

// glyph id -> unicode from an embedded sfnt's best unicode cmap subtable.
bool reverse_cmap(const uint8_t *data, size_t len, std::unordered_map<uint32_t, uint32_t> &out);

// decode() result for a visible glyph with no recoverable text (only for chars
// whose ToUnicode claims "space"); the caller reads that span from pixels.
inline constexpr std::string_view kUnknownGlyph = "\x01";

class GlyphDecoder {
 public:
  // Run once per page before decode(): flags fonts whose mapped text is >= 10%
  // suspicious codepoints (liteparse detect_garbage_unicode_fonts).
  void prescan(FPDF_TEXTPAGE tp, int n_chars);

  // Recovered UTF-8 text for char `index`, or nullptr when pdfium's unicode stands.
  const std::string *decode(FPDF_TEXTPAGE tp, int index, FPDF_PAGEOBJECT text_obj, uint32_t unicode,
                            bool map_error);

  int recovered = 0;  // stats

 private:
  struct FontInfo {
    bool untrusted = false, encoding_lies = false;
    bool cmap_built = false, cmap_ok = false;
    std::unordered_map<uint32_t, uint32_t> rcmap;
    std::unordered_map<uint32_t, std::optional<std::string>> cache;
  };
  std::unordered_map<void *, FontInfo> fonts_;
  std::unordered_set<void *> garbage_;
  FPDF_PAGEOBJECT last_obj_ = nullptr;
  FontInfo *last_info_ = nullptr;
  void *last_font_ = nullptr;
};

}  // namespace hp
