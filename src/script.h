#pragma once
// Script checks for OCR decoding. PP-OCRv6's dictionary is 86% CJK (16k of its
// 18.7k classes), so when it reads something that is not a word, such as a lone
// math glyph, a fraction bar or a bracket, the nearest class is often a CJK
// character (a bar reads as 一, a ≠ as 手). On a page whose own text layer
// is clearly not CJK, the recognizer's argmax is therefore restricted to non-CJK
// classes; the right symbol (−, ≠, …) is in the dictionary too.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hp {

// Han (incl. extensions and compatibility ideographs), kana, hangul, CJK
// symbols and punctuation, and fullwidth/halfwidth forms.
constexpr bool is_cjk(char32_t c) {
  return (c >= 0x2E80 && c <= 0x9FFF) || (c >= 0xA960 && c <= 0xA97F) || (c >= 0xAC00 && c <= 0xD7FF) ||
         (c >= 0xF900 && c <= 0xFAFF) || (c >= 0xFE30 && c <= 0xFE4F) || (c >= 0xFF00 && c <= 0xFFEF) ||
         (c >= 0x20000 && c <= 0x3FFFF);
}

// Decodes the UTF-8 code point at s[i] and advances i; malformed bytes give U+FFFD.
inline char32_t next_codepoint(std::string_view s, size_t &i) {
  const auto b = static_cast<unsigned char>(s[i]);
  const int n = b < 0x80 ? 1 : (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xE ? 3 : (b >> 3) == 0x1E ? 4 : 0;
  if (n == 0 || i + n > s.size()) {
    ++i;
    return 0xFFFD;
  }
  char32_t c = n == 1 ? b : b & (0x7F >> n);
  for (int k = 1; k < n; ++k) c = (c << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  i += n;
  return c;
}

inline bool has_cjk(std::string_view s) {
  for (size_t i = 0; i < s.size();)
    if (is_cjk(next_codepoint(s, i))) return true;
  return false;
}

// Recognizer class mask for non-CJK pages: 1 = allowed. Class 0 is the CTC blank.
inline std::vector<uint8_t> non_cjk_class_mask(const std::vector<std::string> &labels) {
  std::vector<uint8_t> m(labels.size(), 1);
  for (size_t k = 1; k < labels.size(); ++k) m[k] = has_cjk(labels[k]) ? 0 : 1;
  return m;
}

}  // namespace hp
