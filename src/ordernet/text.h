#pragma once
// Python str semantics for the OrderNet port (indra_ocr is Python): UTF-8 <-> code points
// (a Python str indexes code points) and the character classes order_post.py / layout_post.py
// read — str.isspace (what str.strip() and re's \s strip / match), isdigit, isdecimal
// (re's \d), isalpha, isupper / islower of one character, and East Asian width W / F —
// from range tables generated with the reference's own Python (tools/ordernet/gen_unicode_py.py).

#include <string>
#include <string_view>

namespace hp::on::py {

// UTF-8 -> code points (invalid bytes -> U+FFFD) and back.
[[nodiscard]] std::u32string utf32(std::string_view s);
[[nodiscard]] std::string utf8(std::u32string_view s);

// One-character str predicates.
[[nodiscard]] bool isspace(char32_t c);
[[nodiscard]] bool isdigit(char32_t c);
[[nodiscard]] bool isdecimal(char32_t c);
[[nodiscard]] bool isalpha(char32_t c);
[[nodiscard]] bool isupper(char32_t c);
[[nodiscard]] bool islower(char32_t c);
// unicodedata.east_asian_width(c) in "WF".
[[nodiscard]] bool is_wide(char32_t c);

// str.strip() (no argument: Python whitespace).
[[nodiscard]] std::u32string_view strip(std::u32string_view s);
// str.rstrip(chars).
[[nodiscard]] std::u32string_view rstrip(std::u32string_view s, std::u32string_view chars);
// str.isdigit(): non-empty and every character a digit.
[[nodiscard]] bool isdigit(std::u32string_view s);
// Python `c in s` for one character.
[[nodiscard]] inline bool contains(std::u32string_view s, char32_t c) {
  return s.find(c) != std::u32string_view::npos;
}

// html.escape(s, quote=False): & < > only.
[[nodiscard]] std::string html_escape(std::string_view s);

}  // namespace hp::on::py
