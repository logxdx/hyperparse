#pragma once
// UTF-8 <-> code points and the Unicode character classes LayoutNet's features and
// decoder read, matching Rust's `char` methods (liteparse table_model is written in
// Rust): is_alphabetic, is_uppercase / is_lowercase, is_whitespace, is_numeric,
// is_alphanumeric, from range tables generated with Rust itself (unicode_tables.inc).

#include <string>
#include <string_view>
#include <vector>

namespace hp::tn {

[[nodiscard]] std::u32string utf32(std::string_view s);  // invalid bytes -> U+FFFD
[[nodiscard]] std::string utf8(std::u32string_view s);
void append_utf8(std::string &out, char32_t c);

[[nodiscard]] bool is_whitespace(char32_t c);  // Unicode White_Space
[[nodiscard]] bool is_alphabetic(char32_t c);
[[nodiscard]] bool is_uppercase(char32_t c);
[[nodiscard]] bool is_lowercase(char32_t c);
[[nodiscard]] bool is_numeric(char32_t c);  // general category N* (Nd, Nl, No)
[[nodiscard]] inline bool is_alphanumeric(char32_t c) { return is_alphabetic(c) || is_numeric(c); }
[[nodiscard]] inline bool is_ascii_digit(char32_t c) { return c >= U'0' && c <= U'9'; }
[[nodiscard]] inline bool is_ascii_alpha(char32_t c) { return (c | 0x20) >= U'a' && (c | 0x20) <= U'z'; }

// Number of code points (Rust `s.chars().count()`).
[[nodiscard]] size_t char_count(std::string_view s);
// Rust `s.split_whitespace()` (Unicode White_Space separators, empty pieces dropped).
[[nodiscard]] std::vector<std::string> split_whitespace(std::string_view s);
// Rust `s.trim()` / `trim_start()` / `trim_end()` (Unicode White_Space).
[[nodiscard]] std::string_view trim(std::string_view s);
[[nodiscard]] std::string_view trim_start(std::string_view s);
[[nodiscard]] std::string_view trim_end(std::string_view s);

// Rust f64::round_ties_even / f32::round (half away from zero) / total_cmp-style helpers.
[[nodiscard]] double round_ties_even(double v);

}  // namespace hp::tn
