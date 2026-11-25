#include "ordernet/text.h"

#include <algorithm>

namespace hp::on::py {
namespace {

struct Range {
  char32_t lo, hi;
};

#include "ordernet/unicode_py.inc"

template <size_t N>
bool in(const Range (&t)[N], char32_t c) {
  // First range whose end is >= c.
  const Range *r = std::lower_bound(t, t + N, c, [](const Range &a, char32_t v) { return a.hi < v; });
  return r != t + N && r->lo <= c;
}

}  // namespace

std::u32string utf32(std::string_view s) {
  std::u32string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char b = static_cast<unsigned char>(s[i]);
    int len = b < 0x80 ? 1 : (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xE ? 3 : (b >> 3) == 0x1E ? 4 : 0;
    char32_t c = len == 1 ? b : len == 2 ? (b & 0x1F) : len == 3 ? (b & 0x0F) : (b & 0x07);
    bool ok = len > 0 && i + len <= s.size();
    for (int k = 1; ok && k < len; ++k) {
      const unsigned char cb = static_cast<unsigned char>(s[i + k]);
      if ((cb & 0xC0) != 0x80) ok = false;
      c = (c << 6) | (cb & 0x3F);
    }
    // Overlong forms, surrogates and values past U+10FFFF are invalid too.
    if (ok && ((len == 2 && c < 0x80) || (len == 3 && c < 0x800) || (len == 4 && c < 0x10000) ||
               (c >= 0xD800 && c <= 0xDFFF) || c > 0x10FFFF))
      ok = false;
    if (ok) {
      out.push_back(c);
      i += len;
    } else {
      out.push_back(0xFFFD);
      ++i;
    }
  }
  return out;
}

std::string utf8(std::u32string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char32_t c : s) {
    if (c < 0x80) {
      out.push_back(static_cast<char>(c));
    } else if (c < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (c >> 6)));
      out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (c >> 12)));
      out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (c >> 18)));
      out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
  }
  return out;
}

bool isspace(char32_t c) { return in(kSpace, c); }
bool isdigit(char32_t c) { return in(kDigit, c); }
bool isdecimal(char32_t c) { return in(kDecimal, c); }
bool isalpha(char32_t c) { return in(kAlpha, c); }
bool isupper(char32_t c) { return in(kUpper, c); }
bool islower(char32_t c) { return in(kLower, c); }
bool is_wide(char32_t c) { return in(kWide, c); }

std::u32string_view strip(std::u32string_view s) {
  size_t b = 0, e = s.size();
  while (b < e && isspace(s[b])) ++b;
  while (e > b && isspace(s[e - 1])) --e;
  return s.substr(b, e - b);
}

std::u32string_view rstrip(std::u32string_view s, std::u32string_view chars) {
  size_t e = s.size();
  while (e > 0 && contains(chars, s[e - 1])) --e;
  return s.substr(0, e);
}

bool isdigit(std::u32string_view s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](char32_t c) { return isdigit(c); });
}

std::string html_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      default: out.push_back(c);
    }
  }
  return out;
}

}  // namespace hp::on::py
