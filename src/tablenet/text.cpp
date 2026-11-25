#include "tablenet/text.h"

#include <algorithm>
#include <cmath>

namespace hp::tn {
namespace {

struct Range {
  char32_t lo, hi;
};
// Rust's char classes (rustc 1.99.1): a small Rust program walked every code point and printed
// the inclusive ranges where char::is_alphabetic / is_uppercase / is_lowercase / is_numeric /
// is_whitespace hold. glibc's tables differ (non-ASCII digits are "alpha" there, for one), and
// LayoutNet's alpha-fraction feature notices.
#include "tablenet/unicode_tables.inc"

template <size_t N>
bool in(const Range (&t)[N], char32_t c) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (c < t[mid].lo) hi = mid;
    else if (c > t[mid].hi) lo = mid + 1;
    else return true;
  }
  return false;
}

}  // namespace

std::u32string utf32(std::string_view s) {
  std::u32string out;
  out.reserve(s.size());
  size_t i = 0;
  const auto *p = reinterpret_cast<const unsigned char *>(s.data());
  const size_t n = s.size();
  while (i < n) {
    const unsigned char c = p[i];
    char32_t cp;
    size_t len;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
    else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
    else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
    else { out.push_back(0xFFFD); ++i; continue; }
    if (i + len > n) { out.push_back(0xFFFD); break; }
    bool ok = true;
    for (size_t k = 1; k < len; ++k) {
      if ((p[i + k] >> 6) != 0x2) { ok = false; break; }
      cp = (cp << 6) | (p[i + k] & 0x3F);
    }
    if (!ok) { out.push_back(0xFFFD); ++i; continue; }
    out.push_back(cp);
    i += len;
  }
  return out;
}

void append_utf8(std::string &out, char32_t c) {
  if (c < 0x80) out.push_back(static_cast<char>(c));
  else if (c < 0x800) {
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

std::string utf8(std::u32string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char32_t c : s) append_utf8(out, c);
  return out;
}

bool is_whitespace(char32_t c) { return in(kWhitespace, c); }
bool is_alphabetic(char32_t c) { return in(kAlphabetic, c); }
bool is_uppercase(char32_t c) { return in(kUppercase, c); }
bool is_lowercase(char32_t c) { return in(kLowercase, c); }
bool is_numeric(char32_t c) { return in(kNumeric, c); }

size_t char_count(std::string_view s) {
  size_t n = 0;
  for (unsigned char c : s) n += (c & 0xC0) != 0x80;
  return n;
}

std::vector<std::string> split_whitespace(std::string_view s) {
  std::vector<std::string> out;
  std::string cur;
  for (char32_t c : utf32(s)) {
    if (is_whitespace(c)) {
      if (!cur.empty()) out.push_back(std::move(cur)), cur.clear();
    } else {
      append_utf8(cur, c);
    }
  }
  if (!cur.empty()) out.push_back(std::move(cur));
  return out;
}

namespace {
// Byte length of the code point starting at s[i] / ending before s[j].
size_t cp_len(std::string_view s, size_t i) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
  return std::min(len, s.size() - i);
}
char32_t cp_at(std::string_view s, size_t i, size_t len) {
  auto u = utf32(s.substr(i, len));
  return u.empty() ? 0xFFFD : u[0];
}
}  // namespace

std::string_view trim_start(std::string_view s) {
  size_t i = 0;
  while (i < s.size()) {
    const size_t len = cp_len(s, i);
    if (!is_whitespace(cp_at(s, i, len))) break;
    i += len;
  }
  return s.substr(i);
}

std::string_view trim_end(std::string_view s) {
  size_t j = s.size();
  while (j > 0) {
    size_t i = j - 1;
    while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) --i;
    if (!is_whitespace(cp_at(s, i, j - i))) break;
    j = i;
  }
  return s.substr(0, j);
}

std::string_view trim(std::string_view s) { return trim_end(trim_start(s)); }

double round_ties_even(double v) { return std::nearbyint(v); }  // default FE_TONEAREST

}  // namespace hp::tn
