#include "glyph_decoder.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>

#include <fpdf_edit.h>

#include "agl_table.h"

namespace hp {
namespace {

void put_utf8(std::string &s, uint32_t c) {
  if (c < 0x80) s += (char)c;
  else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
  else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
  else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
}

bool valid_scalar(uint32_t v) { return v <= 0x10FFFF && !(v >= 0xD800 && v <= 0xDFFF); }
bool is_control(uint32_t c) { return c < 0x20 || (c >= 0x7F && c < 0xA0); }

const char *presentation_form(uint32_t c) {
  switch (c) {
    case 0xFB00: return "ff";
    case 0xFB01: return "fi";
    case 0xFB02: return "fl";
    case 0xFB03: return "ffi";
    case 0xFB04: return "ffl";
    case 0xFB05: case 0xFB06: return "st";
    default: return nullptr;
  }
}

}  // namespace

uint32_t fold_math_alnum(uint32_t c) {
  if (c >= 0x1D400 && c <= 0x1D6A3) {  // 13 Latin styles x 52 (A-Z a-z)
    const uint32_t i = (c - 0x1D400) % 52;
    return i < 26 ? 'A' + i : 'a' + (i - 26);
  }
  if (c >= 0x1D6A8 && c <= 0x1D7C9) {  // 5 Greek styles x 58
    const uint32_t i = (c - 0x1D6A8) % 58;
    if (i < 25) return i == 17 ? 0x3F4 : 0x391 + i;
    if (i == 25) return 0x2207;
    if (i < 51) return 0x3B1 + (i - 26);
    static const uint32_t extra[] = {0x2202, 0x3F5, 0x3D1, 0x3F0, 0x3D5, 0x3F1, 0x3D6};
    return extra[i - 51];
  }
  if (c >= 0x1D7CE && c <= 0x1D7FF) return '0' + (c - 0x1D7CE) % 10;
  if (c == 0x210E) return 'h';  // planck constant: italic h hole in the italic block
  return c;
}

namespace {

void push_cp(std::string &s, uint32_t c) {
  c = fold_math_alnum(c);
  if (const char *e = presentation_form(c)) s += e;
  else put_utf8(s, c);
}

bool agl_hex(std::string_view h) {  // uppercase hex only (AGL convention)
  return !h.empty() && std::all_of(h.begin(), h.end(), [](char b) { return (b >= '0' && b <= '9') || (b >= 'A' && b <= 'F'); });
}

// UTF-8 -> codepoints (table values are short and valid).
std::vector<uint32_t> cps(std::string_view s) {
  std::vector<uint32_t> out;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i];
    const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
    out.push_back(cp);
    i += n;
  }
  return out;
}

bool resolve_component(std::string_view comp, std::string &out) {
  if (comp.empty()) return false;
  if (comp.size() > 3 && comp.substr(0, 3) == "uni") {
    const auto hex = comp.substr(3);
    if (hex.size() % 4 == 0 && agl_hex(hex)) {
      for (size_t k = 0; k < hex.size(); k += 4) {
        const uint32_t v = std::stoul(std::string(hex.substr(k, 4)), nullptr, 16);
        if (!valid_scalar(v)) return false;
        push_cp(out, v);
      }
      return true;
    }
  }
  if (comp.size() >= 5 && comp.size() <= 7 && comp[0] == 'u' && agl_hex(comp.substr(1))) {
    const uint32_t v = std::stoul(std::string(comp.substr(1)), nullptr, 16);
    if (!valid_scalar(v)) return false;
    push_cp(out, v);
    return true;
  }
  const auto *b = std::begin(agl::kTable), *e = std::end(agl::kTable);
  const auto *it = std::lower_bound(b, e, comp, [](const agl::Entry &x, std::string_view k) { return x.name < k; });
  if (it == e || it->name != comp) return false;
  for (uint32_t c : cps(it->text)) push_cp(out, c);
  return true;
}

uint16_t be16(const uint8_t *p) { return (uint16_t)(p[0] << 8 | p[1]); }
uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

bool is_buggy_font(const std::string &name, int type) {  // extract.rs is_buggy_font
  if (name.rfind("TT", 0) == 0 || name.find("+TT") != std::string::npos) return true;
  return type == FPDF_FONTTYPE_TYPE1 && name.size() >= 7 && name[6] == '_';
}

bool suspicious(uint32_t u) { return u == 0 || u == 0xFFFE || u == 0xFFFF || u < 0x20 || (u >= 0xE000 && u <= 0xF8FF); }

}  // namespace

std::optional<std::string> resolve_glyph_name(std::string_view name) {
  const auto dot = name.find('.');
  const std::string_view base = name.substr(0, dot);
  if (base.empty()) return std::nullopt;
  std::string out;
  size_t start = 0;
  while (true) {
    const size_t us = base.find('_', start);
    if (!resolve_component(base.substr(start, us == std::string_view::npos ? std::string_view::npos : us - start), out))
      return std::nullopt;
    if (us == std::string_view::npos) break;
    start = us + 1;
  }
  if (out.empty()) return std::nullopt;
  return out;
}

bool reverse_cmap(const uint8_t *d, size_t n, std::unordered_map<uint32_t, uint32_t> &map) {
  auto sfnt = [](const uint8_t *p) {
    return (p[0] == 0 && p[1] == 1 && p[2] == 0 && p[3] == 0) || std::memcmp(p, "OTTO", 4) == 0 ||
           std::memcmp(p, "true", 4) == 0;
  };
  if (n < 12) return false;
  if (std::memcmp(d, "ttcf", 4) == 0) {
    const uint32_t first = be32(d + 12);
    if (n < 16 || first + 12 > n || !sfnt(d + first)) return false;
    d += first;
    n -= first;
  } else if (!sfnt(d)) {
    return false;
  }
  // find 'cmap'
  const uint16_t num_tables = be16(d + 4);
  const uint8_t *cmap = nullptr;
  size_t cmap_len = 0;
  for (uint16_t i = 0; i < num_tables; ++i) {
    const size_t rec = 12 + i * 16;
    if (rec + 16 > n) return false;
    if (std::memcmp(d + rec, "cmap", 4) == 0) {
      const uint32_t off = be32(d + rec + 8), len = be32(d + rec + 12);
      if ((size_t)off + len > n) return false;
      cmap = d + off;
      cmap_len = len;
      break;
    }
  }
  if (!cmap || cmap_len < 4) return false;
  auto rd16 = [&](size_t o, uint16_t &v) { if (o + 2 > cmap_len) return false; v = be16(cmap + o); return true; };
  auto rd32 = [&](size_t o, uint32_t &v) { if (o + 4 > cmap_len) return false; v = be32(cmap + o); return true; };
  uint16_t nsub;
  if (!rd16(2, nsub)) return false;
  int best_score = 0;
  uint32_t best_off = 0;
  for (uint16_t i = 0; i < nsub; ++i) {
    uint16_t plat, enc;
    uint32_t off;
    if (!rd16(4 + i * 8, plat) || !rd16(6 + i * 8, enc) || !rd32(8 + i * 8, off)) return false;
    int score = 0;
    if ((plat == 3 && enc == 10) || (plat == 0 && (enc == 4 || enc == 6))) score = 4;
    else if ((plat == 3 && enc == 1) || (plat == 0 && enc <= 3)) score = 3;
    if (score > best_score) { best_score = score; best_off = off; }
  }
  if (!best_score) return false;
  auto rank = [](uint32_t u) { return (u >= 0xE000 && u <= 0xF8FF) ? 2 : (u >= 0x2E80 && u <= 0x2FDF) ? 1 : 0; };
  auto add = [&](uint32_t g, uint32_t u) {
    if (!g || !u) return;
    auto [it, fresh] = map.try_emplace(g, u);
    if (!fresh && std::pair(rank(u), u) < std::pair(rank(it->second), it->second)) it->second = u;
  };
  const size_t s = best_off;
  uint16_t fmt;
  if (!rd16(s, fmt)) return false;
  if (fmt == 0) {
    if (s + 6 + 256 > cmap_len) return false;
    for (uint32_t c = 0; c < 256; ++c) add(cmap[s + 6 + c], c);
  } else if (fmt == 4) {
    uint16_t x2;
    if (!rd16(s + 6, x2)) return false;
    const size_t seg = x2 / 2, ends = s + 14, starts = ends + seg * 2 + 2, deltas = starts + seg * 2,
                 ranges = deltas + seg * 2;
    for (size_t k = 0; k < seg; ++k) {
      uint16_t end, start, delta, ro;
      if (!rd16(ends + k * 2, end) || !rd16(starts + k * 2, start) || !rd16(deltas + k * 2, delta) ||
          !rd16(ranges + k * 2, ro))
        return false;
      if (start == 0xFFFF && end == 0xFFFF) continue;
      for (uint32_t c = start; c <= std::min<uint32_t>(end, 0xFFFE); ++c) {
        uint32_t g;
        if (ro == 0) g = (c + delta) & 0xFFFF;
        else {
          uint16_t gg;
          if (!rd16(ranges + k * 2 + ro + (c - start) * 2, gg)) return false;
          g = gg == 0 ? 0 : (gg + delta) & 0xFFFF;
        }
        add(g, c);
      }
    }
  } else if (fmt == 6) {
    uint16_t first, count;
    if (!rd16(s + 6, first) || !rd16(s + 8, count)) return false;
    for (uint32_t k = 0; k < count; ++k) {
      uint16_t g;
      if (!rd16(s + 10 + k * 2, g)) return false;
      add(g, first + k);
    }
  } else if (fmt == 12) {
    uint32_t groups;
    if (!rd32(s + 12, groups)) return false;
    for (uint32_t k = 0; k < std::min<uint32_t>(groups, 100000); ++k) {
      uint32_t st, en, g0;
      if (!rd32(s + 16 + k * 12, st) || !rd32(s + 20 + k * 12, en) || !rd32(s + 24 + k * 12, g0)) return false;
      if (en < st || en - st > 0x10000) continue;
      for (uint32_t o = 0; o <= en - st; ++o) add(g0 + o, st + o);
    }
  } else {
    return false;
  }
  return !map.empty();
}

void GlyphDecoder::prescan(FPDF_TEXTPAGE tp, int n) {
  bool any = false;
  for (int i = 0; i < n && !any; ++i) {
    const uint32_t u = FPDFText_GetUnicode(tp, i);
    any = suspicious(u) && u != 0x09 && u != 0x0A && u != 0x0D && u != 0x20;
  }
  if (!any) return;
  std::map<void *, std::pair<int, int>> counts;  // font -> (total, suspicious)
  for (int i = 0; i < n; ++i) {
    if (FPDFText_IsGenerated(tp, i) == 1) continue;
    const uint32_t u = FPDFText_GetUnicode(tp, i);
    if (u == 0x09 || u == 0x0A || u == 0x0D || u == 0x20) continue;
    FPDF_PAGEOBJECT o = FPDFText_GetTextObject(tp, i);
    FPDF_FONT f = o ? FPDFTextObj_GetFont(o) : nullptr;
    if (!f) continue;
    auto &c = counts[f];
    ++c.first;
    c.second += suspicious(u);
  }
  for (const auto &[f, c] : counts)
    if (c.first >= 20 && c.second * 10 >= c.first) garbage_.insert(f);
}

const std::string *GlyphDecoder::decode(FPDF_TEXTPAGE tp, int index, FPDF_PAGEOBJECT obj, uint32_t unicode,
                                        bool map_error) {
  const bool cheap = unicode == 0 || unicode == 0xFFFE || unicode == 0xFFFF ||
                     (unicode < 0x20 && unicode != 0x09 && unicode != 0x0A && unicode != 0x0D) ||
                     (unicode >= 0xE000 && unicode <= 0xF8FF);
  if (!obj) return nullptr;
  if (obj != last_obj_) {
    FPDF_FONT f = FPDFTextObj_GetFont(obj);
    if (!f) return nullptr;
    auto [it, fresh] = fonts_.try_emplace(f);
    if (fresh) {
      char buf[256];
      const bool has_tu = FPDFFont_HasToUnicode(f);
      const unsigned long el = FPDFFont_GetEncoding(f, buf, sizeof buf);
      const std::string enc = el > 1 && el <= sizeof buf ? std::string(buf, el - 1) : std::string();
      const size_t nl = FPDFFont_GetBaseFontName(f, buf, sizeof buf);
      const std::string name = nl > 1 && nl <= sizeof buf ? std::string(buf, nl - 1) : std::string();
      const bool name_buggy = FPDFFont_GetIsEmbedded(f) == 1 && !name.empty() && is_buggy_font(name, FPDFFont_GetType(f));
      const bool std_enc = enc == "WinAnsiEncoding" || enc == "MacRomanEncoding" || enc == "MacExpertEncoding" ||
                           enc == "StandardEncoding";
      it->second.untrusted = garbage_.count(f) || name_buggy || (!has_tu && !std_enc);
      it->second.encoding_lies = name_buggy;
    }
    last_obj_ = obj;
    last_font_ = f;
    last_info_ = &it->second;
  }
  FontInfo &info = *last_info_;
  // hp extension (not in liteparse): a visible glyph whose ToUnicode says "space".
  // Word's Cambria Math export maps every math-variable glyph to U+0020, which
  // silently deletes variables ("update ( ) and reset ( )"). Try the font's own
  // cmap and accept only a visible result.
  const bool space_like = unicode == 0x20 || unicode == 0xA0;
  if (!info.untrusted && !cheap && !map_error && !space_like) return nullptr;

  const uint32_t code = FPDFText_GetCharCode(tp, index);
  auto [cit, fresh] = info.cache.try_emplace(code);
  if (fresh) {
    FPDF_FONT f = static_cast<FPDF_FONT>(last_font_);
    std::optional<std::string> r;
    if (!info.encoding_lies) {
      char nbuf[128];
      const unsigned long nl = FPDFFont_GetCharGlyphName(f, code, nbuf, sizeof nbuf);
      if (nl > 1 && nl <= sizeof nbuf) {
        r = resolve_glyph_name(std::string_view(nbuf, nl - 1));
        if (r && std::any_of(r->begin(), r->end(), [](char c) { return (unsigned char)c < 0x20 || c == 0x7F; }))
          r.reset();
      }
    }
    if (!r) {
      const int g = FPDFFont_GetCharGlyphIndex(f, code);
      if (g > 0) {
        if (!info.cmap_built) {
          info.cmap_built = true;
          size_t len = 0;
          if (FPDFFont_GetFontData(f, nullptr, 0, &len) && len > 0) {
            std::vector<uint8_t> data(len);
            if (FPDFFont_GetFontData(f, data.data(), len, &len)) info.cmap_ok = reverse_cmap(data.data(), len, info.rcmap);
          }
        }
        if (info.cmap_ok) {
          auto m = info.rcmap.find((uint32_t)g);
          // A "space" whose glyph the font's own cmap doesn't list is not a space
          // glyph (a real one maps to U+0020): unknown visible glyph, read from pixels.
          if (m == info.rcmap.end() && space_like) r = std::string(kUnknownGlyph);
          if (m != info.rcmap.end()) {
            const uint32_t u = m->second;
            if (!(u >= 0xE000 && u <= 0xF8FF) && !(u == code && u != unicode) && !is_control(u) && valid_scalar(u)) {
              std::string s;
              push_cp(s, u);
              r = s;
            }
          }
        }
      }
    }
    cit->second = std::move(r);
  }
  const auto &r = cit->second;
  if (!r) return nullptr;
  if (space_like && (r->empty() || *r == " " || *r == "\xC2\xA0")) return nullptr;
  if (!space_like && *r == kUnknownGlyph) return nullptr;
  // Don't double-expand a ligature pdfium already mapped (e.g. "fi" containing 'f').
  if (!cheap && cps(*r).size() > 1) {
    std::string u8;
    put_utf8(u8, unicode);
    if (r->find(u8) != std::string::npos) return nullptr;
  }
  ++recovered;
  return &*r;
}

}  // namespace hp
