#include "text_layer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cwctype>
#include <initializer_list>
#include <map>
#include <set>
#include <string_view>
#include <unordered_map>

#include <fpdf_edit.h>
#include <fpdf_text.h>
#include <fpdfview.h>

#include "glyph_decoder.h"
#include "tex_math/tex_math.h"

namespace hp {
namespace {

// ── liteparse constants (ocr_merge.rs / extract.rs) ─────────────────────────
constexpr int kMinNativeTextLen = 20;          // MIN_NATIVE_TEXT_LEN (UTF-8 bytes)
constexpr int kSparseTextTrustedLen = 500;     // SPARSE_TEXT_TRUSTED_LEN
constexpr float kMinImageSizePt = 25.f;        // MIN_IMAGE_SIZE_PT
constexpr float kFullPageFrac = 0.9f;          // MAX_IMAGE_PAGE_COVERAGE
constexpr float kUncoveredVectorArea = 400.f;  // UNCOVERED_VECTOR_AREA_THRESHOLD (pt^2)
constexpr float kInvisibleLayerMinShare = 0.9f;
constexpr float kYTol = 2.0f, kMaxInlineGap = 15.0f;
constexpr float kMaxRuleThickness = 2.0f;      // STRIKE_MAX_THICKNESS_PT: strike / underline line

// ── unicode helpers (glibc wide-char classes; locale set to C.UTF-8 by caller) ─
bool is_alpha(char32_t c) { return c < 128 ? std::isalpha((int)c) : std::iswalpha((wint_t)c); }
bool is_lower(char32_t c) { return c < 128 ? std::islower((int)c) : std::iswlower((wint_t)c); }
bool is_upper(char32_t c) { return c < 128 ? std::isupper((int)c) : std::iswupper((wint_t)c); }
bool is_alnum(char32_t c) { return c < 128 ? std::isalnum((int)c) : std::iswalnum((wint_t)c); }
bool is_control(char32_t c) { return c < 0x20 || (c >= 0x7F && c < 0xA0); }
bool is_space(char32_t c) { return c == ' ' || c == '\t' || c == 0xA0 || (c >= 0x2000 && c <= 0x200B) || c == 0x3000; }

void put_utf8(std::string &s, char32_t c) {
  if (c < 0x80) s += (char)c;
  else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
  else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
  else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
}

std::u32string decode_utf8(std::string_view s) {
  std::u32string out;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
    char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
    out += cp;
    i += n;
  }
  return out;
}

// liteparse ACCENTED_VOWELS (lowercase precomposed Latin vowels with diacritics)
const std::u32string &accented_vowels() {
  static const std::u32string v = decode_utf8(
      "àáâãäåāăąǎȁȃȧạảấầẩẫậắằẳẵặèéêëēĕėęěȅȇẹẻẽếềểễệìíîïĩīĭįǐȉȋỉịòóôõöøōŏőǒȍȏȯ"
      "ọỏốồổỗộơớờởỡợùúûüũūŭůűųǔǖǘǚǜȕȗụủưứừửữự");
  return v;
}

void letters_vowels(const std::u32string &t, int &letters, int &vowels) {
  letters = vowels = 0;
  for (char32_t c : t) {
    if (c < 128 && std::isalpha((int)c)) {
      ++letters;
      const char l = (char)std::tolower((int)c);
      if (l == 'a' || l == 'e' || l == 'i' || l == 'o' || l == 'u') ++vowels;
    } else if (c >= 128 && accented_vowels().find((char32_t)std::towlower((wint_t)c)) != std::u32string::npos) {
      ++letters;
      ++vowels;
    }
  }
}

bool likely_garbled_item(const std::u32string &t) {  // is_likely_garbled
  int l, v;
  letters_vowels(t, l, v);
  return l >= 10 && v * 10 < l;
}
bool ratio_garbled(int l, int v) { return l >= 30 && v * 5 < l; }

void symbol_intrusions(const std::u32string &cs, int &letters, int &n) {
  static const std::u32string sym = decode_utf8("$¥©«»@¢£¤¦§¨ª¬®¯¶¸º¼½¾¿");
  letters = n = 0;
  for (char32_t c : cs) letters += is_alpha(c);
  for (size_t i = 1; i + 1 < cs.size(); ++i)
    n += sym.find(cs[i]) != std::u32string::npos && is_alpha(cs[i - 1]) && is_alpha(cs[i + 1]);
}
bool symbols_garbled(int letters, int n) { return n >= 20 && n >= letters * 0.03f; }

void control_chars(const std::u32string &t, int &ctrl, int &total) {
  int vis = 0;
  ctrl = 0;
  for (char32_t c : t) {
    if (is_control(c) && c != '\t' && c != '\n' && c != '\r') ++ctrl;
    else if (!is_space(c)) ++vis;
  }
  total = vis + ctrl;
}
bool controls_garbled(int ctrl, int n) { return ctrl >= 5 && ctrl >= n * 0.02f; }

unsigned script_of(char32_t o) {  // liteparse script_of
  if (o <= 0x24F || (o >= 0x1E00 && o <= 0x1EFF) || (o >= 0x2C60 && o <= 0x2C7F) || (o >= 0xA720 && o <= 0xA7FF)) return 0;
  if ((o >= 0x1100 && o <= 0x11FF) || (o >= 0x3000 && o <= 0x30FF) || (o >= 0x3400 && o <= 0x9FFF) ||
      (o >= 0xAC00 && o <= 0xD7AF) || (o >= 0xF900 && o <= 0xFAFF) || (o >= 0xFF00 && o <= 0xFFEF)) return 1;
  if ((o >= 0x370 && o <= 0x3FF) || (o >= 0x1F00 && o <= 0x1FFF) || (o >= 0x2C80 && o <= 0x2CFF)) return 0x370;
  if ((o >= 0x400 && o <= 0x52F) || (o >= 0x1C80 && o <= 0x1C8F) || (o >= 0x2DE0 && o <= 0x2DFF) || (o >= 0xA640 && o <= 0xA69F)) return 0x400;
  if ((o >= 0x590 && o <= 0x5FF) || (o >= 0xFB1D && o <= 0xFB4F)) return 0x590;
  if ((o >= 0x600 && o <= 0x6FF) || (o >= 0x750 && o <= 0x77F) || (o >= 0x860 && o <= 0x8FF) || (o >= 0xFB50 && o <= 0xFDFF) || (o >= 0xFE70 && o <= 0xFEFF)) return 0x600;
  if (o >= 0x900 && o <= 0xDFF) return o & ~0x7Fu;
  if (o >= 0xE00 && o <= 0xE7F) return 0xE00;
  return 2;
}

// liteparse token_is_malformed: -1 = not judged, 0/1 = clean/malformed.
int token_malformed(std::u32string tok) {
  if (tok.find(U'$') != std::u32string::npos || tok.find(U'\\') != std::u32string::npos) return -1;
  size_t b = 0, e = tok.size();
  while (b < e && !is_alnum(tok[b])) ++b;
  while (e > b && !is_alnum(tok[e - 1])) --e;
  const std::u32string cs = tok.substr(b, e - b);
  int alpha = 0;
  for (char32_t c : cs) alpha += is_alpha(c);
  if (cs.size() < 3 || alpha < 2) return -1;
  const bool has_sep = std::any_of(cs.begin(), cs.end(), [](char32_t c) { return c == '_' || c == '.' || c == '/' || c == '@'; });
  if (!has_sep) {
    int flips = 0;
    bool short_run = false;
    for (size_t i = 1; i + 1 < cs.size(); ++i)
      if (is_lower(cs[i - 1]) && is_upper(cs[i]) && is_lower(cs[i + 1])) {
        ++flips;
        size_t run = 0;
        while (i + 1 + run < cs.size() && is_lower(cs[i + 1 + run])) ++run;
        short_run |= run < 3;
      }
    if (flips >= 2 || short_run) return 1;
  }
  for (size_t i = 1; i + 1 < cs.size(); ++i)
    if ((cs[i] == ';' || cs[i] == '=') && is_alpha(cs[i - 1]) && is_alpha(cs[i + 1])) return 1;
  auto other = [](char32_t c) { return is_alpha(c) && !(c >= 0x300 && c <= 0x36F) && script_of(c) > 1; };
  auto accented = [](char32_t c) { return is_alpha(c) && c >= 0xC0 && c <= 0x24F; };
  long seen = -1;
  for (char32_t c : cs)
    if (other(c)) {
      const long s = script_of(c);
      if (seen >= 0 && seen != s) return 1;
      seen = s;
    }
  for (size_t i = 0; i + 1 < cs.size(); ++i)
    if ((accented(cs[i]) && other(cs[i + 1])) || (other(cs[i]) && accented(cs[i + 1]))) return 1;
  return 0;
}
void malformed_tokens(const std::u32string &t, int &judged, int &bad) {
  judged = bad = 0;
  std::u32string tok;
  auto flush = [&] {
    if (tok.empty()) return;
    const int m = token_malformed(tok);
    if (m >= 0) { ++judged; bad += m; }
    tok.clear();
  };
  for (char32_t c : t) {
    if (is_space(c) || is_control(c)) flush();
    else tok += c;
  }
  flush();
}
bool tokens_garbled(int judged, int bad) { return bad >= 10 && bad >= judged * 0.2f; }

bool is_symbol_font(const std::string &name) {
  std::string n = name;
  for (auto &c : n) c = (char)std::tolower((unsigned char)c);
  if (auto p = n.find('+'); p != std::string::npos) n = n.substr(p + 1);
  for (const char *s : {"wingding", "webding", "dingbat", "symbol", "zapf", "math", "sym", "greekwith"})
    if (n.find(s) != std::string::npos) return true;
  for (const char *s : {"cmex", "cmsy", "cmmi", "msam", "msbm", "rtxm", "rtxs", "txex", "txsy", "xyatip", "xydash",
                        "eufm", "esint", "stmary", "wasy"})
    if (n.rfind(s, 0) == 0) return true;
  return false;
}

// Page space -> viewport (rotation + crop box), derived from FPDF_PageToDevice at 64x.
struct Viewport {
  double a, b, c, d, e, f;  // vx = a x + c y + e ; vy = b x + d y + f
  void init(FPDF_PAGE page, float w_pt, float h_pt) {
    constexpr int S = 64;
    const int W = (int)std::lround(w_pt * S), H = (int)std::lround(h_pt * S);
    int x0, y0, x1, y1, x2, y2;
    FPDF_PageToDevice(page, 0, 0, W, H, 0, 0, 0, &x0, &y0);
    FPDF_PageToDevice(page, 0, 0, W, H, 0, 1000, 0, &x1, &y1);
    FPDF_PageToDevice(page, 0, 0, W, H, 0, 0, 1000, &x2, &y2);
    a = (x1 - x0) / (1000.0 * S);
    b = (y1 - y0) / (1000.0 * S);
    c = (x2 - x0) / (1000.0 * S);
    d = (y2 - y0) / (1000.0 * S);
    e = x0 / double(S);
    f = y0 / double(S);
  }
  RectF rect(double l, double bo, double r, double t) const {
    const double xs[4] = {l, r, r, l}, ys[4] = {bo, bo, t, t};
    RectF o{1e9f, 1e9f, -1e9f, -1e9f};
    for (int k = 0; k < 4; ++k) {
      const float vx = (float)(a * xs[k] + c * ys[k] + e), vy = (float)(b * xs[k] + d * ys[k] + f);
      o.x0 = std::min(o.x0, vx); o.x1 = std::max(o.x1, vx);
      o.y0 = std::min(o.y0, vy); o.y1 = std::max(o.y1, vy);
    }
    return o;
  }
};

float inter_area(const RectF &p, const RectF &q) {
  const float w = std::min(p.x1, q.x1) - std::max(p.x0, q.x0), h = std::min(p.y1, q.y1) - std::max(p.y0, q.y0);
  return w > 0 && h > 0 ? w * h : 0.f;
}

struct Mat { double a = 1, b = 0, c = 0, d = 1, e = 0, f = 0; };
Mat mul(const Mat &m, const FS_MATRIX &n) {  // apply n then m
  return {n.a * m.a + n.b * m.c, n.a * m.b + n.b * m.d, n.c * m.a + n.d * m.c,
          n.c * m.b + n.d * m.d, n.e * m.a + n.f * m.c + m.e, n.e * m.b + n.f * m.d + m.f};
}

// liteparse collect_filled_paths: filled, opaque-ish, dark, >= 3pt, not full-page; forms to depth 4.
void collect_paths(FPDF_PAGEOBJECT obj, const Mat &m, int depth, const Viewport &vp, float pw, float ph,
                   std::vector<RectF> &out) {
  const int type = FPDFPageObj_GetType(obj);
  if (type == FPDF_PAGEOBJ_FORM && depth < 4) {
    FS_MATRIX fm;
    Mat mm = m;
    if (FPDFPageObj_GetMatrix(obj, &fm)) mm = mul(m, fm);
    const int n = FPDFFormObj_CountObjects(obj);
    for (int i = 0; i < n; ++i) collect_paths(FPDFFormObj_GetObject(obj, i), mm, depth + 1, vp, pw, ph, out);
    return;
  }
  if (type != FPDF_PAGEOBJ_PATH) return;
  int fill = 0;
  FPDF_BOOL stroke = 0;
  if (!FPDFPath_GetDrawMode(obj, &fill, &stroke) || fill == FPDF_FILLMODE_NONE) return;
  unsigned r = 0, g = 0, bl = 0, al = 255;
  if (FPDFPageObj_GetFillColor(obj, &r, &g, &bl, &al)) {
    if (al < 128 || 0.299 * r + 0.587 * g + 0.114 * bl > 140) return;
  }
  float l, bo, rt, t;
  if (!FPDFPageObj_GetBounds(obj, &l, &bo, &rt, &t)) return;
  if (depth > 0) {  // child bounds are in form space: map corners through the composed matrix
    const double xs[4] = {l, rt, rt, l}, ys[4] = {bo, bo, t, t};
    double X0 = 1e9, Y0 = 1e9, X1 = -1e9, Y1 = -1e9;
    for (int k = 0; k < 4; ++k) {
      const double x = m.a * xs[k] + m.c * ys[k] + m.e, y = m.b * xs[k] + m.d * ys[k] + m.f;
      X0 = std::min(X0, x); X1 = std::max(X1, x); Y0 = std::min(Y0, y); Y1 = std::max(Y1, y);
    }
    l = (float)X0; bo = (float)Y0; rt = (float)X1; t = (float)Y1;
  }
  const RectF v = vp.rect(l, bo, rt, t);
  if (v.w() < 3 || v.h() < 3) return;
  if (v.w() > kFullPageFrac * pw && v.h() > kFullPageFrac * ph) return;
  out.push_back(v);
}

// Merge rects that touch or lie within `gap` pt of each other.
std::vector<RectF> merge_rects(std::vector<RectF> rs, float gap) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < rs.size() && !changed; ++i)
      for (size_t j = i + 1; j < rs.size(); ++j) {
        const RectF &p = rs[i], &q = rs[j];
        if (p.x0 - gap <= q.x1 && q.x0 - gap <= p.x1 && p.y0 - gap <= q.y1 && q.y0 - gap <= p.y1) {
          rs[i] = {std::min(p.x0, q.x0), std::min(p.y0, q.y0), std::max(p.x1, q.x1), std::max(p.y1, q.y1)};
          rs.erase(rs.begin() + (long)j);
          changed = true;
          break;
        }
      }
  }
  return rs;
}

// ── inline styles (liteparse projection.rs font_is_bold / font_is_italic / is_mono_item,
// extract.rs assign_strikethrough / assign_underline, projection.rs mark_script_items) ──

// A thin horizontal mark: a stroked line or a thin filled rect (viewport pt; y = centre line).
struct HSeg {
  float x0, x1, y, h;
};

void collect_hsegs_obj(FPDF_PAGEOBJECT obj, const Mat &m, int depth, const Viewport &vp, std::vector<HSeg> &out) {
  const int type = FPDFPageObj_GetType(obj);
  if (type == FPDF_PAGEOBJ_FORM && depth < 4) {
    FS_MATRIX fm;
    Mat mm = m;
    if (FPDFPageObj_GetMatrix(obj, &fm)) mm = mul(m, fm);
    const int n = FPDFFormObj_CountObjects(obj);
    for (int i = 0; i < n; ++i) collect_hsegs_obj(FPDFFormObj_GetObject(obj, i), mm, depth + 1, vp, out);
    return;
  }
  if (type != FPDF_PAGEOBJ_PATH) return;
  float l, bo, rt, t;
  if (!FPDFPageObj_GetBounds(obj, &l, &bo, &rt, &t)) return;
  if (depth > 0) {
    const double xs[4] = {l, rt, rt, l}, ys[4] = {bo, bo, t, t};
    double X0 = 1e9, Y0 = 1e9, X1 = -1e9, Y1 = -1e9;
    for (int k = 0; k < 4; ++k) {
      const double x = m.a * xs[k] + m.c * ys[k] + m.e, y = m.b * xs[k] + m.d * ys[k] + m.f;
      X0 = std::min(X0, x); X1 = std::max(X1, x); Y0 = std::min(Y0, y); Y1 = std::max(Y1, y);
    }
    l = (float)X0; bo = (float)Y0; rt = (float)X1; t = (float)Y1;
  }
  const RectF v = vp.rect(l, bo, rt, t);
  if (v.h() <= kMaxRuleThickness && v.w() > v.h()) out.push_back({v.x0, v.x1, 0.5f * (v.y0 + v.y1), v.h()});
}

std::vector<HSeg> collect_hsegs(FPDF_PAGE page, const Viewport &vp) {
  std::vector<HSeg> out;
  const int n = FPDFPage_CountObjects(page);
  for (int i = 0; i < n; ++i) collect_hsegs_obj(FPDFPage_GetObject(page, i), Mat{}, 0, vp, out);
  return out;
}

// Bit set in a font's flags (above the PDF's 19 defined ones) once its glyphs measured proportional.
constexpr int kMeasuredProportional = 1 << 30;
void font_proportional_mark(int &flags) { flags = (flags & ~1) | kMeasuredProportional; }

bool name_has(const std::string &name, std::initializer_list<const char *> hints) {
  for (const char *h : hints)
    if (name.find(h) != std::string::npos) return true;
  return false;
}

size_t trimmed_chars(const std::string &s) {
  size_t n = 0;
  for (unsigned char c : s) n += (c & 0xC0) != 0x80 && c != ' ';
  return n;
}

void assign_styles(PageText &pt, const std::vector<int> &flags, const std::vector<int> &weight,
                   const std::vector<HSeg> &segs) {
  constexpr int kFixedPitch = 1, kItalicFlag = 64, kForceBold = 262144;  // PDF font descriptor flags
  for (TextItem &it : pt.items) {
    if (it.font < 0 || it.font >= (int)flags.size()) continue;
    const std::string &n = pt.fonts[it.font];
    const int f = flags[it.font];
    if ((f & kForceBold) || weight[it.font] >= 600 || name_has(n, {"Bold", "Black", "Heavy", "Semibold", "Demibold"}))
      it.style |= kStyleBold;
    if ((f & kItalicFlag) || name_has(n, {"Italic", "Oblique"})) it.style |= kStyleItalic;
    if (!(f & kMeasuredProportional) &&
        ((f & kFixedPitch) ||
         name_has(n, {"Courier", "Mono", "Consolas", "Menlo", "Source Code", "Inconsolata", "Hack", "Fira Code"})))
      it.style |= kStyleMono;
  }
  // Strikethrough: a thin line through the item's middle band covering >= 60% of its width.
  for (TextItem &it : pt.items) {
    const RectF &b = it.box;
    if ((it.style & kStyleMath) || b.w() <= 0 || b.h() <= 0) continue;
    const float top = b.y0 + 0.20f * b.h(), bot = b.y0 + 0.65f * b.h();
    for (const HSeg &s : segs) {
      if (s.y < top || s.y > bot) continue;
      if (std::max(0.f, std::min(b.x1, s.x1) - std::max(b.x0, s.x0)) >= 0.6f * b.w()) {
        it.style |= kStyleStrike;
        break;
      }
    }
  }
  // Underline: judged per line, the text on it must cover >= 70% of its length (rules and
  // table borders run on past their text); items it runs under by >= half their width.
  for (const HSeg &s : segs) {
    const float len = s.x1 - s.x0;
    if (len < 2.f || len <= 4.f * std::max(s.h, 0.1f)) continue;
    std::vector<size_t> on;
    for (size_t i = 0; i < pt.items.size(); ++i) {
      const RectF &b = pt.items[i].box;
      if ((pt.items[i].style & kStyleMath) || b.w() <= 0 || b.h() <= 0) continue;
      if (s.y >= b.y0 + 0.80f * b.h() && s.y <= b.y1 + std::max(0.25f * b.h(), 2.f) && b.x0 < s.x1 && b.x1 > s.x0)
        on.push_back(i);
    }
    if (on.empty()) continue;
    std::vector<std::pair<float, float>> sp;
    for (size_t i : on) sp.push_back({std::max(pt.items[i].box.x0, s.x0), std::min(pt.items[i].box.x1, s.x1)});
    std::sort(sp.begin(), sp.end());
    float covered = 0, reach = -1e30f;
    for (auto [a, c] : sp) {
      a = std::max(a, reach);
      if (c > a) covered += c - a;
      reach = std::max(reach, c);
    }
    if (covered < 0.7f * len) continue;
    for (size_t i : on) {
      const RectF &b = pt.items[i].box;
      if (std::max(0.f, std::min(b.x1, s.x1) - std::max(b.x0, s.x0)) >= 0.5f * b.w()) pt.items[i].style |= kStyleUnderline;
    }
  }
  // Superscript / subscript: a short item (<= 6 chars) set clearly smaller than a body item it
  // hugs on the same line (gap <= half the body height), raised above or dropped below that
  // item's bottom.
  for (size_t i = 0; i < pt.items.size(); ++i) {
    TextItem &it = pt.items[i];
    const size_t nch = trimmed_chars(it.text);
    if (nch == 0 || nch > 6 || (it.style & kStyleMath)) continue;
    const RectF &a = it.box;
    float body_h = 0, base = 0;
    for (size_t j = 0; j < pt.items.size(); ++j) {
      if (j == i || (pt.items[j].style & kStyleMath)) continue;
      const RectF &b = pt.items[j].box;
      if (b.h() <= 0 || a.h() > 0.78f * b.h() || trimmed_chars(pt.items[j].text) == 0) continue;
      if (!(b.y0 < a.y1 && b.y1 > a.y0)) continue;  // same line
      const float gap = a.x0 >= b.x1 ? a.x0 - b.x1 : b.x0 >= a.x1 ? b.x0 - a.x1 : -1.f;
      if (gap < -0.5f || gap > 0.5f * b.h()) continue;
      if (b.h() > body_h) body_h = b.h(), base = b.y1;
    }
    if (body_h <= 0) continue;
    if (a.y1 <= base - 0.2f * body_h) it.style |= kStyleSup;
    else if (a.y1 >= base + 0.1f * body_h && a.y0 >= base - 0.6f * body_h) it.style |= kStyleSub;
  }
}

}  // namespace

PageText extract_page_text(void *doc, int page_index) {
  PageText pt;
  FPDF_PAGE page = FPDF_LoadPage(static_cast<FPDF_DOCUMENT>(doc), page_index);
  if (!page) return pt;
  pt.w_pt = FPDF_GetPageWidthF(page);
  pt.h_pt = FPDF_GetPageHeightF(page);
  Viewport vp;
  vp.init(page, pt.w_pt, pt.h_pt);

  // ── page objects: images, scan detection, filled paths ──
  const int nobj = FPDFPage_CountObjects(page);
  int text_objs = 0, text_before_scan = 0;
  bool scan_seen = false;
  for (int i = 0; i < nobj; ++i) {
    FPDF_PAGEOBJECT o = FPDFPage_GetObject(page, i);
    const int type = FPDFPageObj_GetType(o);
    if (type == FPDF_PAGEOBJ_TEXT) {
      ++text_objs;
      if (!scan_seen) ++text_before_scan;
    } else if (type == FPDF_PAGEOBJ_IMAGE) {
      float l, b, r, t;
      if (!FPDFPageObj_GetBounds(o, &l, &b, &r, &t)) continue;
      const RectF v = vp.rect(l, b, r, t);
      if (v.w() < kMinImageSizePt || v.h() < kMinImageSizePt) continue;
      if (v.w() > kFullPageFrac * pt.w_pt && v.h() > kFullPageFrac * pt.h_pt) {
        pt.full_page_image = true;
        scan_seen = true;
      } else {
        pt.images.push_back(v);
      }
    }
    collect_paths(o, Mat{}, 0, vp, pt.w_pt, pt.h_pt, pt.paths);
  }
  pt.hidden_under_scan_share = (scan_seen && text_objs) ? (float)text_before_scan / text_objs : 0.f;

  // ── characters -> items ──
  FPDF_TEXTPAGE tp = FPDFText_LoadPage(page);
  const int n = tp ? FPDFText_CountChars(tp) : 0;
  // pass 1: visible / invisible counts (should_skip_invisible)
  int visible = 0, invisible = 0;
  {
    FPDF_PAGEOBJECT last = nullptr;
    bool inv = false;
    for (int i = 0; i < n; ++i) {
      const unsigned u = FPDFText_GetUnicode(tp, i);
      if (FPDFText_IsGenerated(tp, i) == 1 || u == 0 || u == 0xFFFE || u == 0xFFFF || is_space(u) || is_control(u)) continue;
      FPDF_PAGEOBJECT o = FPDFText_GetTextObject(tp, i);
      if (o != last) {
        last = o;
        inv = o && FPDFTextObj_GetTextRenderMode(o) == FPDF_TEXTRENDERMODE_INVISIBLE;
      }
      (inv ? invisible : visible)++;
    }
  }
  const bool skip_invisible = visible > 0 && invisible > 0 && (float)invisible / (visible + invisible) < 0.3f;
  pt.invisible_share = (visible + invisible) ? (float)invisible / (visible + invisible) : 0.f;

  GlyphDecoder glyphs;
  // Lying fonts: some glyph's ToUnicode says "space" but the glyph is visible (Word
  // math). pdfium collapses or drops those chars, so every text object drawn in such a
  // font becomes a span read from pixels; native items are split around the spans.
  std::set<void *> lying_fonts;
  for (int i = 0; i < n; ++i) {
    const unsigned u = FPDFText_GetUnicode(tp, i);
    if ((u != 0x20 && u != 0xA0) || FPDFText_IsGenerated(tp, i) == 1) continue;
    FPDF_PAGEOBJECT o = FPDFText_GetTextObject(tp, i);
    if (o && glyphs.decode(tp, i, o, u, FPDFText_HasUnicodeMapError(tp, i) == 1)) lying_fonts.insert(FPDFTextObj_GetFont(o));
  }
  if (!lying_fonts.empty()) {
    std::vector<RectF> spans;
    for (int k = 0; k < nobj; ++k) {
      FPDF_PAGEOBJECT o = FPDFPage_GetObject(page, k);
      if (FPDFPageObj_GetType(o) != FPDF_PAGEOBJ_TEXT || !lying_fonts.count(FPDFTextObj_GetFont(o))) continue;
      float l, b, r, t;
      if (FPDFPageObj_GetBounds(o, &l, &b, &r, &t)) {
        const RectF v = vp.rect(l, b, r, t);
        if (v.w() > 0.5f && v.h() > 0.5f) spans.push_back(v);
      }
    }
    pt.lie_spans = merge_rects(std::move(spans), 0.5f);
  }
  auto crosses_span = [&](const RectF &prev, const RectF &next) {
    for (const auto &sp : pt.lie_spans) {
      const float cy = 0.5f * (next.y0 + next.y1);
      if (cy < sp.y0 || cy > sp.y1) continue;
      if (prev.x1 <= sp.x0 + 0.5f && next.x0 >= sp.x1 - 0.5f) return true;
    }
    return false;
  };
  if (tp) glyphs.prescan(tp, n);
  // TeX math (tex_math.h): LaTeX from the glyphs of born-digital TeX pages. The glyphs a span
  // replaces end the current item and are skipped (liteparse treats them like hidden chars); the
  // span becomes one "$tex$" item, placed in the item sequence where its first glyph was.
  std::vector<tex::MathSpan> math;
  std::vector<char> consumed;
  if (tp) {
    math = tex::page_math(page, tp);
    if (!math.empty()) {
      consumed.assign(static_cast<size_t>(n), 0);
      for (const auto &m : math)
        for (int c : m.chars)
          if (c >= 0 && c < n) consumed[static_cast<size_t>(c)] = 1;
    }
  }
  std::vector<int> item_first;  // per pt.items entry: pdfium index of its first char
  int cur_first = 0;
  std::unordered_map<std::string, int> font_ids;
  std::vector<int> font_flags, font_weight;  // per PageText::fonts entry (first object seen)
  std::vector<float> font_wmin, font_wmax;   // per font: narrowest / widest alphanumeric advance (per em)
  std::vector<int> font_wn;
  TextItem cur;
  std::u32string cur32;
  RectF last{};
  float width_sum = 0;
  float item_fs = 0, last_ink_bottom = 0;  // effective font size of the item's first char; last glyph's ink bottom
  bool next_band = false;                   // the next run is a script split: it takes next_y0..next_y1
  float next_y0 = 0, next_y1 = 0;
  int font = -1;
  bool pending_space = false, inv = false;
  FPDF_PAGEOBJECT last_obj = nullptr;
  char fbuf[256];

  auto flush = [&] {
    while (!cur32.empty() && cur32.back() == U' ') { cur32.pop_back(); cur.text.pop_back(); }
    if (!cur32.empty()) {
      const RectF &b = cur.box;
      const bool outside = b.x1 < 0 || b.y1 < 0 || b.x0 > pt.w_pt || b.y0 > pt.h_pt;
      if (!outside) {
        cur.unusable = cur.unmapped * 2 >= cur.chars && cur.unmapped > 0;
        cur.unusable |= likely_garbled_item(cur32);
        pt.unmapped_chars += cur.unmapped;
        pt.nonspace_chars += cur.chars;
        pt.items.push_back(std::move(cur));
        item_first.push_back(cur_first);
      }
    }
    cur = TextItem{};
    cur32.clear();
    width_sum = 0;
    next_band = false;
  };

  for (int i = 0; i < n; ++i) {
    if (!consumed.empty() && consumed[static_cast<size_t>(i)]) {
      flush();
      continue;
    }
    char32_t u = FPDFText_GetUnicode(tp, i);
    if (u >= 0xD800 && u <= 0xDBFF && i + 1 < n) {  // surrogate pair
      const unsigned lo = FPDFText_GetUnicode(tp, i + 1);
      if (lo >= 0xDC00 && lo <= 0xDFFF) { u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00); ++i; }
    }
    if (FPDFText_IsGenerated(tp, i) == 1) {
      if (u == '\n' || u == '\r') flush();
      else if (is_space(u)) pending_space = true;
      continue;
    }
    if (u == '\n' || u == '\r') { flush(); pending_space = false; continue; }
    FPDF_PAGEOBJECT o = FPDFText_GetTextObject(tp, i);
    const bool map_error = FPDFText_HasUnicodeMapError(tp, i) == 1;
    // Glyph-level recovery (liteparse GlyphDecoder) runs before the sentinel skip
    // and the control-code remaps, which apply only when nothing was recovered.
    const std::string *rec = u != '\t' ? glyphs.decode(tp, i, o, u, map_error) : nullptr;
    const char *lig = nullptr;
    if (rec && (u == 0x20 || u == 0xA0)) continue;  // lying glyph: inside an object span read from pixels
    if (rec) {
      if (*rec == " ") { pending_space = true; continue; }
      lig = rec->c_str();
    } else {
      if (is_space(u) || u == 0x01) { pending_space = true; continue; }
      if (u == 0 || u == 0xFFFE || u == 0xFFFF) continue;
      if (u == 0x02) u = '-';  // extract.rs control-code remaps
      else if (u >= 0x1A && u <= 0x1F) {
        static const char *const kLig[] = {"ff", "ft", "fi", "Th", "ffi", "fl"};
        lig = kLig[u - 0x1A];
      }
    }

    if (o != last_obj) {
      last_obj = o;
      inv = o && FPDFTextObj_GetTextRenderMode(o) == FPDF_TEXTRENDERMODE_INVISIBLE;
      int flags = 0;
      const unsigned long len = FPDFText_GetFontInfo(tp, i, fbuf, sizeof fbuf, &flags);
      const std::string name = len > 1 ? std::string(fbuf, std::min<size_t>(len - 1, sizeof fbuf - 1)) : std::string();
      auto [it, added] = font_ids.try_emplace(name, (int)pt.fonts.size());
      if (added) {
        pt.fonts.push_back(name);
        font_flags.push_back(flags);
        font_weight.push_back(FPDFText_GetFontWeight(tp, i));
        font_wmin.push_back(1e9f), font_wmax.push_back(0.f), font_wn.push_back(0);
      }
      font = it->second;
    }
    if (inv && skip_invisible) continue;
    double l, r, bo, t;
    FS_RECTF lr;
    if (!FPDFText_GetLooseCharBox(tp, i, &lr)) continue;
    l = lr.left; r = lr.right; bo = lr.bottom; t = lr.top;
    const RectF b = vp.rect(l, bo, r, t);
    if (b.h() < 0.5f) continue;
    if (!cur32.empty() && !pt.lie_spans.empty() && crosses_span(last, b)) flush();
    // Effective size (font size x text-matrix scale) and the glyph's ink bottom, for the script
    // split below (liteparse extract.rs size_break).
    float fs = (float)FPDFText_GetFontSize(tp, i);
    {
      FS_MATRIX cm;
      if (FPDFText_GetMatrix(tp, i, &cm)) fs *= (float)std::sqrt((double)cm.c * cm.c + (double)cm.d * cm.d);
    }
    float ink_bottom = b.y1, ink_h = b.h();
    {
      double sl, sr, sb, st;
      if (FPDFText_GetCharBox(tp, i, &sl, &sr, &sb, &st)) {
        const RectF ink = vp.rect(sl, sb, sr, st);
        ink_bottom = ink.y1, ink_h = ink.h();
      }
    }

    if (!cur32.empty()) {
      const float gap = b.x0 - last.x1;
      const float em = std::max((float)FPDFText_GetFontSize(tp, i), last.h());
      const float avg_w = width_sum / std::max<size_t>(cur32.size(), 1);
      const float char_w = std::max(avg_w, 0.5f * em);
      const bool y_overlap = b.y0 < last.y1 + kYTol && b.y1 > last.y0 - kYTol;
      // A size step with a baseline shift (into or out of a superscript / subscript run) starts
      // a new item, so the run can be marked; a size change alone is no script.
      const bool shifted = std::fabs(ink_bottom - last_ink_bottom) > 0.15f * std::max(cur.box.h(), ink_h);
      const bool size_break = shifted && item_fs > 0 && fs > 0 && (fs < item_fs * 0.78f || item_fs < fs * 0.78f);
      const bool other = !y_overlap || gap >= kMaxInlineGap || gap < -3 * char_w || font != cur.font ||
                         b.y0 > last.y1 + kYTol;
      bool brk = other || size_break;
      if (!brk && pending_space && gap > 2.2f * avg_w) brk = true;
      if (brk) {
        // Split by the size step alone: the new run keeps its host's band (liteparse script_geom).
        const bool keep_band = size_break && !other;
        const float h0 = cur.has_band ? cur.band_y0 : cur.box.y0, h1 = cur.has_band ? cur.band_y1 : cur.box.y1;
        flush();
        if (keep_band) next_band = true, next_y0 = h0, next_y1 = h1;
      } else if (pending_space || (gap > 0.35f * em && is_alnum(cur32.back()) && is_alnum(lig ? (char32_t)(unsigned char)lig[0] : u))) {
        cur.text += ' ';
        cur32 += U' ';
      }
    }
    if (cur32.empty()) {
      cur.box = b; cur.font = font; item_fs = fs; cur_first = i;
      if (next_band) cur.has_band = true, cur.band_y0 = next_y0, cur.band_y1 = next_y1;
      next_band = false;
    }
    else {
      cur.box.x0 = std::min(cur.box.x0, b.x0); cur.box.y0 = std::min(cur.box.y0, b.y0);
      cur.box.x1 = std::max(cur.box.x1, b.x1); cur.box.y1 = std::max(cur.box.y1, b.y1);
    }
    if (lig) { cur.text += lig; cur32 += decode_utf8(lig); }
    else { u = fold_math_alnum(u); put_utf8(cur.text, u); cur32 += u; }
    cur.chars += 1;
    cur.unmapped += map_error && !rec;
    pt.recovered += rec != nullptr;
    width_sum += b.w();
    if (font >= 0 && fs > 0 && !lig && u < 0x80 && std::isalnum(static_cast<int>(u))) {
      const float wn = b.w() / fs;
      font_wmin[font] = std::min(font_wmin[font], wn), font_wmax[font] = std::max(font_wmax[font], wn);
      ++font_wn[font];
    }
    last = b;
    last_ink_bottom = ink_bottom;
    pending_space = false;
  }
  flush();
  if (!math.empty()) {
    // Splice the math items into the item sequence by first glyph (merge_into_lines joins items
    // in sequence order, so a span must sit between the words it was set between).
    std::vector<std::pair<int, TextItem>> seq;
    seq.reserve(pt.items.size() + math.size());
    for (size_t k = 0; k < pt.items.size(); ++k) seq.emplace_back(item_first[k], std::move(pt.items[k]));
    for (const auto &m : math) {
      TextItem it;
      it.text = "$" + m.tex + "$";
      it.box = {m.x0, m.top, m.x1, m.bottom};
      it.chars = static_cast<int>(m.chars.size());
      it.style = kStyleMath;
      seq.emplace_back(m.chars.empty() ? n : m.chars.front(), std::move(it));
    }
    std::stable_sort(seq.begin(), seq.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    pt.items.clear();
    for (auto &[k, it] : seq) pt.items.push_back(std::move(it));
  }
  // Monospace only if the glyphs are: a font whose letters' advances differ by more than 30%
  // ("i" vs "m") is proportional whatever its FixedPitch flag says (seen on embedded Roboto).
  for (size_t f = 0; f < font_flags.size(); ++f)
    if (font_wn[f] >= 8 && font_wmax[f] > 1.3f * font_wmin[f]) font_proportional_mark(font_flags[f]);
  assign_styles(pt, font_flags, font_weight, collect_hsegs(page, vp));
  for (TextItem &it : pt.items)
    if (it.has_band) it.box.y0 = std::min(it.box.y0, it.band_y0), it.box.y1 = std::max(it.box.y1, it.band_y1);
  if (tp) FPDFText_ClosePage(tp);
  FPDF_ClosePage(page);
  return pt;
}

Route route_page(PageText &pt) {
  Route rt;
  const float page_area = std::max(pt.w_pt * pt.h_pt, 1.f);
  float area = 0;
  for (const auto &it : pt.items)
    if (!it.unusable) { rt.text_len += (int)it.text.size(); area += it.box.w() * it.box.h(); }
  rt.coverage = area / page_area;

  // ── garbled_scope: page-wide, then per font ──
  struct Acc { int l = 0, v = 0, sl = 0, sy = 0, c = 0, n = 0, j = 0, m = 0; };
  Acc page;
  std::map<int, Acc> per_font;
  for (const auto &it : pt.items) {
    if (it.style & kStyleMath) continue;  // LaTeX ("$\\mathbb{N}$"), not text: no garbled-text evidence
    const std::u32string t = decode_utf8(it.text);
    const bool sym = it.font >= 0 && is_symbol_font(pt.fonts[it.font]);
    Acc a;
    letters_vowels(t, a.l, a.v);
    symbol_intrusions(t, a.sl, a.sy);
    if (!sym) { control_chars(t, a.c, a.n); malformed_tokens(t, a.j, a.m); }
    for (Acc *x : {&page, &per_font[it.font]}) {
      x->l += a.l; x->v += a.v; x->sl += a.sl; x->sy += a.sy; x->c += a.c; x->n += a.n; x->j += a.j; x->m += a.m;
    }
  }
  const bool whole = ratio_garbled(page.l, page.v) || symbols_garbled(page.sl, page.sy) ||
                     controls_garbled(page.c, page.n) || tokens_garbled(page.j, page.m);
  std::set<int> bad_fonts;
  if (!whole) {
    const int min_share = (int)std::ceil(page.l * 0.15f);
    for (const auto &[f, a] : per_font)
      if ((a.l >= min_share && ratio_garbled(a.l, a.v)) || symbols_garbled(a.sl, a.sy) ||
          controls_garbled(a.c, a.n) || tokens_garbled(a.j, a.m))
        bad_fonts.insert(f);
  }
  const bool unmapped_page = pt.unmapped_chars >= 30 && pt.unmapped_chars >= 0.15f * pt.nonspace_chars;

  // ── page-level decision (liteparse calculate_page_complexity) ──
  const bool sparse = rt.text_len < 2000 && rt.coverage < 0.15f;
  if (rt.text_len < kMinNativeTextLen) {
    rt.reasons |= pt.full_page_image ? kScanned : kNoText;
  } else if (sparse && (pt.full_page_image || rt.text_len < kSparseTextTrustedLen)) {
    rt.reasons |= kSparseText;
  }
  if (whole) rt.reasons |= kGarbledPage;
  if (pt.full_page_image && rt.text_len >= kMinNativeTextLen &&
      (pt.invisible_share >= kInvisibleLayerMinShare || pt.hidden_under_scan_share >= kInvisibleLayerMinShare))
    rt.reasons |= kInvisibleLayer;

  // Experiment switch: trust a scan's existing invisible OCR layer instead of re-OCRing.
  static const bool trust_invisible = std::getenv("HP_TRUST_INVISIBLE") != nullptr;
  if (trust_invisible && pt.full_page_image && pt.invisible_share > 0.5f && rt.text_len >= kMinNativeTextLen &&
      !(rt.reasons & kGarbledPage)) {
    for (auto &it : pt.items) it.dropped = it.unusable;
    rt.kind = kNative;
    return rt;
  }
  if (rt.reasons & (kGarbledPage | kInvisibleLayer)) {  // native text replaced by OCR
    rt.kind = kFullOcr;
    rt.keep_native = false;
    for (auto &it : pt.items) it.dropped = true;
    return rt;
  }
  for (auto &it : pt.items) it.dropped = it.unusable;
  if (rt.reasons & (kNoText | kScanned | kSparseText)) {  // OCR the page; good native text still wins
    rt.kind = kFullOcr;
    return rt;
  }

  // ── region-level: only what has no usable text layer ──
  std::vector<RectF> regions = pt.images;
  if (!pt.images.empty()) rt.reasons |= kImages;
  for (auto &it : pt.items) {
    const bool bad_font = bad_fonts.count(it.font) > 0;
    if (bad_font) it.dropped = true;
    if (it.dropped) {  // exact box known from the text layer: recognise it directly
      rt.rec_boxes.push_back(it.box);
      rt.reasons |= bad_font ? kGarbledFonts : kUnusableItems;
    }
  }
  if (unmapped_page) rt.reasons |= kUnusableItems;
  if (!pt.lie_spans.empty()) {
    rt.reasons |= kLyingGlyphs;
    rt.rec_boxes.insert(rt.rec_boxes.end(), pt.lie_spans.begin(), pt.lie_spans.end());
  }
  float uncovered_total = 0;
  std::vector<RectF> vec;
  for (const auto &p : pt.paths) {
    float u = p.w() * p.h();
    for (const auto &it : pt.items) u -= inter_area(p, it.box);
    if (u > 0) { uncovered_total += u; vec.push_back(p); }
  }
  if (uncovered_total >= kUncoveredVectorArea) {
    rt.reasons |= kVectorText;
    regions.insert(regions.end(), vec.begin(), vec.end());
  }
  rt.regions = merge_rects(std::move(regions), 3.0f);
  rt.kind = (rt.regions.empty() && rt.rec_boxes.empty()) ? kNative : kPartial;
  return rt;
}

bool overlaps_native(const PageText &pt, const RectF &r, float tol) {
  for (const auto &it : pt.items) {
    if (it.dropped) continue;
    if (r.x0 < it.box.x1 + tol && r.x1 > it.box.x0 - tol && r.y0 < it.box.y1 + tol && r.y1 > it.box.y0 - tol) return true;
  }
  return false;
}

std::string reasons_str(unsigned r) {
  static const char *names[] = {"no_text", "scanned", "sparse_text", "images", "garbled_page", "garbled_fonts",
                                "unusable_items", "invisible_layer", "vector_text", "annotation_text", "lying_glyphs"};
  std::string s;
  for (int i = 0; i < 11; ++i)
    if (r & (1u << i)) { if (!s.empty()) s += ','; s += names[i]; }
  return s;
}

// ── binary hand-off ─────────────────────────────────────────────────────────
namespace {
struct W {
  std::vector<uint8_t> &o;
  template <class T> void pod(const T &v) { const auto *p = reinterpret_cast<const uint8_t *>(&v); o.insert(o.end(), p, p + sizeof v); }
  void str(const std::string &s) { pod((uint32_t)s.size()); o.insert(o.end(), s.begin(), s.end()); }
};
struct R {
  const uint8_t *p, *end;
  template <class T> bool pod(T &v) { if (p + sizeof v > end) return false; std::memcpy(&v, p, sizeof v); p += sizeof v; return true; }
  bool str(std::string &s) { uint32_t n; if (!pod(n) || p + n > end) return false; s.assign((const char *)p, n); p += n; return true; }
};
}  // namespace

void serialize(const PageText &pt, const Route &rt, std::vector<uint8_t> &out) {
  out.clear();
  W w{out};
  w.pod(pt.w_pt); w.pod(pt.h_pt);
  w.pod(rt.kind); w.pod(rt.keep_native); w.pod(rt.reasons); w.pod(rt.text_len); w.pod(rt.coverage);
  w.pod((uint32_t)rt.regions.size());
  for (const auto &r : rt.regions) w.pod(r);
  w.pod((uint32_t)rt.rec_boxes.size());
  for (const auto &r : rt.rec_boxes) w.pod(r);
  w.pod((uint32_t)pt.fonts.size());
  for (const auto &f : pt.fonts) w.str(f);
  w.pod((uint32_t)pt.items.size());
  for (const auto &it : pt.items) {
    w.str(it.text); w.pod(it.box); w.pod(it.font); w.pod(it.unusable); w.pod(it.dropped); w.pod(it.style);
  }
}

bool deserialize(const uint8_t *p, size_t n, PageText &pt, Route &rt) {
  R r{p, p + n};
  uint32_t k;
  if (!(r.pod(pt.w_pt) && r.pod(pt.h_pt) && r.pod(rt.kind) && r.pod(rt.keep_native) && r.pod(rt.reasons) &&
        r.pod(rt.text_len) && r.pod(rt.coverage) && r.pod(k))) return false;
  rt.regions.resize(k);
  for (auto &x : rt.regions) if (!r.pod(x)) return false;
  if (!r.pod(k)) return false;
  rt.rec_boxes.resize(k);
  for (auto &x : rt.rec_boxes) if (!r.pod(x)) return false;
  if (!r.pod(k)) return false;
  pt.fonts.resize(k);
  for (auto &f : pt.fonts) if (!r.str(f)) return false;
  if (!r.pod(k)) return false;
  pt.items.resize(k);
  for (auto &it : pt.items)
    if (!(r.str(it.text) && r.pod(it.box) && r.pod(it.font) && r.pod(it.unusable) && r.pod(it.dropped) &&
          r.pod(it.style)))
      return false;
  return true;
}

}  // namespace hp
