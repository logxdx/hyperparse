// pdfium side of the math pass: which glyphs are collected, their boxes, and the
// rules, as LiteParse's page_math_spans (extract.rs) and TextClip (text_clip.rs)
// do. Coordinates are hp's text-layer space (text_layer.cpp): viewport points,
// origin top-left, y down, page size FPDF_GetPageWidthF / FPDF_GetPageHeightF.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <unordered_map>
#include <unordered_set>

#include <fpdf_edit.h>
#include <fpdf_text.h>
#include <fpdf_transformpage.h>
#include <fpdfview.h>

#include "detail.h"
#include "agl_table.h"
#include "glyph_decoder.h"

namespace hp::tex {
namespace {

using namespace detail;

// Page space -> viewport (rotation + crop box): LiteParse's ViewportTransform
// (pdfium crate page.rs), probed through FPDF_PageToDevice at 1000x and applied
// in f32. Same space as hp's text layer (viewport points, origin top-left, y
// down; text_layer.cpp probes at 64x, so the two agree to < 0.01 pt). The finer
// probe is kept because TeX sets glyphs on exact multiples, and the math pass has
// thresholds those ties land on (a cmex piece 0.25 em off a baseline).
struct Viewport {
  FPDF_PAGE page = nullptr;
  float mult = 1000.0f;
  int dw = 0, dh = 0;
  float a = 1, b = 0, c = 0, d = 1, e = 0, f = 0;  // vx = a x + b y + e ; vy = c x + d y + f
  void init(FPDF_PAGE pg, float vw, float vh) {
    page = pg;
    mult = std::min(1000.0f, (float)(std::numeric_limits<int32_t>::max() / 4) / std::max(std::max(vw, vh), 1.0f));
    dw = (int)std::round(vw * mult);
    dh = (int)std::round(vh * mult);
    const auto [e0, f0] = to_viewport(0, 0);
    const auto [ax, cx] = to_viewport(1, 0);
    const auto [by, dy] = to_viewport(0, 1);
    a = ax - e0; b = by - e0; c = cx - f0; d = dy - f0; e = e0; f = f0;
  }
  // Page::page_to_viewport: one FPDF_PageToDevice probe.
  std::pair<float, float> to_viewport(float px, float py) const {
    int dx = 0, dy = 0;
    FPDF_PageToDevice(page, 0, 0, dw, dh, 0, (double)px, (double)py, &dx, &dy);
    return {(float)dx / mult, (float)dy / mult};
  }
  std::pair<float, float> point(float px, float py) const { return {a * px + b * py + e, c * px + d * py + f}; }
  // (x0, top, x1, bottom) of a page-space box (ViewportTransform::transform_bounds).
  struct R { float x0, top, x1, bottom; };
  R rect(float l, float bo, float r, float t) const {
    const auto [lx, ly] = point(l, bo);
    const auto [ux, uy] = point(r, t);
    return {std::min(lx, ux), std::min(ly, uy), std::max(lx, ux), std::max(ly, uy)};
  }
  // Page::bounds_to_viewport: corners probed through pdfium.
  R probe_rect(float l, float bo, float r, float t) const {
    const auto [lx, ly] = to_viewport(l, bo);
    const auto [ux, uy] = to_viewport(r, t);
    return {std::min(lx, ux), std::min(ly, uy), std::max(lx, ux), std::max(ly, uy)};
  }
};

bool is_ws(uint32_t c) {  // Rust char::is_whitespace
  return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
bool valid_scalar(uint32_t c) { return c <= 0x10FFFF && !(c >= 0xD800 && c <= 0xDFFF); }
bool is_control(uint32_t c) { return c < 0x20 || (c >= 0x7F && c < 0xA0); }

// ── TextClip (LiteParse text_clip.rs): glyphs hidden by rectangular clips ─────

struct PRect {  // y-up page rect
  float left, top, right, bottom;
  bool operator==(const PRect &) const = default;
};
struct Mat {
  float a = 1, b = 0, c = 0, d = 1, e = 0, f = 0;
};
Mat compose(const Mat &p, const Mat &c) {
  return {p.a * c.a + p.c * c.b, p.b * c.a + p.d * c.b, p.a * c.c + p.c * c.d,
          p.b * c.c + p.d * c.d, p.a * c.e + p.c * c.f + p.e, p.b * c.e + p.d * c.f + p.f};
}
bool valid_matrix(const Mat &m) {
  for (float v : {m.a, m.b, m.c, m.d, m.e, m.f})
    if (!std::isfinite(v)) return false;
  return std::abs((double)m.a * (double)m.d - (double)m.b * (double)m.c) > 1e-12;
}

struct Seg {
  int kind;  // FPDF_SEGMENT_*
  float x, y;
  bool close;
};

// Accept exactly one closed rectangle, never the bounding box of a compound path
// (which can contain holes). Rotated/sheared non-axis-aligned clips fail open.
std::optional<PRect> rectangle(const std::vector<Seg> &path, const Mat &parent) {
  if (!(path.size() == 4 || path.size() == 5) || !path.back().close || !valid_matrix(parent)) return std::nullopt;
  std::vector<std::pair<float, float>> pts;
  for (size_t i = 0; i < path.size(); ++i) {
    const int expected = i == 0 ? FPDF_SEGMENT_MOVETO : FPDF_SEGMENT_LINETO;
    if (path[i].kind != expected || (i + 1 != path.size() && path[i].close)) return std::nullopt;
    const float x = path[i].x, y = path[i].y;
    const std::pair<float, float> p{parent.a * x + parent.c * y + parent.e, parent.b * x + parent.d * y + parent.f};
    if (!std::isfinite(p.first) || !std::isfinite(p.second)) return std::nullopt;
    pts.push_back(p);
  }
  if (pts.size() == 5) {
    const auto last = pts.back();
    pts.pop_back();
    if (last != pts[0]) return std::nullopt;
  }
  bool horizontal[4];
  for (int i = 0; i < 4; ++i) {
    const auto a = pts[i], b = pts[(i + 1) % 4];
    horizontal[i] = a.second == b.second && a.first != b.first;
    if (!(horizontal[i] || (a.first == b.first && a.second != b.second))) return std::nullopt;
  }
  for (int i = 0; i < 4; ++i)
    if (horizontal[i] == horizontal[(i + 1) % 4]) return std::nullopt;
  PRect r{std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
  for (const auto &[x, y] : pts) {
    r.left = std::min(r.left, x); r.right = std::max(r.right, x);
    r.bottom = std::min(r.bottom, y); r.top = std::max(r.top, y);
  }
  return r;
}

// Clip paths of an object (its own matrix applied); nullopt when unreadable.
std::optional<std::vector<std::vector<Seg>>> clip_paths(FPDF_PAGEOBJECT obj) {
  FPDF_CLIPPATH clip = FPDFPageObj_GetClipPath(obj);
  if (!clip) return std::nullopt;
  const int count = FPDFClipPath_CountPaths(clip);
  std::vector<std::vector<Seg>> paths;
  if (count == -1) return paths;
  if (count < 0 || count > 1024) return std::nullopt;
  int budget = 1024;
  for (int p = 0; p < count; ++p) {
    const int n = FPDFClipPath_CountPathSegments(clip, p);
    if (n < 0 || n > budget) return std::nullopt;
    budget -= n;
    std::vector<Seg> segs;
    for (int k = 0; k < n; ++k) {
      FPDF_PATHSEGMENT s = FPDFClipPath_GetPathSegment(clip, p, k);
      if (!s) return std::nullopt;
      const int t = FPDFPathSegment_GetType(s);
      const int kind = (t == FPDF_SEGMENT_MOVETO || t == FPDF_SEGMENT_LINETO || t == FPDF_SEGMENT_BEZIERTO) ? t : -1;
      float x = 0, y = 0;
      if (!FPDFPathSegment_GetPoint(s, &x, &y)) return std::nullopt;
      segs.push_back({kind, x, y, FPDFPathSegment_GetClose(s) != 0});
    }
    paths.push_back(std::move(segs));
  }
  return paths;
}

// Only objects whose complete path stack is understood have an entry. Bounds are
// y-up page coordinates; inverted bounds mean an empty intersection.
struct ObjectClips {
  std::unordered_map<FPDF_PAGEOBJECT, PRect> objects;

  void walk(FPDF_PAGEOBJECT obj, const Mat &parent, std::optional<PRect> bounds, int depth) {
    if (depth > 64) return;
    const auto paths = clip_paths(obj);
    if (!paths) return;
    for (const auto &path : *paths) {
      const auto rect = rectangle(path, parent);
      if (!rect) return;
      if (!bounds) bounds = *rect;
      else
        bounds = PRect{std::max(bounds->left, rect->left), std::min(bounds->top, rect->top),
                       std::min(bounds->right, rect->right), std::max(bounds->bottom, rect->bottom)};
    }
    const int type = FPDFPageObj_GetType(obj);
    if (type == FPDF_PAGEOBJ_TEXT) {
      if (bounds) objects[obj] = *bounds;
    } else if (type == FPDF_PAGEOBJ_FORM) {
      FS_MATRIX m{1, 0, 0, 1, 0, 0};
      if (!FPDFPageObj_GetMatrix(obj, &m)) return;
      const Mat mm = compose(parent, Mat{m.a, m.b, m.c, m.d, m.e, m.f});
      if (!valid_matrix(mm)) return;
      const int n = FPDFFormObj_CountObjects(obj);
      if (n < 0) return;
      for (int k = 0; k < n; ++k)
        if (FPDF_PAGEOBJECT ch = FPDFFormObj_GetObject(obj, (unsigned long)k)) walk(ch, mm, bounds, depth + 1);
    }
  }

  const PRect *bounds(FPDF_PAGEOBJECT obj) const {
    if (!obj) return nullptr;
    const auto it = objects.find(obj);
    return it == objects.end() ? nullptr : &it->second;
  }
};

std::optional<PRect> loose_box(FPDF_TEXTPAGE tp, int i) {
  FS_RECTF r;
  if (!FPDFText_GetLooseCharBox(tp, i, &r)) return std::nullopt;
  return PRect{r.left, r.top, r.right, r.bottom};
}
std::optional<PRect> strict_box(FPDF_TEXTPAGE tp, int i) {
  double l, r, b, t;
  if (!FPDFText_GetCharBox(tp, i, &l, &r, &b, &t)) return std::nullopt;
  return PRect{(float)l, (float)t, (float)r, (float)b};
}

// The larger side of the loose box, about one em (ascent to descent) at any rotation.
double em_extent(const PRect &l) {
  const double em = std::max(std::abs(l.right - l.left), std::abs(l.top - l.bottom));
  return std::isfinite(em) ? em : 0.0;
}

// Whether two loose glyph boxes touch, within a quarter em, at any rotation.
bool adjacent(const PRect &a, const PRect &b) {
  const float gx = std::max(b.left - a.right, a.left - b.right);
  const float gy = std::max(b.bottom - a.top, a.bottom - b.top);
  return (double)std::max(gx, gy) <= 0.25 * std::max(em_extent(a), em_extent(b));
}

class TextClip {
 public:
  TextClip(FPDF_PAGE page, FPDF_TEXTPAGE tp, bool keep) {
    if (keep) return;
    const int nobj = FPDFPage_CountObjects(page);
    for (int k = 0; k < nobj; ++k)
      if (FPDF_PAGEOBJECT o = FPDFPage_GetObject(page, k)) clips_.walk(o, Mat{}, std::nullopt, 0);
    // Most pages clip no text: skip the char pass entirely.
    if (clips_.objects.empty()) return;
    std::vector<std::tuple<int, uint32_t, bool>> word;
    std::optional<PRect> last, last_clip;
    const int n = FPDFText_CountChars(tp);
    for (int i = 0; i < n; ++i) {
      const uint32_t u = FPDFText_GetUnicode(tp, i);
      const bool gen = FPDFText_IsGenerated(tp, i) == 1;
      FPDF_PAGEOBJECT obj = FPDFText_GetTextObject(tp, i);
      if (gen || (valid_scalar(u) && is_ws(u))) {
        close(word);
        last.reset();
        // A space is judged on its own loose box.
        if (hides(tp, i, gen, u, obj)) hidden_.insert(i);
        continue;
      }
      // pdfium emits no separator at every line break or between objects drawn over
      // each other, so a word also ends where its glyphs stop touching or its clip
      // changes (overflow from one cell running into the next).
      const auto loose = loose_box(tp, i);
      const PRect *cb = clips_.bounds(obj);
      const std::optional<PRect> clip = cb ? std::optional<PRect>(*cb) : std::nullopt;
      if (!word.empty() && (clip != last_clip || (last && loose && !adjacent(*last, *loose)))) close(word);
      last = loose;
      last_clip = clip;
      word.emplace_back(i, u, hides(tp, i, gen, u, obj));
    }
    close(word);
  }
  bool hides(int i) const { return hidden_.count(i) > 0; }

 private:
  ObjectClips clips_;
  std::unordered_set<int> hidden_;

  // A glyph is visible when its ink box, widened by INK_MARGIN of its em extent,
  // meets the clip; glyphs without ink are judged by their loose box.
  bool hides(FPDF_TEXTPAGE tp, int i, bool generated, uint32_t u, FPDF_PAGEOBJECT obj) const {
    constexpr double INK_MARGIN = 0.02, EPS = 0.001;
    if (generated) return false;
    const PRect *bounds = clips_.bounds(obj);
    if (!bounds) return false;
    const bool inked = !(valid_scalar(u) && is_ws(u));
    const auto loose = loose_box(tp, i);
    auto strict = strict_box(tp, i);
    if (strict && !(inked && strict->right > strict->left && strict->top > strict->bottom)) strict.reset();
    PRect glyph;
    double margin;
    if (strict) {
      glyph = *strict;
      margin = INK_MARGIN * (loose ? em_extent(*loose) : 0.0);
    } else if (loose) {
      glyph = *loose;
      margin = 0.0;
    } else {
      return false;
    }
    if (!(std::isfinite(glyph.left) && std::isfinite(glyph.right) && std::isfinite(glyph.bottom) &&
          std::isfinite(glyph.top)) ||
        glyph.left > glyph.right || glyph.bottom > glyph.top)
      return false;
    const double tol = EPS + margin;
    return bounds->left > bounds->right || bounds->bottom > bounds->top || (double)glyph.right < bounds->left - tol ||
           (double)glyph.left > bounds->right + tol || (double)glyph.top < bounds->bottom - tol ||
           (double)glyph.bottom > bounds->top + tol;
  }

  // A word with any visible glyph is kept whole, except a leader run inside it.
  void close(std::vector<std::tuple<int, uint32_t, bool>> &word) {
    bool visible = false;
    for (const auto &[i, c, h] : word) visible |= !h;
    auto is_leader = [](uint32_t c) {
      return c == '.' || c == 0x2026 || c == '_' || c == '-' || c == 0x2013 || c == 0x2014 || c == 0xB7 ||
             c == 0x2025 || c == 0x22EF;
    };
    std::vector<bool> leader(word.size(), false);
    size_t start = 0;
    while (start < word.size()) {
      size_t end = start;
      while (end < word.size() && is_leader(std::get<1>(word[end]))) ++end;
      if (end - start >= 3)
        for (size_t k = start; k < end; ++k) leader[k] = true;
      start = end + 1;
    }
    for (size_t k = 0; k < word.size(); ++k) {
      const auto &[i, c, h] = word[k];
      if (h && (!visible || leader[k])) hidden_.insert(i);
    }
    word.clear();
  }
};

// ── rules ─────────────────────────────────────────────────────────────────────

struct PathBox {
  Viewport::R box;
  bool stroked;
  float stroke_width;
};

// LiteParse Page::path_objects: every path object, forms to depth 6.
void collect_paths(FPDF_PAGEOBJECT obj, const Mat &parent, const Viewport &vp, int depth, std::vector<PathBox> &out) {
  const int type = FPDFPageObj_GetType(obj);
  if (type == FPDF_PAGEOBJ_FORM) {
    if (depth >= 6) return;
    FS_MATRIX fm{1, 0, 0, 1, 0, 0};
    FPDFPageObj_GetMatrix(obj, &fm);
    const Mat m = compose(parent, Mat{fm.a, fm.b, fm.c, fm.d, fm.e, fm.f});
    const int n = FPDFFormObj_CountObjects(obj);
    for (int i = 0; i < n; ++i)
      if (FPDF_PAGEOBJECT ch = FPDFFormObj_GetObject(obj, (unsigned long)i)) collect_paths(ch, m, vp, depth + 1, out);
    return;
  }
  if (type != FPDF_PAGEOBJ_PATH) return;
  float l, b, r, t;
  if (!FPDFPageObj_GetBounds(obj, &l, &b, &r, &t)) return;
  // Bounds are in the object's content-stream space: lift the corners through the
  // ancestor forms' matrix.
  float X0 = std::numeric_limits<float>::infinity(), Y0 = X0, X1 = -X0, Y1 = -X0;
  for (auto [x, y] : {std::pair{l, b}, {l, t}, {r, b}, {r, t}}) {
    const float px = parent.a * x + parent.c * y + parent.e, py = parent.b * x + parent.d * y + parent.f;
    X0 = std::min(X0, px); X1 = std::max(X1, px); Y0 = std::min(Y0, py); Y1 = std::max(Y1, py);
  }
  int fill = 0;
  FPDF_BOOL stroke = 0;
  const bool stroked = FPDFPath_GetDrawMode(obj, &fill, &stroke) && stroke != 0;
  float sw = 0;
  FPDFPageObj_GetStrokeWidth(obj, &sw);
  out.push_back({vp.rect(X0, Y0, X1, Y1), stroked, sw});
}

std::vector<Rule> page_rules(FPDF_PAGE page, const Viewport &vp, float page_h) {
  std::vector<PathBox> paths;
  const int nobj = FPDFPage_CountObjects(page);
  for (int i = 0; i < nobj; ++i)
    if (FPDF_PAGEOBJECT o = FPDFPage_GetObject(page, i)) collect_paths(o, Mat{}, vp, 0, paths);
  std::vector<Rule> rules;
  // Note: pdfium pads a stroked path's bounds by the stroke width on every side
  // (a 0.6pt fraction bar reads 1.2pt tall); take the geometric line.
  for (const auto &p : paths) {
    const float pad = p.stroked ? std::max(p.stroke_width, 0.0f) : 0.0f;
    const auto &b = p.box;
    const float x0 = b.x0 + pad, x1 = std::max(b.x1 - pad, b.x0 + pad);
    const float top = b.top + pad, bottom = std::max(b.bottom - pad, b.top + pad);
    if (auto r = Rule::from_box(x0, x1, top, bottom, page_h)) rules.push_back(*r);
  }
  // Note: some PDFs draw rules as stretched 1x1 images (top-level image objects).
  for (int i = 0; i < nobj; ++i) {
    FPDF_PAGEOBJECT o = FPDFPage_GetObject(page, i);
    if (!o || FPDFPageObj_GetType(o) != FPDF_PAGEOBJ_IMAGE) continue;
    float l, b, r, t;
    if (!FPDFPageObj_GetBounds(o, &l, &b, &r, &t)) continue;
    const auto v = vp.probe_rect(l, b, r, t);
    if (auto rule = Rule::from_box(v.x0, v.x1, v.top, v.bottom, page_h)) rules.push_back(*rule);
  }
  return rules;
}

// ── glyph collection ──────────────────────────────────────────────────────────

std::string font_base_name(FPDF_FONT f, bool &ok) {
  ok = false;
  if (!f) return {};
  const size_t n = FPDFFont_GetBaseFontName(f, nullptr, 0);
  if (n == 0) return {};
  std::string s(n, '\0');
  const size_t w = FPDFFont_GetBaseFontName(f, s.data(), n);
  if (w == 0) return {};
  s.resize(w > 0 && s[w - 1] == '\0' ? w - 1 : w);
  ok = true;
  return s;
}

std::string char_font_name(FPDF_TEXTPAGE tp, int i, bool &ok) {
  ok = false;
  int flags = 0;
  const unsigned long n = FPDFText_GetFontInfo(tp, i, nullptr, 0, &flags);
  if (n == 0) return {};
  std::string s(n, '\0');
  const unsigned long w = FPDFText_GetFontInfo(tp, i, s.data(), n, &flags);
  if (w == 0) return {};
  s.resize(s[w - 1] == '\0' ? w - 1 : w);
  ok = true;
  return s;
}

// The name LiteParse checks: the text object's font base name, else pdfium's per-char font name.
std::string glyph_font_name(FPDF_TEXTPAGE tp, int i, FPDF_PAGEOBJECT obj) {
  bool ok = false;
  std::string n = font_base_name(obj ? FPDFTextObj_GetFont(obj) : nullptr, ok);
  if (ok) return n;
  n = char_font_name(tp, i, ok);
  return ok ? n : std::string();
}

// extract.rs is_buggy_font (as glyph_decoder.cpp).
bool is_buggy_font(const std::string &name, int type) {
  if (name.rfind("TT", 0) == 0 || name.find("+TT") != std::string::npos) return true;
  return type == FPDF_FONTTYPE_TYPE1 && name.size() >= 7 && name[6] == '_';
}

std::optional<std::string> font_encoding(FPDF_FONT f) {
  const unsigned long n = FPDFFont_GetEncoding(f, nullptr, 0);
  if (n == 0) return std::nullopt;
  std::string s(n, '\0');
  const unsigned long w = FPDFFont_GetEncoding(f, s.data(), n);
  if (w == 0) return std::nullopt;
  s.resize(s[w - 1] == '\0' ? w - 1 : w);
  return s;
}

std::optional<std::string> char_glyph_name(FPDF_FONT f, uint32_t code) {
  const unsigned long n = FPDFFont_GetCharGlyphName(f, code, nullptr, 0);
  if (n == 0) return std::nullopt;
  std::string s(n, '\0');
  const unsigned long w = FPDFFont_GetCharGlyphName(f, code, s.data(), n);
  if (w == 0) return std::nullopt;
  s.resize(s[w - 1] == '\0' ? w - 1 : w);
  return s;
}

const char *presentation_form(char32_t c) {
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

void push_glyph_char(std::string &out, char32_t c) {
  if (const char *e = presentation_form(c)) out += e;
  else append_utf8(out, c);
}

// LiteParse glyph_names::resolve_glyph_name. hp's resolve_glyph_name
// (glyph_decoder.h) also folds Mathematical Alphanumeric Symbols to plain
// letters ("𝑽" -> "V"); LiteParse keeps them, and so does the math pass.
bool resolve_component(std::string_view comp, std::string &out) {
  auto agl_hex = [](std::string_view h) {
    return !h.empty() &&
           std::all_of(h.begin(), h.end(), [](char b) { return (b >= '0' && b <= '9') || (b >= 'A' && b <= 'F'); });
  };
  if (comp.empty()) return false;
  if (comp.size() > 3 && comp.substr(0, 3) == "uni" && (comp.size() - 3) % 4 == 0 && agl_hex(comp.substr(3))) {
    for (size_t k = 3; k < comp.size(); k += 4) {
      const uint32_t v = (uint32_t)std::stoul(std::string(comp.substr(k, 4)), nullptr, 16);
      if (!valid_scalar(v)) return false;
      push_glyph_char(out, v);
    }
    return true;
  }
  if (comp[0] == 'u' && comp.size() >= 5 && comp.size() <= 7 && agl_hex(comp.substr(1))) {
    const uint32_t v = (uint32_t)std::stoul(std::string(comp.substr(1)), nullptr, 16);
    if (!valid_scalar(v)) return false;
    push_glyph_char(out, v);
    return true;
  }
  const auto *b = std::begin(agl::kTable), *e = std::end(agl::kTable);
  const auto *it = std::lower_bound(b, e, comp, [](const agl::Entry &x, std::string_view k) { return x.name < k; });
  if (it == e || it->name != comp) return false;
  for (char32_t c : utf8_to_u32(it->text)) push_glyph_char(out, c);
  return true;
}

std::optional<std::string> lp_resolve_glyph_name(std::string_view name) {
  const std::string_view base = name.substr(0, name.find('.'));
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

// LiteParse's GlyphDecoder::decode as page_math_spans builds it (no garbage-font
// prescan, no outline resolver): glyph name -> AGL, else glyph index -> the
// embedded font's own cmap (hp::reverse_cmap). hp's GlyphDecoder adds math-alnum
// folding and "space" glyph recovery, which would change the math pass's input.
class LpGlyphDecoder {
 public:
  const std::string *decode(FPDF_TEXTPAGE tp, int index, FPDF_PAGEOBJECT obj, uint32_t unicode, bool map_error) {
    const bool cheap = unicode == 0 || unicode == 0xFFFE || unicode == 0xFFFF ||
                       (unicode < 0x20 && unicode != 0x09 && unicode != 0x0A && unicode != 0x0D) ||
                       (unicode >= 0xE000 && unicode <= 0xF8FF);
    if (!obj) return nullptr;
    if (obj != last_obj_) {
      FPDF_FONT f = FPDFTextObj_GetFont(obj);
      if (!f) return nullptr;
      auto [it, fresh] = fonts_.try_emplace(f);
      if (fresh) {
        const auto enc = font_encoding(f);
        bool named = false;
        const std::string name = font_base_name(f, named);
        const bool buggy = FPDFFont_GetIsEmbedded(f) == 1 && named && is_buggy_font(name, FPDFFont_GetType(f));
        const bool std_enc = enc && (*enc == "WinAnsiEncoding" || *enc == "MacRomanEncoding" ||
                                     *enc == "MacExpertEncoding" || *enc == "StandardEncoding");
        it->second.untrusted = buggy || (!FPDFFont_HasToUnicode(f) && !std_enc);
        it->second.encoding_lies = buggy;
      }
      last_obj_ = obj;
      last_font_ = f;
      last_ = &it->second;
    }
    Font &info = *last_;
    // The map-error check decides only when the cheap checks and font trust don't.
    if (!info.untrusted && !cheap && !map_error) return nullptr;
    const uint32_t code = FPDFText_GetCharCode(tp, index);
    auto [cit, fresh] = info.cache.try_emplace(code);
    if (fresh) {
      std::optional<std::string> r;
      // Glyph names of buggy-subset fonts derive from a lying base encoding: skip them.
      if (!info.encoding_lies)
        if (const auto nm = char_glyph_name(last_font_, code)) {
          r = lp_resolve_glyph_name(*nm);
          if (r)
            for (char32_t c : utf8_to_u32(*r))
              if (is_control(c)) { r.reset(); break; }
        }
      if (!r) {
        // Fallback: reverse-map the glyph index through the embedded font program's own cmap.
        const int g = FPDFFont_GetCharGlyphIndex(last_font_, code);
        if (g >= 0) {
          if (!info.cmap_built) {
            info.cmap_built = true;
            size_t len = 0;
            if (FPDFFont_GetFontData(last_font_, nullptr, 0, &len) && len > 0) {
              std::vector<uint8_t> data(len);
              if (FPDFFont_GetFontData(last_font_, data.data(), len, &len))
                info.cmap_ok = reverse_cmap(data.data(), len, info.rcmap);
            }
          }
          if (info.cmap_ok)
            if (auto m = info.rcmap.find((uint32_t)g); m != info.rcmap.end()) {
              const uint32_t u = m->second;
              // Synthetic subset cmaps echo the char code back: not a real mapping.
              if (!(u >= 0xE000 && u <= 0xF8FF) && !(u == code && u != unicode) && valid_scalar(u) && !is_control(u)) {
                std::string s;
                push_glyph_char(s, u);
                r = std::move(s);
              }
            }
        }
      }
      cit->second = std::move(r);
    }
    const auto &r = cit->second;
    if (!r) return nullptr;
    // Don't double-expand a ligature pdfium already split.
    if (!cheap && valid_scalar(unicode)) {
      const auto u = utf8_to_u32(*r);
      if (u.size() > 1 && u.find((char32_t)unicode) != std::u32string::npos) return nullptr;
    }
    return &*r;
  }

 private:
  struct Font {
    bool untrusted = false, encoding_lies = false, cmap_built = false, cmap_ok = false;
    std::unordered_map<uint32_t, uint32_t> rcmap;
    std::unordered_map<uint32_t, std::optional<std::string>> cache;
  };
  std::unordered_map<FPDF_FONT, Font> fonts_;
  FPDF_PAGEOBJECT last_obj_ = nullptr;
  FPDF_FONT last_font_ = nullptr;
  Font *last_ = nullptr;
};

float adjust_angle_for_rotation(float a, int rotation) {
  constexpr float PI = std::numbers::pi_v<float>;
  switch (rotation) {
    case 1: a -= 3.0f * PI / 2.0f; break;
    case 2: a -= PI; break;
    case 3: a -= PI / 2.0f; break;
    default: break;
  }
  const float m = 2.0f * PI;
  float r = std::fmod(a, m);  // Rust f32::rem_euclid
  if (r < 0) r += m;
  a = r;
  // Float error can leave a full turn a hair under 2pi.
  if (2.0f * PI - a < 1e-4f) a = 0.0f;
  return a;
}

// Smaller scale factor of a 2D affine matrix (eigenvalues of M^T M), LiteParse decompose_scale().1.
float decompose_scale_y(const FS_MATRIX &m) {
  const double a = m.a, b = m.b, c = m.c, d = m.d;
  const double mt_a = a * a + b * b, mt_b = a * c + b * d, mt_d = c * c + d * d;
  const double first = (mt_a + mt_d) / 2.0;
  const double disc = std::sqrt((mt_a + mt_d) * (mt_a + mt_d) - 4.0 * (mt_a * mt_d - mt_b * mt_b)) / 2.0;
  const double sy = std::sqrt(first - disc);
  return std::isnan(sy) ? 1.0f : (float)sy;
}

struct ObjInfo {
  std::string name;
  float size = 0, descent = 0;
  FPDF_FONT no_tu_font = nullptr;  // the font when it has no ToUnicode
};

// Display: nothing but math on the span's line within 2 em on either side.
void mark_display(std::vector<MathSpan> &spans, const std::vector<MathSpan> &emitted, const std::vector<RawChar> &chars) {
  std::unordered_set<int> math;
  for (const auto &s : emitted) math.insert(s.chars.begin(), s.chars.end());
  for (auto &s : spans) {
    const float w0 = s.x0 - 2.0f * s.size, w1 = s.x1 + 2.0f * s.size, h = s.bottom - s.top;
    bool alone = true;
    for (const auto &c : chars) {
      if (math.count(c.index) || (!c.is_code && is_ws(c.value))) continue;
      if (c.x1 <= w0 || c.x0 >= w1) continue;
      const float ov = std::min(s.bottom, c.loose_bottom) - std::max(s.top, c.loose_top);
      if (ov > 0.5f * std::min(h, c.loose_bottom - c.loose_top)) { alone = false; break; }
    }
    s.display = alone;
  }
}

}  // namespace

std::vector<MathSpan> page_math(void *fpdf_page, void *fpdf_textpage, bool keep_clipped_text, bool all_spans) {
  auto page = static_cast<FPDF_PAGE>(fpdf_page);
  auto tp = static_cast<FPDF_TEXTPAGE>(fpdf_textpage);
  if (!page || !tp) return {};
  const int n = FPDFText_CountChars(tp);
  if (n <= 0) return {};
  // Most pages have no TeX math font: check each text object's font once before
  // paying for glyph boxes and decoding.
  {
    std::unordered_set<FPDF_PAGEOBJECT> seen;
    bool has_tex_font = false;
    for (int i = 0; i < n && !has_tex_font; ++i) {
      FPDF_PAGEOBJECT obj = FPDFText_GetTextObject(tp, i);
      if (!seen.insert(obj).second) continue;
      has_tex_font = is_tex_math_font(glyph_font_name(tp, i, obj));
    }
    if (!has_tex_font) return {};
  }
  const TextClip clips(page, tp, keep_clipped_text);
  const int rotation = FPDFPage_GetRotation(page);
  const float w_pt = FPDF_GetPageWidthF(page), page_h = FPDF_GetPageHeightF(page);
  Viewport vp;
  vp.init(page, w_pt, page_h);
  // The same glyph-name recovery as the text pass: a Symbol font with no ToUnicode
  // otherwise reads as Latin ("t" for tau).
  LpGlyphDecoder decoder;
  std::unordered_map<FPDF_PAGEOBJECT, ObjInfo> objs;
  // (font, char code) -> glyph-name text.
  struct PairHash {
    size_t operator()(const std::pair<FPDF_FONT, uint32_t> &p) const {
      return std::hash<void *>()(p.first) ^ (std::hash<uint32_t>()(p.second) * 0x9E3779B97F4A7C15ull);
    }
  };
  std::unordered_map<std::pair<FPDF_FONT, uint32_t>, std::optional<char32_t>, PairHash> names;
  std::vector<RawChar> chars;
  chars.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (FPDFText_IsGenerated(tp, i) == 1) continue;
    FPDF_PAGEOBJECT obj = FPDFText_GetTextObject(tp, i);
    if (obj && FPDFTextObj_GetTextRenderMode(obj) == FPDF_TEXTRENDERMODE_INVISIBLE) continue;
    if (clips.hides(i)) continue;
    // Upright glyphs only.
    const float angle = adjust_angle_for_rotation(std::max(FPDFText_GetCharAngle(tp, i), 0.0f), rotation);
    if (angle > 0.01f && angle < 2.0f * std::numbers::pi_v<float> - 0.01f) continue;
    auto oit = objs.find(obj);
    if (oit == objs.end()) {
      ObjInfo oi;
      FPDF_FONT font = obj ? FPDFTextObj_GetFont(obj) : nullptr;
      oi.name = glyph_font_name(tp, i, obj);
      const float fs = (float)FPDFText_GetFontSize(tp, i);
      FS_MATRIX m;
      const float scale = FPDFText_GetMatrix(tp, i, &m) ? decompose_scale_y(m) : 1.0f;
      float desc = 0;
      if (!(font && FPDFFont_GetDescent(font, fs, &desc))) desc = 0;
      oi.size = fs * scale;
      oi.descent = desc * scale;
      if (font && !FPDFFont_HasToUnicode(font)) oi.no_tu_font = font;
      oit = objs.emplace(obj, std::move(oi)).first;
    }
    const ObjInfo &oi = oit->second;
    const uint32_t unicode = FPDFText_GetUnicode(tp, i);
    const bool map_error = FPDFText_HasUnicodeMapError(tp, i) == 1;
    // Note: with no ToUnicode, the encoding's glyph name is the truth: a Symbol font
    // under WinAnsiEncoding + Differences names "tau" for the code pdfium reads as "t".
    std::optional<char32_t> text;
    if (oi.no_tu_font) {
      const uint32_t code = FPDFText_GetCharCode(tp, i);
      auto [it, fresh] = names.try_emplace({oi.no_tu_font, code});
      if (fresh) {
        if (const auto nm = char_glyph_name(oi.no_tu_font, code))
          if (const auto r = lp_resolve_glyph_name(*nm)) {
            const auto u = utf8_to_u32(*r);
            if (u.size() == 1 && !is_control(u[0])) it->second = u[0];
          }
      }
      text = it->second;
    }
    if (!text)
      if (const std::string *d = decoder.decode(tp, i, obj, unicode, map_error); d && !d->empty())
        text = utf8_to_u32(*d)[0];
    RawChar rc;
    rc.index = i;
    if (text) {
      rc.value = *text;
    } else if (map_error || unicode < 0x20) {
      rc.is_code = true;
      rc.value = FPDFText_GetCharCode(tp, i);
    } else if (valid_scalar(unicode)) {
      rc.value = unicode;
    } else {
      continue;
    }
    double ox, oy;
    FS_RECTF lr;
    if (!FPDFText_GetCharOrigin(tp, i, &ox, &oy) || !FPDFText_GetLooseCharBox(tp, i, &lr)) continue;
    const auto loose = vp.rect(lr.left, lr.bottom, lr.right, lr.top);
    rc.origin_y = vp.point((float)ox, (float)oy).second;
    double l, r, b, t;
    if (FPDFText_GetCharBox(tp, i, &l, &r, &b, &t)) {
      const auto ink = vp.rect((float)l, (float)b, (float)r, (float)t);
      rc.has_ink = true;
      rc.ink_top = ink.top;
      rc.ink_bottom = ink.bottom;
    }
    rc.font_name = &oi.name;
    rc.size = oi.size > 0.0f ? oi.size : loose.bottom - loose.top;
    rc.descent = oi.descent;
    rc.x0 = loose.x0;
    rc.x1 = loose.x1;
    rc.loose_top = loose.top;
    rc.loose_bottom = loose.bottom;
    chars.push_back(rc);
  }
  const std::vector<Rule> rules = page_rules(page, vp, page_h);
  auto spans = detail::page_math(chars, rules, page_h);
  if (!spans) return {};
  std::vector<MathSpan> emitted;
  for (const auto &s : *spans)
    if (has_structure(s.tex)) emitted.push_back(s);
  if (all_spans) {
    mark_display(*spans, emitted, chars);
    return std::move(*spans);
  }
  mark_display(emitted, emitted, chars);
  return emitted;
}

}  // namespace hp::tex
