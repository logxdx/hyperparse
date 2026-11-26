#pragma once
// PDF text layer + routing, ported from liteparse (crates/liteparse/src/extract.rs,
// ocr_merge.rs; thresholds and formulas kept), extended from liteparse's per-PAGE
// decision to per-REGION: only the parts of a page without a usable text layer
// (embedded images, items in garbled fonts / unusable items, outline text) are
// OCR'd; the rest keeps its native text.
//
// All functions take a pdfium FPDF_DOCUMENT as void* and must run where pdfium
// is serialised (a render worker process, or under pdfium_lock()).

#include <cstdint>
#include <string>
#include <vector>

namespace hp {

struct RectF {
  float x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // viewport points: origin top-left, y down
  [[nodiscard]] float w() const { return x1 - x0; }
  [[nodiscard]] float h() const { return y1 - y0; }
};

// Inline style of a text item (liteparse SpanStyle; rendered by style_md.h).
enum StyleBit : uint8_t {
  kStyleBold = 1, kStyleItalic = 2, kStyleMono = 4, kStyleStrike = 8, kStyleUnderline = 16,
  kStyleSup = 32, kStyleSub = 64,
  kStyleMath = 128,  // the item is a TeX math span ("$...$", tex_math): never wrapped in emphasis
};

struct TextItem {
  std::string text;  // UTF-8
  RectF box;
  int font = -1;     // index into PageText::fonts
  int chars = 0, unmapped = 0;
  bool unusable = false;  // liteparse is_unusable_native: unicode-map-error majority or vowel ratio < 10%
  bool dropped = false;   // set by route_page (garbled font / unusable / replaced page)
  uint8_t style = 0;      // StyleBit set
};

struct PageText {
  float w_pt = 0, h_pt = 0;
  std::vector<std::string> fonts;
  std::vector<TextItem> items;
  std::vector<RectF> images;   // top-level image objects >= 25 pt, not full-page
  bool full_page_image = false;
  std::vector<RectF> paths;    // filled vector paths (outline-text candidates)
  float invisible_share = 0, hidden_under_scan_share = 0;
  int unmapped_chars = 0, nonspace_chars = 0;
  int recovered = 0;  // chars whose text came from glyph names / the font's cmap
  // Gaps between visible neighbours that hold glyphs whose ToUnicode claims "space"
  // (Word math). pdfium collapses such a run into ONE space char, so the text is
  // unrecoverable from the text layer; the span is read from pixels instead.
  std::vector<RectF> lie_spans;
};

enum RouteKind : int { kNative = 0, kPartial = 1, kFullOcr = 2 };
enum Reason : unsigned {
  kNoText = 1u << 0, kScanned = 1u << 1, kSparseText = 1u << 2, kImages = 1u << 3,
  kGarbledPage = 1u << 4, kGarbledFonts = 1u << 5, kUnusableItems = 1u << 6,
  kInvisibleLayer = 1u << 7, kVectorText = 1u << 8, kAnnotationText = 1u << 9, kLyingGlyphs = 1u << 10,
};

struct Route {
  int kind = kNative;
  bool keep_native = true;  // native items (not dropped) win over overlapping OCR
  unsigned reasons = 0;
  std::vector<RectF> regions;    // kPartial: run det here (images, outline text) (viewport pt)
  std::vector<RectF> rec_boxes;  // kPartial: dropped native items -> recognise these boxes directly, no det
  int text_len = 0;            // usable native text, UTF-8 bytes (liteparse units)
  float coverage = 0;
};

PageText extract_page_text(void *fpdf_document, int page_index);
Route route_page(PageText &pt);  // marks items dropped
std::string reasons_str(unsigned reasons);

// liteparse overlaps_existing_text: any intersection with `tol` pt slack.
bool overlaps_native(const PageText &pt, const RectF &r, float tol = 2.0f);

// Compact binary form for the worker -> parent hand-off.
void serialize(const PageText &pt, const Route &rt, std::vector<uint8_t> &out);
bool deserialize(const uint8_t *p, size_t n, PageText &pt, Route &rt);

}  // namespace hp
