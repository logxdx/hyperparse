#pragma once
// LayoutNet text units: runs of text the writer put down together, cut wherever a gap is wide
// enough to separate two table cells. A port of liteparse table_model/units.rs (pdf_units and the
// OCR-item units of mod.rs ocr_units). Points, top-left origin, rendered-page frame.
//
// native_units() reads PDFium's characters, so it must run where PDFium is serialised (a render
// worker). It has to run before extract_page_text on the same page: PDFium reports character
// boxes differently once that extraction has run (as the reference notes).

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hp::tn {

struct Word {
  double x0, x1;
  std::string text;
};

struct Unit {
  std::array<double, 4> bx{};    // reading-frame box
  std::array<double, 4> pbox{};  // rendered-page box
  std::string text;
  double h = 0, size = 0;
  int8_t bold = -1;  // -1 unknown, 0, 1
  uint8_t q = 0;     // reading direction on the page: right, up, left, down = 0..3
  bool minor = false;  // runs against the page's main direction (a sidebar, a rotated label)
  std::vector<Word> words;  // reading frame, for split_aligned
};

// Pages with more units than this skip the model (attention and decoding grow with N^2).
inline constexpr size_t kMaxUnits = 1500;

struct PageUnits {
  std::vector<Unit> units;
  double w = 0, h = 0;  // rendered page size, points
};

// Units of a page's text layer (empty when it has none). `fpdf_page` is an FPDF_PAGE.
PageUnits native_units(void *fpdf_page);
// The same for page `page_index` of an FPDF_DOCUMENT (loads and closes the page).
PageUnits native_units_doc(void *fpdf_document, int page_index);

// One unit per OCR line (mod.rs ocr_units): box in page points, height un-padded (/1.4).
Unit ocr_unit(std::string_view text, double x0, double y0, double x1, double y1);

// reading frame <-> page (q quarter turns), as units.rs to_frame / from_frame
std::array<double, 4> to_frame(const std::array<double, 4> &b, int q, double w, double h);
std::array<double, 4> from_frame(const std::array<double, 4> &b, int q, double w, double h);

// Compact binary form for the worker -> parent hand-off (appended to the page meta).
void serialize(const PageUnits &pu, std::vector<uint8_t> &out);
// Reads one PageUnits at p (n bytes left); returns bytes consumed, 0 on malformed input.
size_t deserialize(const uint8_t *p, size_t n, PageUnits &pu);

}  // namespace hp::tn
