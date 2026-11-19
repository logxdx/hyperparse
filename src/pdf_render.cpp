#include "pdf_render.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <mutex>

#include <fpdfview.h>

#include "core/pdf/pdf_text_layer.h"
#include "core/pdf/text/pdf_text_internal.h"

namespace hp {

using hp::core::pdf::detail::pdfium_lock;

PdfRaster::PdfRaster(const uint8_t *data, size_t len) {
  hp::core::pdf::ensure_pdfium_initialized();
  if (len > static_cast<size_t>(INT_MAX)) return;
  std::lock_guard<std::mutex> g(pdfium_lock());
  doc_ = FPDF_LoadMemDocument(data, static_cast<int>(len), nullptr);
}

PdfRaster::~PdfRaster() {
  if (!doc_) return;
  std::lock_guard<std::mutex> g(pdfium_lock());
  FPDF_CloseDocument(static_cast<FPDF_DOCUMENT>(doc_));
}

int PdfRaster::page_count() const {
  if (!doc_) return 0;
  std::lock_guard<std::mutex> g(pdfium_lock());
  return FPDF_GetPageCount(static_cast<FPDF_DOCUMENT>(doc_));
}

void PdfRaster::page_size_pt(int i, float &w, float &h) const {
  std::lock_guard<std::mutex> g(pdfium_lock());
  FS_SIZEF sz{};
  FPDF_GetPageSizeByIndexF(static_cast<FPDF_DOCUMENT>(doc_), i, &sz);
  w = sz.width;
  h = sz.height;
}

void PdfRaster::page_size_px(int i, float dpi, int &w, int &h) const {
  float pw = 0, ph = 0;
  page_size_pt(i, pw, ph);
  w = std::max(1, static_cast<int>(std::lround(pw * dpi / 72.0)));
  h = std::max(1, static_cast<int>(std::lround(ph * dpi / 72.0)));
}

bool PdfRaster::render_bgr(int i, uint8_t *dst, int w, int h) const {
  std::lock_guard<std::mutex> g(pdfium_lock());
  FPDF_PAGE page = FPDF_LoadPage(static_cast<FPDF_DOCUMENT>(doc_), i);
  if (!page) return false;
  FPDF_BITMAP bmp = FPDFBitmap_CreateEx(w, h, FPDFBitmap_BGR, dst, w * 3);
  bool ok = bmp != nullptr;
  if (ok) {
    FPDFBitmap_FillRect(bmp, 0, 0, w, h, 0xFFFFFFFF);
    FPDF_RenderPageBitmap(bmp, page, 0, 0, w, h, 0, FPDF_ANNOT);
    FPDFBitmap_Destroy(bmp);
  }
  FPDF_ClosePage(page);
  return ok;
}

}  // namespace hp
