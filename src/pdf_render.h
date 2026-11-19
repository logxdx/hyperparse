#pragma once
// Minimal pdfium page rasteriser: renders a page as packed u8 BGR straight into
// a caller-owned (pinned) buffer, so the GPU upload needs no staging copy.
// All pdfium calls hold hp::core::pdf::detail::pdfium_lock() (pdfium is not
// thread-safe; the text-layer code in core/pdf takes the same lock).

#include <cstddef>
#include <cstdint>

namespace hp {

class PdfRaster {
 public:
  PdfRaster(const uint8_t *data, size_t len);
  ~PdfRaster();
  PdfRaster(const PdfRaster &) = delete;
  PdfRaster &operator=(const PdfRaster &) = delete;

  [[nodiscard]] bool ok() const { return doc_ != nullptr; }
  [[nodiscard]] void *doc() const { return doc_; }
  [[nodiscard]] int page_count() const;
  // Page size in PDF points.
  void page_size_pt(int i, float &w, float &h) const;
  // Pixel size of page `i` at `dpi`.
  void page_size_px(int i, float dpi, int &w, int &h) const;
  // Render into dst (w*h*3 bytes, stride w*3). Returns false on failure.
  bool render_bgr(int i, uint8_t *dst, int w, int h) const;

 private:
  void *doc_ = nullptr;
};

}  // namespace hp
