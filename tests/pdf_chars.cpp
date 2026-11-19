// Debug dump of pdfium's per-char view of one page (fork APIs included):
//   pdf_chars <in.pdf> <page0> [first_char] [count]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <fpdf_edit.h>
#include <fpdf_text.h>
#include <fpdfview.h>
#include <map>
#include <unordered_map>
#include "glyph_decoder.h"

int main(int argc, char **argv) {
  if (argc < 3) return 1;
  std::ifstream f(argv[1], std::ios::binary);
  const std::vector<char> pdf((std::istreambuf_iterator<char>(f)), {});
  FPDF_InitLibrary();
  FPDF_DOCUMENT doc = FPDF_LoadMemDocument(pdf.data(), (int)pdf.size(), nullptr);
  FPDF_PAGE page = FPDF_LoadPage(doc, std::atoi(argv[2]));
  FPDF_TEXTPAGE tp = FPDFText_LoadPage(page);
  const int n = FPDFText_CountChars(tp);
  const int a = argc > 3 ? std::atoi(argv[3]) : 0, cnt = argc > 4 ? std::atoi(argv[4]) : n;
  std::printf("%d chars\n", n);
  for (int i = a; i < std::min(n, a + cnt); ++i) {
    const unsigned u = FPDFText_GetUnicode(tp, i);
    FPDF_PAGEOBJECT o = FPDFText_GetTextObject(tp, i);
    FPDF_FONT ft = o ? FPDFTextObj_GetFont(o) : nullptr;
    char name[128] = "", gname[64] = "", enc[64] = "";
    if (ft) {
      FPDFFont_GetBaseFontName(ft, name, sizeof name);
      FPDFFont_GetCharGlyphName(ft, FPDFText_GetCharCode(tp, i), gname, sizeof gname);
      FPDFFont_GetEncoding(ft, enc, sizeof enc);
    }
    double x0, x1, y0, y1;
    FPDFText_GetCharBox(tp, i, &x0, &x1, &y0, &y1);
    int gidx = ft ? FPDFFont_GetCharGlyphIndex(ft, FPDFText_GetCharCode(tp, i)) : -1;
    static std::map<void *, std::unordered_map<uint32_t, uint32_t>> rc;
    if (ft && !rc.count(ft)) {
      size_t len = 0;
      std::vector<uint8_t> d;
      if (FPDFFont_GetFontData(ft, nullptr, 0, &len) && len) { d.resize(len); FPDFFont_GetFontData(ft, d.data(), len, &len); }
      hp::reverse_cmap(d.data(), d.size(), rc[ft]);
    }
    const unsigned cm = (ft && rc[ft].count(gidx)) ? rc[ft][gidx] : 0;
    std::printf("gid=%d cmap=U+%04X | ", gidx, cm);
    std::printf("%5d U+%04X '%c' gen=%d err=%d code=%u glyph=%s font=%s tu=%d enc=%s type=%d mode=%d box=%.1f,%.1f\n", i, u,
                (u >= 32 && u < 127) ? (char)u : '?', FPDFText_IsGenerated(tp, i), FPDFText_HasUnicodeMapError(tp, i),
                FPDFText_GetCharCode(tp, i), gname, name, ft ? (int)FPDFFont_HasToUnicode(ft) : -1, enc,
                ft ? (int)FPDFFont_GetType(ft) : -1, o ? FPDFTextObj_GetTextRenderMode(o) : -1, x0, y1);
  }
  return 0;
}
