// TeX math font encodings (OMS = cmsy, OMX = cmex, OML = cmmi, msam): glyph code
// -> LaTeX, for glyphs pdfium could not map to Unicode. Port of LiteParse's
// tex_math/encodings.rs.

#include "detail.h"

namespace hp::tex::detail {
namespace {

using SV = std::optional<std::string_view>;

SV oms(uint32_t code) {
  static constexpr std::string_view lo[65] = {
      "-",          R"(\cdot)",     R"(\times)",         R"(\ast)",       R"(\div)",        R"(\diamond)",
      R"(\pm)",     R"(\mp)",       R"(\oplus)",         R"(\ominus)",    R"(\otimes)",     R"(\oslash)",
      R"(\odot)",   R"(\bigcirc)",  R"(\circ)",          R"(\bullet)",    R"(\asymp)",      R"(\equiv)",
      R"(\subseteq)", R"(\supseteq)", R"(\leq)",         R"(\geq)",       R"(\preceq)",     R"(\succeq)",
      R"(\sim)",    R"(\approx)",   R"(\subset)",        R"(\supset)",    R"(\ll)",         R"(\gg)",
      R"(\prec)",   R"(\succ)",     R"(\leftarrow)",     R"(\to)",        R"(\uparrow)",    R"(\downarrow)",
      R"(\leftrightarrow)", R"(\nearrow)", R"(\searrow)", R"(\simeq)",    R"(\Leftarrow)",  R"(\Rightarrow)",
      R"(\Uparrow)", R"(\Downarrow)", R"(\Leftrightarrow)", R"(\nwarrow)", R"(\swarrow)",  R"(\propto)",
      "'",          R"(\infty)",    R"(\in)",            R"(\ni)",        R"(\triangle)",   R"(\triangledown)",
      "NOT",        "MAPSTO",       R"(\forall)",        R"(\exists)",    R"(\neg)",        R"(\emptyset)",
      R"(\Re)",     R"(\Im)",       R"(\top)",           R"(\perp)",      R"(\aleph)"};
  if (code <= 64) return lo[code];
  static constexpr std::string_view hi[37] = {
      R"(\cup)",     R"(\cap)",         R"(\uplus)",   R"(\wedge)",     R"(\vee)",         R"(\vdash)",
      R"(\dashv)",   R"(\lfloor)",      R"(\rfloor)",  R"(\lceil)",     R"(\rceil)",       R"(\{)",
      R"(\})",       R"(\langle)",      R"(\rangle)",  "|",             R"(\|)",           R"(\updownarrow)",
      R"(\Updownarrow)", R"(\setminus)", R"(\wr)",     R"(\surd)",      R"(\amalg)",       R"(\nabla)",
      R"(\int)",     R"(\sqcup)",       R"(\sqcap)",   R"(\sqsubseteq)", R"(\sqsupseteq)", R"(\S)",
      R"(\dagger)",  R"(\ddagger)",     R"(\P)",       R"(\clubsuit)",  R"(\diamondsuit)", R"(\heartsuit)",
      R"(\spadesuit)"};
  if (code >= 91 && code <= 127) return hi[code - 91];
  return std::nullopt;
}

// cmex: big delimiters read as their plain selves (KaTeX renders \big( as "(").
// Extension pieces of tall delimiters: the top piece stands for the delimiter,
// the rest map to "".
SV omx(uint32_t code) {
  static constexpr std::string_view d[16] = {"(",        ")",        "[",         "]",
                                             R"(\lfloor)", R"(\rfloor)", R"(\lceil)", R"(\rceil)",
                                             R"(\{)",     R"(\})",     R"(\langle)", R"(\rangle)",
                                             "|",         R"(\|)",     "/",          R"(\backslash)"};
  if (code < 16) return d[code];
  switch (code) {
    case 16: case 18: case 32: case 48: return "(";
    case 17: case 19: case 33: case 49: return ")";
    case 20: case 34: case 50: case 104: return "[";
    case 21: case 35: case 51: case 105: return "]";
    case 22: case 36: case 106: return R"(\lfloor)";
    case 23: case 37: case 107: return R"(\rfloor)";
    case 24: case 38: case 108: return R"(\lceil)";
    case 25: case 39: case 109: return R"(\rceil)";
    case 26: case 40: case 56: case 110: return R"(\{)";
    case 27: case 41: case 57: case 111: return R"(\})";
    case 28: case 42: case 68: return R"(\langle)";
    case 29: case 43: case 69: return R"(\rangle)";
    case 30: case 44: case 46: return "/";
    case 31: case 45: case 47: return R"(\backslash)";
    case 70: case 71: return R"(\bigsqcup)";
    case 72: case 73: return R"(\oint)";
    case 74: case 75: return R"(\bigodot)";
    case 76: case 77: return R"(\bigoplus)";
    case 78: case 79: return R"(\bigotimes)";
    case 80: case 88: return R"(\sum)";
    case 81: case 89: return R"(\prod)";
    case 82: case 90: return R"(\int)";
    case 83: case 91: return R"(\bigcup)";
    case 84: case 92: return R"(\bigcap)";
    case 85: case 93: return R"(\biguplus)";
    case 86: case 94: return R"(\bigwedge)";
    case 87: case 95: return R"(\bigvee)";
    case 96: case 97: return R"(\coprod)";
    default: break;
  }
  if ((code >= 52 && code <= 55) || (code >= 58 && code <= 67)) return "";
  if (code >= 98 && code <= 100) return "ACC:widehat";
  if (code >= 101 && code <= 103) return "ACC:widetilde";
  if (code >= 112 && code <= 116) return "SQRT";
  return std::nullopt;
}

// cmmi non-letters.
SV oml(uint32_t code) {
  static constexpr std::string_view greek[40] = {
      R"(\Gamma)",   R"(\Delta)",  R"(\Theta)",      R"(\Lambda)",   R"(\Xi)",      R"(\Pi)",       R"(\Sigma)",
      R"(\Upsilon)", R"(\Phi)",    R"(\Psi)",        R"(\Omega)",    R"(\alpha)",   R"(\beta)",     R"(\gamma)",
      R"(\delta)",   R"(\epsilon)", R"(\zeta)",      R"(\eta)",      R"(\theta)",   R"(\iota)",     R"(\kappa)",
      R"(\lambda)",  R"(\mu)",     R"(\nu)",         R"(\xi)",       R"(\pi)",      R"(\rho)",      R"(\sigma)",
      R"(\tau)",     R"(\upsilon)", R"(\phi)",       R"(\chi)",      R"(\psi)",     R"(\omega)",    R"(\varepsilon)",
      R"(\vartheta)", R"(\varpi)", R"(\varrho)",     R"(\varsigma)", R"(\varphi)"};
  if (code < 40) return greek[code];
  switch (code) {
    case 58: return ".";
    case 59: return ",";
    case 60: return "<";
    case 61: return "/";
    case 62: return ">";
    case 63: return R"(\star)";
    case 64: return R"(\partial)";
    case 91: return R"(\flat)";
    case 92: return R"(\natural)";
    case 93: return R"(\sharp)";
    case 96: return R"(\ell)";
    case 123: return R"(\imath)";
    case 124: return R"(\jmath)";
    case 125: return R"(\wp)";
    case 126: return "ACC:vec";
    default: return std::nullopt;
  }
}

SV msam(uint32_t code) {
  switch (code) {
    case 3: return R"(\square)";
    case 4: return R"(\blacksquare)";
    case 6: return R"(\Diamond)";
    case 54: case 44: return R"(\leqslant)";
    case 62: return R"(\geqslant)";
    case 38: return R"(\gtrsim)";
    case 45: return R"(\eqslantless)";
    case 46: return R"(\lesssim)";
    default: return std::nullopt;
  }
}

}  // namespace

std::optional<std::string_view> decode(Enc enc, uint32_t code) {
  switch (enc) {
    case Enc::Oms: return oms(code);
    case Enc::Omx: return omx(code);
    case Enc::Oml: return oml(code);
    case Enc::Msam: return msam(code);
    default: return std::nullopt;
  }
}

}  // namespace hp::tex::detail
