// LaTeX from the glyphs of TeX-generated PDF pages: port of LiteParse's
// tex_math/mod.rs (see tex_math.h). Comments marked "Note:" carry the
// prototype's traps.

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <utility>

#include "detail.h"
#include "unicode_alpha.h"

namespace hp::tex::detail {

// ── UTF-8 / Unicode helpers (Rust char semantics) ─────────────────────────────

std::u32string utf8_to_u32(std::string_view s) {
  std::u32string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i];
    const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
    char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
    out += cp;
    i += n;
  }
  return out;
}

void append_utf8(std::string &s, char32_t c) {
  if (c < 0x80) s += (char)c;
  else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
  else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
  else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
}

namespace {

bool alphabetic(char32_t c) {
  if (c < 0x80) return (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
  const auto *b = std::begin(kAlphabetic), *e = std::end(kAlphabetic);
  const auto *it = std::upper_bound(b, e, (uint32_t)c, [](uint32_t v, const AlphaRange &r) { return v < r.lo; });
  return it != b && (uint32_t)c <= (it - 1)->hi;
}

// Rust char::is_whitespace (Unicode White_Space).
bool whitespace(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// Rust str::to_uppercase for the letters a \mathbb glyph can carry (ASCII,
// Latin-1, basic Greek); other characters are kept.
std::string to_upper(std::string_view t) {
  std::string out;
  for (char32_t c : utf8_to_u32(t)) {
    if (c >= 'a' && c <= 'z') c -= 0x20;
    else if (c == 0xDF) { out += "SS"; continue; }
    else if (c >= 0xE0 && c <= 0xFE && c != 0xF7) c -= 0x20;
    else if (c == 0xFF) c = 0x178;
    else if (c == 0xB5) c = 0x39C;
    else if (c == 0x3C2) c = 0x3A3;
    else if (c >= 0x3B1 && c <= 0x3C9) c -= 0x20;
    append_utf8(out, c);
  }
  return out;
}

// Exactly one char in `t`; returns it.
std::optional<char32_t> single(std::string_view t) {
  if (t.empty()) return std::nullopt;
  const auto u = utf8_to_u32(t);
  if (u.size() != 1) return std::nullopt;
  return u[0];
}

bool contains_char(std::string_view set, char32_t c) {
  for (char32_t x : utf8_to_u32(set))
    if (x == c) return true;
  return false;
}

// Rust str::contains(&str): substring (true for an empty needle).
bool contains_sub(std::string_view hay, std::string_view needle) { return hay.find(needle) != std::string_view::npos; }

std::string_view trim(std::string_view s) {
  const auto u = utf8_to_u32(s);
  size_t b = 0, e = u.size();
  while (b < e && whitespace(u[b])) ++b;
  while (e > b && whitespace(u[e - 1])) --e;
  // byte offsets of the kept range
  size_t bb = 0, eb = 0, k = 0;
  for (size_t i = 0; i < s.size(); ++k) {
    const unsigned char c = s[i];
    const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
    if (k == b) bb = i;
    i += n;
    if (k + 1 == e) eb = i;
  }
  if (b >= e) return s.substr(0, 0);
  return s.substr(bb, eb - bb);
}

std::string replace_all(std::string s, std::string_view from, std::string_view to) {
  std::string out;
  size_t pos = 0;
  while (true) {
    const size_t p = s.find(from, pos);
    if (p == std::string::npos) break;
    out.append(s, pos, p - pos);
    out += to;
    pos = p + from.size();
  }
  out.append(s, pos, std::string::npos);
  return out;
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

// ── font roles ────────────────────────────────────────────────────────────────

// `prefix` + `rest` where the prefix is one of the tx / px font families: does
// any rest satisfy `pred`?
template <size_t N, class P>
bool tx_px(std::string_view f, const std::string_view (&prefixes)[N], P pred) {
  for (std::string_view p : prefixes)
    if (starts_with(f, p) && pred(f.substr(p.size()))) return true;
  return false;
}

bool any_of_sv(std::string_view f, std::initializer_list<std::string_view> xs) {
  for (auto x : xs)
    if (f == x) return true;
  return false;
}

// Math font families (subset prefix and size stripped, upper case), checked in
// order: (role, encoding). The encoding is set when the font follows cmmi / cmsy
// / cmex (OML / OMS / OMX): a glyph without Unicode is then read through that
// table, and a capital in an OMS font is \mathcal.
std::pair<Role, Enc> font_info(std::string_view f) {
  static constexpr std::string_view TXPX[4] = {"TX", "RTX", "PX", "RPX"};
  static constexpr std::string_view TXPX_NEW[5] = {"TX", "RTX", "PX", "RPX", "NEWTX"};
  static constexpr std::string_view TXPX_MI[7] = {"TX", "RTX", "PX", "RPX", "NEWTX", "ZX", "NTX"};
  const bool mi_oml = any_of_sv(f, {"CMMI", "CMMIB", "LMMATHITALIC", "MLMMATHITALIC", "CMBRMI"}) ||
                      tx_px(f, TXPX_MI, [](std::string_view r) { return starts_with(r, "MI") || starts_with(r, "BMI"); });
  if (mi_oml) return {Role::Mi, Enc::Oml};
  if (any_of_sv(f, {"EURM", "EURB", "PAZOMATH-ITALIC", "STIXMATH-ITALIC"}) || contains_sub(f, "MATHITALIC"))
    return {Role::Mi, Enc::None};
  if (any_of_sv(f, {"CMSY", "CMBSY", "LMMATHSYMBOLS", "TXBSY", "TXBSYS"}) ||
      tx_px(f, TXPX_NEW, [](std::string_view r) { return r == "SY" || r == "SYS"; }))
    return {Role::Sy, Enc::Oms};
  if (tx_px(f, TXPX, [](std::string_view r) { return r == "SYB"; }) || any_of_sv(f, {"MSBM", "BBOLD"}) ||
      starts_with(f, "DSROM") || starts_with(f, "BBM") || starts_with(f, "PAZOMATHBLACKBOARDBOLD"))
    return {Role::Bb, Enc::None};
  if (any_of_sv(f, {"CMEX", "LMMATHEXTENSION", "EUEX", "YHCMEX"}) || contains_sub(f, "EXTENSION") ||
      tx_px(f, TXPX_NEW, [](std::string_view r) { return r == "EX" || r == "EXA" || r == "EXS" || r == "EXAS"; }))
    return {Role::Ex, Enc::Omx};
  if (f == "MSAM") return {Role::Sy, Enc::Msam};
  if (any_of_sv(f, {"EUFM", "EUFB"})) return {Role::Frak, Enc::None};
  if (starts_with(f, "RSFS") || starts_with(f, "STIXMATHSCRIPT")) return {Role::Scr, Enc::None};
  if (any_of_sv(f, {"EUSM", "EUSB"})) return {Role::Cal, Enc::None};
  auto ends_with = [&](std::string_view s) { return f.size() >= s.size() && f.substr(f.size() - s.size()) == s; };
  const bool sy =
      tx_px(f, TXPX, [](std::string_view r) { return r == "SYA" || r == "SYC"; }) ||
      any_of_sv(f, {"TEX-MATH", "TEX-MATHA", "TEX-MATHB", "TEX-MATHC", "TEX-MATHX", "LASY", "WASY", "OAMATHSYMBOLS",
                    "STIXMATH-REGULAR"}) ||
      starts_with(f, "MNSYMBOL") || starts_with(f, "STANDARDSYML") || starts_with(f, "ESINT") ||
      starts_with(f, "STMARY") || ends_with("SYMBOL") || ends_with("SYMBOL-") ||
      (starts_with(f, "MATHDESIGN") && contains_sub(f.substr(10), "SYMBOL"));
  if (sy) return {Role::Sy, Enc::None};
  return {Role::None, Enc::None};
}

// Font family: subset prefix, a `-Regular` suffix and size digits stripped,
// upper case ("ABCDEF+CMMI10" -> "CMMI").
std::string family(std::string_view font_name) {
  const size_t plus = font_name.rfind('+');
  const std::string_view name = plus == std::string_view::npos ? font_name : font_name.substr(plus + 1);
  std::string up(name);
  for (auto &c : up)
    if (c >= 'a' && c <= 'z') c = (char)(c - 32);
  constexpr std::string_view kReg = "-REGULAR";
  if (up.size() >= kReg.size() && std::string_view(up).substr(up.size() - kReg.size()) == kReg)
    up.resize(up.size() - kReg.size());
  while (!up.empty() && up.back() >= '0' && up.back() <= '9') up.pop_back();
  return up;
}

bool text_fam(std::string_view f) { return f == "CMR" || f == "CMBX" || f == "CMSL" || f == "CMTI"; }

constexpr std::string_view OPERATORS[35] = {
    "sin",  "cos",  "tan",  "log",    "ln",     "exp",    "max",    "min",    "arg", "lim", "det", "tr",
    "sup",  "inf",  "ker",  "dim",    "deg",    "mod",    "sgn",    "sinh",   "cosh", "tanh", "arcsin",
    "arccos", "arctan", "liminf", "limsup", "Pr", "gcd", "lcm", "Re", "Im", "diag", "rank", "span"};

const std::string_view *greek(std::string_view t) {
  static const std::unordered_map<std::string_view, std::string_view> m = {
      {"α", "alpha"},   {"β", "beta"},     {"γ", "gamma"},   {"δ", "delta"},      {"ϵ", "epsilon"},
      {"ε", "varepsilon"}, {"ǫ", "epsilon"}, {"ζ", "zeta"},  {"η", "eta"},        {"θ", "theta"},
      {"ϑ", "vartheta"}, {"ι", "iota"},    {"κ", "kappa"},   {"λ", "lambda"},     {"µ", "mu"},
      {"μ", "mu"},      {"ν", "nu"},       {"ξ", "xi"},      {"π", "pi"},         {"ϖ", "varpi"},
      {"ρ", "rho"},     {"ϱ", "varrho"},   {"σ", "sigma"},   {"ς", "varsigma"},   {"τ", "tau"},
      {"υ", "upsilon"}, {"φ", "phi"},      {"ϕ", "phi"},     {"χ", "chi"},        {"ψ", "psi"},
      {"ω", "omega"},   {"Γ", "Gamma"},    {"∆", "Delta"},   {"Δ", "Delta"},      {"Θ", "Theta"},
      {"Λ", "Lambda"},  {"Ξ", "Xi"},       {"Π", "Pi"},      {"Σ", "Sigma"},      {"Υ", "Upsilon"},
      {"Φ", "Phi"},     {"Ψ", "Psi"},      {"Ω", "Omega"},   {"Ω", "Omega"}};
  const auto it = m.find(t);
  return it == m.end() ? nullptr : &it->second;
}

const std::string_view *symbol(std::string_view t) {
  static const std::unordered_map<std::string_view, std::string_view> m = {
      {"∈", R"(\in)"}, {"∉", R"(\notin)"}, {"≤", R"(\leq)"}, {"⩽", R"(\leq)"}, {"≥", R"(\geq)"}, {"⩾", R"(\geq)"},
      {"≠", R"(\neq)"}, {"≈", R"(\approx)"}, {"∼", R"(\sim)"}, {"~", R"(\sim)"}, {"≃", R"(\simeq)"},
      {"≡", R"(\equiv)"}, {"∝", R"(\propto)"}, {"≪", R"(\ll)"}, {"≫", R"(\gg)"}, {"⊂", R"(\subset)"},
      {"⊆", R"(\subseteq)"}, {"⊃", R"(\supset)"}, {"⊇", R"(\supseteq)"}, {"∪", R"(\cup)"}, {"∩", R"(\cap)"},
      {"∞", R"(\infty)"}, {"∂", R"(\partial)"}, {"∇", R"(\nabla)"}, {"±", R"(\pm)"}, {"∓", R"(\mp)"},
      {"×", R"(\times)"}, {"·", R"(\cdot)"}, {"⋅", R"(\cdot)"}, {"÷", R"(\div)"}, {"∗", R"(\ast)"},
      {"˚", R"(\ast)"}, {"◦", R"(\circ)"}, {"∘", R"(\circ)"}, {"→", R"(\to)"}, {"←", R"(\leftarrow)"},
      {"↔", R"(\leftrightarrow)"}, {"⇒", R"(\Rightarrow)"}, {"⇐", R"(\Leftarrow)"}, {"⇔", R"(\Leftrightarrow)"},
      {"↦", R"(\mapsto)"}, {"∀", R"(\forall)"}, {"∃", R"(\exists)"}, {"¬", R"(\neg)"}, {"∧", R"(\wedge)"},
      {"∨", R"(\vee)"}, {"∅", R"(\emptyset)"}, {"∑", R"(\sum)"}, {"∏", R"(\prod)"}, {"∫", R"(\int)"},
      {"∮", R"(\oint)"}, {"⊗", R"(\otimes)"}, {"⊕", R"(\oplus)"}, {"∥", R"(\|)"}, {"‖", R"(\|)"},
      {"⟨", R"(\langle)"}, {"⟩", R"(\rangle)"}, {"⌊", R"(\lfloor)"}, {"⌋", R"(\rfloor)"}, {"⌈", R"(\lceil)"},
      {"⌉", R"(\rceil)"}, {"…", R"(\ldots)"}, {"⋯", R"(\cdots)"}, {"−", "-"}, {"′", "'"}, {"{", R"(\{)"},
      {"}", R"(\})"}, {"\\", R"(\setminus)"}, {"∖", R"(\setminus)"}, {"†", R"(\dagger)"}, {"⊤", R"(\top)"},
      {"⊥", R"(\perp)"}, {"ℓ", R"(\ell)"}, {"ı", R"(\imath)"}, {"ȷ", R"(\jmath)"}, {"≺", R"(\prec)"},
      {"≻", R"(\succ)"}, {"⪯", R"(\preceq)"}, {"≼", R"(\preceq)"}, {"♯", R"(\#)"}, {"#", R"(\#)"},
      {"%", R"(\%)"}, {"&", R"(\&)"}, {"_", R"(\_)"}, {"^", R"(\hat{})"}, {"⊢", R"(\vdash)"}, {"|", "|"},
      {"√", R"(\surd)"}, {"ℝ", R"(\mathbb{R})"}, {"ℕ", R"(\mathbb{N})"}, {"ℤ", R"(\mathbb{Z})"},
      {"ℚ", R"(\mathbb{Q})"}, {"ℂ", R"(\mathbb{C})"}, {"⊊", R"(\subsetneq)"}, {"≜", R"(\triangleq)"},
      {"≔", ":="}, {"⟶", R"(\longrightarrow)"}, {"⟹", R"(\Longrightarrow)"}, {"↪", R"(\hookrightarrow)"},
      {"⇀", R"(\rightharpoonup)"}, {"̸", "NOT"}, {"", "("}, {"", "["}, {"", R"(\{)"},
      {"", ")"}, {"", "]"}, {"", R"(\})"}, {"", ""}, {"", ""}, {"", ""},
      {"", ""}, {"", ""}, {"", ""}, {"", ""}, {"", ""}, {"", ""},
      {"", ""}, {"", ""}, {"", ""}, {"", ""}, {"", ""}, {"", ""},
      {"⋆", R"(\star)"}, {"⋄", R"(\diamond)"}, {"□", R"(\square)"}, {"△", R"(\triangle)"}};
  const auto it = m.find(t);
  return it == m.end() ? nullptr : &it->second;
}

constexpr std::string_view TEXT_MATH_OK = "0123456789()[]{}=+-−,;:!/|.'<>*′";

bool is_ascii_upper(std::string_view t) {
  if (t.empty()) return false;
  bool upper = false;
  for (unsigned char c : t) {
    if (c >= 0x80) return false;
    if (c >= 'a' && c <= 'z') return false;
    upper |= c >= 'A' && c <= 'Z';
  }
  return upper;
}

// ── glyphs ────────────────────────────────────────────────────────────────────

// Turn collected chars into math-pass glyphs.
std::vector<G> glyphs(const std::vector<RawChar> &chars, float page_h) {
  std::vector<G> out;
  out.reserve(chars.size());
  struct Fam { std::string fam; Role role; Enc enc; };
  std::unordered_map<const std::string *, Fam> fam_cache;
  for (const RawChar &c : chars) {
    auto it = fam_cache.find(c.font_name);
    if (it == fam_cache.end()) {
      std::string fam = family(*c.font_name);
      const auto [r, e] = font_info(fam);
      it = fam_cache.emplace(c.font_name, Fam{std::move(fam), r, e}).first;
    }
    const Fam &fi = it->second;
    // Note: pdfium flags glyphs it could not map; only those are read through
    // the TeX encoding (a ToUnicode'd "{" must stay "{").
    std::string t;
    if (!c.is_code) {
      if (whitespace(c.value)) continue;
      append_utf8(t, c.value);
    } else {
      const uint32_t code = c.value;
      const bool valid = code <= 0x10FFFF && !(code >= 0xD800 && code <= 0xDFFF);
      // Note: letters and the ASCII a cmmi font draws as itself stay plain text.
      if (fi.enc == Enc::Oml && valid && code < 0x80 &&
          (((code | 0x20) >= 'a' && (code | 0x20) <= 'z') || std::string_view(".,</>").find((char)code) != std::string_view::npos)) {
        t = std::string(1, (char)code);
      } else if (auto dec = fi.enc != Enc::None ? decode(fi.enc, code) : std::nullopt) {
        t = std::string(1, PRE);
        t += *dec;
      } else if (valid && code >= 0x21 && code <= 0x7E) {
        t = std::string(1, (char)code);
      } else {
        continue;
      }
    }
    if (t.empty()) continue;
    Role role = fi.role;
    if (role == Role::None && text_fam(fi.fam) && greek(t)) role = Role::Mi;  // upright capital Greek is set in the roman font
    const float base = page_h - c.origin_y;
    // Note: the ink box, not an em box. TeX sets \hat{x}'s accent on x's own
    // baseline, so only ink tells the accent sits above; a radical's ink top is
    // where its rule starts.
    float top, bottom;
    if (c.has_ink) {
      top = c.ink_top;
      bottom = c.ink_bottom;
    } else {
      const float bottom_up = base + c.descent;
      top = page_h - (bottom_up + c.size);
      bottom = page_h - bottom_up;
    }
    G g;
    g.t = std::move(t);
    g.role = role;
    g.enc = fi.enc;
    g.fam = fi.fam;
    g.size = c.size;
    g.base = base;
    g.x0 = c.x0;
    g.x1 = c.x1;
    g.top = top;
    g.bottom = bottom;
    g.loose_top = c.loose_top;
    g.loose_bottom = c.loose_bottom;
    g.src = {c.index};
    out.push_back(std::move(g));
  }
  return out;
}

void sort_by_x0(const std::vector<G> &all, Idx &v) {
  std::stable_sort(v.begin(), v.end(), [&](int a, int b) { return total_less(all[a].x0, all[b].x0); });
}

struct Base {
  float y, size;
  Idx gs;
  bool has_ext = false;
  float e0 = 0, e1 = 0;
};

// Group glyphs into lines: full-size glyphs set the line baselines, smaller ones
// (scripts) attach to the nearest line whose band they fall in.
std::vector<Idx> lines(std::vector<G> &all, const Idx &live) {
  if (live.empty()) return {};
  std::vector<float> sizes;
  for (int i : live) sizes.push_back(all[i].size);
  const float body = median(std::move(sizes));
  Idx big;
  for (int i : live)
    if (all[i].size >= 0.85f * body) big.push_back(i);
  std::stable_sort(big.begin(), big.end(), [&](int a, int b) { return total_less(all[b].base, all[a].base); });
  std::vector<Base> bases;
  for (int g : big) {
    const float gb = all[g].base, gsz = all[g].size;
    Base *hit = nullptr;
    for (auto &b : bases)
      if (std::abs(b.y - gb) < 0.25f * gsz) { hit = &b; break; }
    if (hit) hit->gs.push_back(g);
    else bases.push_back(Base{gb, gsz, {g}});
  }
  // Scripts attach to the nearest baseline among the rows whose x-extent holds
  // them (one baseline can hold a row of each column on a two-column page).
  for (auto &b : bases) {
    float x0 = std::numeric_limits<float>::max(), x1 = std::numeric_limits<float>::lowest();
    for (int i : b.gs) { x0 = std::min(x0, all[i].x0); x1 = std::max(x1, all[i].x1); }
    b.has_ext = true;
    b.e0 = x0 - 1.0f * b.size;
    b.e1 = x1 + 1.5f * b.size;
  }
  for (int g : live) {
    if (all[g].role != Role::Ex) continue;
    // Note: a cmex glyph hangs from its origin; its line is the nearest baseline below that.
    const float xc = all[g].cx();
    const float gb = all[g].base, gs = all[g].size;
    int row = -1;
    for (size_t k = 0; k < bases.size(); ++k) {
      const Base &b = bases[k];
      if (!b.has_ext) continue;
      const bool ok = b.e0 <= xc && xc <= b.e1 && gb - 3.0f * gs <= b.y && b.y <= gb + 0.3f * gs &&
                      !b.gs.empty() && all[b.gs.front()].role != Role::Ex;
      if (ok && (row < 0 || b.y > bases[row].y)) row = (int)k;
    }
    if (row >= 0) {
      for (auto &b : bases) {
        auto p = std::find(b.gs.begin(), b.gs.end(), g);
        if (p != b.gs.end()) b.gs.erase(p);
      }
      all[g].base = bases[row].y;
      all[g].size = bases[row].size;
      bases[row].gs.push_back(g);
    }
  }
  std::erase_if(bases, [](const Base &b) { return b.gs.empty(); });
  for (int g : live) {
    if (all[g].size >= 0.85f * body || all[g].role == Role::Ex) continue;
    const float xc = all[g].cx();
    const float gb = all[g].base;
    int best = -1;
    for (size_t k = 0; k < bases.size(); ++k) {
      const Base &b = bases[k];
      if (!b.has_ext) continue;
      const float d = gb - b.y;
      if (!(-0.6f * b.size < d && d < 0.8f * b.size && b.e0 <= xc && xc <= b.e1)) continue;
      if (best < 0 || std::abs(gb - b.y) < std::abs(gb - bases[best].y)) best = (int)k;
    }
    if (best >= 0) bases[best].gs.push_back(g);
    else bases.push_back(Base{gb, all[g].size, {g}});
  }
  // Note: then split each row at wide gaps nothing fills: the gutter between two columns.
  std::vector<std::pair<float, Idx>> split;
  for (const auto &b : bases) {
    Idx row = b.gs;
    sort_by_x0(all, row);
    Idx cur{row[0]};
    float right = all[row[0]].x1;
    for (size_t k = 1; k < row.size(); ++k) {
      const int g = row[k];
      if (all[g].x0 - right > 1.5f * b.size) split.emplace_back(b.y, std::exchange(cur, {}));
      cur.push_back(g);
      right = std::max(right, all[g].x1);
    }
    split.emplace_back(b.y, std::move(cur));
  }
  std::stable_sort(split.begin(), split.end(), [&](const auto &a, const auto &b) {
    if (total_key(a.first) != total_key(b.first)) return total_less(b.first, a.first);
    return total_less(all[a.second[0]].x0, all[b.second[0]].x0);
  });
  std::vector<Idx> out;
  out.reserve(split.size());
  for (auto &[y, gs] : split) {
    sort_by_x0(all, gs);
    out.push_back(std::move(gs));
  }
  return out;
}

const char *accent_name(std::string_view t) {
  if (t == "ˆ" || t == "^") return "hat";
  if (t == "˜" || t == "~") return "tilde";
  if (t == "¯") return "bar";
  if (t == "˙") return "dot";
  if (t == "¨") return "ddot";
  if (t == "´") return "acute";
  if (t == "`") return "grave";
  if (t == "ˇ") return "check";
  if (t == "˘") return "breve";
  if (t == "⃗") return "vec";
  return nullptr;
}

std::optional<std::string> accent_of(const G &g) {
  using namespace std::literals;
  constexpr auto kAcc = "\0ACC:"sv;
  if (starts_with(g.t, kAcc)) return std::string(g.t.substr(kAcc.size()));
  if (g.role == Role::None || g.role == Role::Mi) {
    if (const char *a = accent_name(g.t); a && (g.role == Role::None || contains_sub("˜¯˙¨ˆ⃗", g.t)))
      return std::string(a);
  }
  return std::nullopt;
}

std::string cmd_space(std::string_view s) {
  if (!s.empty() && s[0] == '\\') {
    const auto u = utf8_to_u32(s);
    if (!u.empty() && alphabetic(u.back())) return std::string(s) + " ";
  }
  return std::string(s);
}

std::string tex_char_inner(const G &g) {
  const std::string_view t = g.t;
  if (!t.empty() && t[0] == PRE) return cmd_space(t.substr(1));
  if (text_fam(g.fam))
    if (const auto *n = greek(t)) return "\\" + std::string(*n) + " ";
  if (g.role == Role::Bb && is_alpha(t)) return R"(\mathbb{)" + to_upper(t) + "}";
  if (g.role == Role::Frak && is_alpha(t)) return R"(\mathfrak{)" + std::string(t) + "}";
  if (g.role == Role::Scr && is_alpha(t)) return R"(\mathscr{)" + std::string(t) + "}";
  if (((g.role == Role::Sy && g.enc == Enc::Oms) || g.role == Role::Cal) && is_ascii_upper(t))
    return R"(\mathcal{)" + std::string(t) + "}";
  if (const auto *n = greek(t)) return "\\" + std::string(*n) + " ";
  if (const auto *s = symbol(t)) return cmd_space(*s);
  return std::string(t);
}

std::string tex_char(const G &g) { return g.pre + tex_char_inner(g) + g.post; }

bool in_text_math_ok(std::string_view t) {
  const auto c = single(t);
  return c && contains_char(TEXT_MATH_OK, *c);
}

// Glyphs a footnote marker is made of: digits and the reference symbols; commas only between them.
bool is_marker(std::string_view t, bool first) {
  const auto c = single(t);
  return c && ((*c >= '0' && *c <= '9') || contains_char("∗*†‡§¶⋆", *c) || (!first && *c == ','));
}

// Drops a text-font bracket at either end of a span that has no partner inside
// it (the `)` of "(q and q̇)" belongs to the sentence).
void trim_unmatched_brackets(const std::vector<G> &all, Idx &sp) {
  std::vector<bool> unmatched(sp.size(), false);
  // Note: any opener closes with any closer: half-open intervals ([0, ∞)).
  std::vector<size_t> stack;
  for (size_t k = 0; k < sp.size(); ++k) {
    const G &x = all[sp[k]];
    if (x.role != Role::None) continue;
    if (x.t == "(" || x.t == "[") stack.push_back(k);
    else if (x.t == ")" || x.t == "]") {
      if (stack.empty()) unmatched[k] = true;
      else stack.pop_back();
    }
  }
  for (size_t k : stack) unmatched[k] = true;
  size_t lo = 0;
  while (lo < unmatched.size() && unmatched[lo]) ++lo;
  size_t tail = 0;
  for (size_t k = unmatched.size(); k > lo && unmatched[k - 1]; --k) ++tail;
  const size_t hi = sp.size() - tail;
  sp.resize(hi);
  sp.erase(sp.begin(), sp.begin() + (long)lo);
}

// Runs of math glyphs in a line, extended over text-font digits / punctuation /
// operator names that sit between or next to math glyphs.
std::vector<Idx> spans_in_line(const std::vector<G> &all, const Idx &ln) {
  const size_t n = ln.size();
  if (n == 0) return {};
  auto g = [&](size_t k) -> const G & { return all[ln[k]]; };
  const float body = max_size(all, ln);
  std::vector<bool> math(n);
  bool any_role = false;
  for (size_t k = 0; k < n; ++k) {
    math[k] = g(k).role != Role::None || g(k).size < 0.85f * body;
    any_role |= g(k).role != Role::None;
  }
  // Note: a small text-font glyph is a script only when the line also has math-font glyphs.
  if (!any_role) return {};
  // Note: a raised small run set tight after a text-font word or sentence
  // punctuation is a footnote marker (Kim∗, Lee∗†, b.³), not a superscript.
  // Operator names keep their scripts (sin²).
  size_t k = 1;
  while (k < n) {
    const G &prev = g(k - 1);
    const bool after_text = prev.role == Role::None && prev.size >= 0.85f * body &&
                            (is_alpha(prev.t) || contains_sub(".,;:", prev.t)) && g(k).x0 - prev.x1 < 0.3f * prev.size;
    auto raised = [&](size_t m) { return g(m).size < 0.85f * body && g(m).base > prev.base + 0.2f * prev.size; };
    if (!(after_text && raised(k) && is_marker(g(k).t, true))) {
      ++k;
      continue;
    }
    size_t w = k - 1;
    while (w > 0 && g(w - 1).role == Role::None && is_alpha(g(w - 1).t)) --w;
    std::string word;
    for (size_t m = w; m < k; ++m) word += g(m).t;
    size_t j = k;
    while (j < n && raised(j) && is_marker(g(j).t, false)) ++j;
    // Note: a word whose script is set tight against a bracket or math is a function name (Aut⁰(X)).
    const bool tight_after = j < n && g(j).x0 - g(j - 1).x1 < 0.15f * g(j).size && (math[j] || g(j).t == "(");
    if (!is_operator(word) && !tight_after)
      for (size_t m = k; m < j; ++m) math[m] = false;
    k = j;
  }
  // Operator names in text font: a run of letters matching OPERATORS next to math.
  size_t i = 0;
  while (i < n) {
    if (g(i).role == Role::None && is_alpha(g(i).t)) {
      size_t j = i;
      while (j < n && g(j).role == Role::None && is_alpha(g(j).t) && (j == i || g(j).x0 - g(j - 1).x1 < 0.15f * g(j).size))
        ++j;
      std::string word;
      for (size_t m = i; m < j; ++m) word += g(m).t;
      const bool near = (i > 0 && math[i - 1]) || (j < n && math[j]);
      const float em = g(i).size;
      const bool tight_l = i > 0 && math[i - 1] && g(i).x0 - g(i - 1).x1 < 0.2f * em;
      const bool tight_r = j < n && g(j).x0 - g(j - 1).x1 < 0.12f * em && (math[j] || g(j).t == "(");
      // Note: a bold letter against math is a vector (\mathbf{x}); a word set
      // tight against a bracket or math is a function name (Hom(A, B)).
      const bool bold_letter = j - i == 1 && (g(i).fam == "CMBX" || g(i).fam == "CMB") && (tight_l || tight_r);
      if ((is_operator(word) && near) || bold_letter || (j - i >= 2 && tight_r && near))
        for (size_t m = i; m < j; ++m) math[m] = true;
      i = j;
    } else {
      ++i;
    }
  }
  for (size_t k2 = 0; k2 < n; ++k2) {
    if (!math[k2] && accent_of(g(k2))) {
      const float cx = g(k2).cx();
      bool any = false;
      for (size_t m = k2 >= 3 ? k2 - 3 : 0; m < std::min(k2 + 4, n); ++m)
        if (math[m] && g(m).x0 - 1.0f <= cx && cx <= g(m).x1 + 1.0f) { any = true; break; }
      math[k2] = any;
    }
  }
  // Grow over text-font digits / punctuation adjacent to math, both directions, until stable.
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t k2 = 0; k2 < n; ++k2) {
      if (math[k2] || !in_text_math_ok(g(k2).t)) continue;
      const float sz = g(k2).size;
      const bool left = k2 > 0 && math[k2 - 1] && g(k2).x0 - g(k2 - 1).x1 < 0.6f * sz;
      const bool right = k2 + 1 < n && math[k2 + 1] && g(k2 + 1).x0 - g(k2).x1 < 0.6f * sz;
      if (left || right) {
        // Note: a sentence's full stop or comma after math stays text.
        if (contains_sub(".,;:", g(k2).t) && !right) continue;
        math[k2] = true;
        changed = true;
      }
    }
  }
  std::vector<Idx> out;
  Idx cur;
  for (size_t k2 = 0; k2 < n; ++k2) {
    if (math[k2]) {
      // Note: TeX's widest math space is ~0.3em; a wider gap before a text-font
      // number is a table column (3.6→ | 18.7→), not math.
      if (!cur.empty()) {
        const float d = g(k2).x0 - all[cur.back()].x1;
        bool numeral = g(k2).role == Role::None;
        for (unsigned char c : g(k2).t) numeral &= c >= '0' && c <= '9';
        if (d > 1.5f * body || (numeral && d > 0.6f * body)) out.push_back(std::exchange(cur, {}));
      }
      cur.push_back(ln[k2]);
    } else if (!cur.empty()) {
      out.push_back(std::exchange(cur, {}));
    }
  }
  if (!cur.empty()) out.push_back(std::move(cur));
  for (auto &sp : out) trim_unmatched_brackets(all, sp);
  std::erase_if(out, [&](const Idx &s) {
    const bool any = std::any_of(s.begin(), s.end(), [&](int x) { return all[x].role != Role::None; });
    return !(any || s.size() > 1);
  });
  return out;
}

// Note: a small glyph back on the baseline is a script's script (the l of \|g\|_{C^l}).
bool on_base(const G &g, float base, float size) {
  return g.size >= 0.92f * size || (std::abs(g.base - base) <= 0.12f * size && g.size >= 0.8f * size);
}

std::string tidy(std::string_view s0) {
  std::string s = replace_all(std::string(s0), "...", R"(\ldots )");
  s = replace_all(std::move(s), R"(\cdot \cdot \cdot )", R"(\cdots )");
  s = replace_all(std::move(s), "NOT=", R"(\neq )");
  s = replace_all(std::move(s), "NOT", R"(\not )");
  s = replace_all(std::move(s), R"(MAPSTO\to )", R"(\mapsto )");
  s = replace_all(std::move(s), "MAPSTO", "");
  for (std::string_view tok : {"HAT", "TILDE", "SQRT"}) s = replace_all(std::move(s), tok, "");
  return s;
}

// The item box for a span on baseline `base`: the font box of the real glyphs on
// that baseline (the span's own, else its line's), so the item sits in its line
// like text does.
//
// Note: neither scripts' ink nor a fraction's stacked parts may set it: an item
// that rises above its line drops out of the line's band in layout, and the text
// after it then reads as a new column start (a two-column page turned "tabular").
std::pair<float, float> line_box(const std::vector<G> &all, const Idx &ln, const Idx &kept, float base, float size,
                                 float page_h) {
  auto real = [&](int x) {
    const G &g = all[x];
    return g.fam != STAND_IN_FAM && std::abs(g.base - base) <= 0.1f * size && g.size >= 0.8f * size;
  };
  float top = std::numeric_limits<float>::max(), bottom = std::numeric_limits<float>::lowest();
  bool any = false;
  for (int x : kept)
    if (real(x)) { any = true; top = std::min(top, all[x].loose_top); bottom = std::max(bottom, all[x].loose_bottom); }
  if (!any)
    for (int x : ln)
      if (real(x)) { any = true; top = std::min(top, all[x].loose_top); bottom = std::max(bottom, all[x].loose_bottom); }
  if (!any) {
    // Typical font box: 0.78 em above the baseline, 0.22 em below.
    const float y = page_h - base;
    return {y - 0.78f * size, y + 0.22f * size};
  }
  return {top, bottom};
}

}  // namespace

float median(std::vector<float> v) {
  std::sort(v.begin(), v.end(), total_less);
  const size_t n = v.size();
  if (n == 0) return 0.0f;
  if (n % 2 == 1) return v[n / 2];
  return (v[n / 2 - 1] + v[n / 2]) / 2.0f;
}

float max_size(const std::vector<G> &gs, const Idx &idx) {
  float m = std::numeric_limits<float>::lowest();
  for (int i : idx) m = std::max(m, gs[i].size);
  return m;
}

bool is_alpha(std::string_view t) {
  if (t.empty()) return false;
  for (char32_t c : utf8_to_u32(t))
    if (!alphabetic(c)) return false;
  return true;
}

bool is_operator(std::string_view w) { return std::find(std::begin(OPERATORS), std::end(OPERATORS), w) != std::end(OPERATORS); }

// Drop accent glyphs from a span, wrapping the glyphs under each in \hat{..} etc.
Idx attach_accents(std::vector<G> &all, const Idx &sp) {
  Idx accs, rest;
  for (int i : sp) (accent_of(all[i]) ? accs : rest).push_back(i);
  if (accs.empty()) return sp;
  for (int a : accs) {
    const float ax0 = all[a].x0, ax1 = all[a].x1, atop = all[a].top, asz = all[a].size;
    const float cx = (ax0 + ax1) / 2.0f;
    const std::string name = accent_of(all[a]).value_or("");
    const bool wide = starts_with(name, "wide");
    Idx under;
    if (wide) {
      const float rest_max = max_size(all, rest);
      for (int gi : rest) {
        const G &gg = all[gi];
        if (gg.top > atop && ax0 - 0.1f * asz <= gg.cx() && gg.cx() <= ax1 + 0.1f * asz && gg.size >= 0.9f * rest_max)
          under.push_back(gi);
      }
    } else {
      for (int gi : rest) {
        const G &gg = all[gi];
        if (gg.top > atop + 0.1f * asz && gg.x0 - 0.15f * asz <= cx && cx <= gg.x1 + 0.15f * asz) under.push_back(gi);
      }
    }
    if (under.empty()) continue;
    sort_by_x0(all, under);
    if (!wide) {
      int best = under[0];
      for (size_t k = 1; k < under.size(); ++k)
        if (std::abs(all[under[k]].cx() - cx) < std::abs(all[best].cx() - cx)) best = under[k];
      under = {best};
    }
    const int first = under.front(), last = under.back();
    all[first].pre += "\\" + name + "{";
    all[last].post = "}" + all[last].post;
  }
  return rest;
}

// LaTeX for glyphs on baseline `base` at `size`, with sub/superscripts recursively.
std::string build(const std::vector<G> &all, const Idx &gs, float base, float size) {
  static constexpr std::string_view OPERATORNAME[6] = {"diag", "rank", "span", "sgn", "Re", "Im"};
  std::string out;
  const size_t n = gs.size();
  size_t i = 0;
  while (i < n) {
    const G &g = all[gs[i]];
    if (on_base(g, base, size)) {
      if (g.role == Role::None && is_alpha(g.t)) {
        size_t j = i;
        while (j < n && all[gs[j]].role == Role::None && is_alpha(all[gs[j]].t) && on_base(all[gs[j]], base, size)) ++j;
        std::string word;
        for (size_t m = i; m < j; ++m) word += all[gs[m]].t;
        if (is_operator(word)) {
          if (std::find(std::begin(OPERATORNAME), std::end(OPERATORNAME), word) != std::end(OPERATORNAME))
            out += R"(\operatorname{)" + word + "}";
          else
            out += "\\" + word + " ";
          i = j;
          continue;
        }
      }
      out += tex_char(g);
      ++i;
      continue;
    }
    // A script run: consecutive glyphs off the baseline.
    size_t j = i;
    while (j < n && !on_base(all[gs[j]], base, size)) ++j;
    const Idx run(gs.begin() + (long)i, gs.begin() + (long)j);
    const float top = max_size(all, run);
    // Note: first-level script glyphs pick sub or sup by their baseline; smaller
    // (second-level) glyphs go with the first-level glyph before them (E_{g^{-1}}:
    // the -1 rides on the g).
    Idx sub, sup;
    int last_sup = -1;  // -1 none, 0 sub, 1 sup
    bool has_sub_base = false, has_sup_base = false;
    float sub_base = 0, sup_base = 0;  // baselines of the latest first-level sub and sup glyphs
    for (int x : run) {
      if (all[x].size >= 0.9f * top || last_sup < 0) {
        const bool up = all[x].base > base;
        last_sup = up;
        if (up) { sup_base = all[x].base; has_sup_base = true; }
        else { sub_base = all[x].base; has_sub_base = true; }
      } else if (has_sub_base && has_sup_base) {
        // Note: with both scripts open, a smaller glyph rides the one whose
        // baseline is nearer (\Phi_{B_s}^{ren}: the s is B's).
        last_sup = std::abs(all[x].base - sup_base) < std::abs(all[x].base - sub_base);
      }
      (last_sup == 1 ? sup : sub).push_back(x);
    }
    for (int pass = 0; pass < 2; ++pass) {
      const Idx &grp = pass == 0 ? sub : sup;
      const bool is_sup = pass == 1;
      if (grp.empty()) continue;
      Idx lvl1;
      for (int x : grp)
        if (all[x].size >= 0.9f * top) lvl1.push_back(x);
      if (lvl1.empty()) lvl1 = grp;
      std::vector<float> bs;
      for (int x : lvl1) bs.push_back(all[x].base);
      const float b = median(std::move(bs));
      std::string body = build(all, grp, b, max_size(all, lvl1));
      if (is_sup) {
        // Note: primes are written f', not f^{'} (a double superscript in KaTeX).
        size_t k2 = 0;
        while (k2 < body.size() && body[k2] == '\'') ++k2;
        if (k2 > 0) {
          out.append(k2, '\'');
          body = std::string(trim(std::string_view(body).substr(k2)));
        }
      }
      if (!body.empty()) {
        out += is_sup ? '^' : '_';
        out += '{';
        out += body;
        out += '}';
      }
    }
    i = j;
  }
  return tidy(trim(out));
}

bool is_tex_math_font(std::string_view font_name) {
  const auto [role, enc] = font_info(family(font_name));
  return enc != Enc::None || role == Role::Bb || role == Role::Frak || role == Role::Cal || role == Role::Scr;
}

std::optional<std::vector<MathSpan>> page_math(const std::vector<RawChar> &chars, const std::vector<Rule> &rules,
                                               float page_h) {
  std::vector<G> all = glyphs(chars, page_h);
  // Note: only pages set by TeX (a math font in a TeX encoding, or an AMS /
  // Euler / rsfs font) get math: Word's Symbol font and table rules elsewhere
  // would otherwise read as math.
  const bool tex_page = std::any_of(all.begin(), all.end(), [](const G &g) {
    return g.enc != Enc::None || g.role == Role::Bb || g.role == Role::Frak || g.role == Role::Cal || g.role == Role::Scr;
  });
  if (!tex_page) return std::nullopt;
  Idx live(all.size());
  for (size_t k = 0; k < all.size(); ++k) live[k] = (int)k;
  live = fold(all, std::move(live), rules, page_h);
  std::vector<MathSpan> res;
  for (const Idx &ln : lines(all, live)) {
    for (const Idx &sp : spans_in_line(all, ln)) {
      const Idx kept = attach_accents(all, sp);
      if (kept.empty()) continue;
      const float big = max_size(all, kept);
      std::vector<float> bs;
      for (int x : kept)
        if (all[x].size >= 0.92f * big) bs.push_back(all[x].base);
      const float base = median(std::move(bs));
      std::string tex = build(all, kept, base, big);
      if (tex.empty()) continue;
      const auto [top, bottom] = line_box(all, ln, kept, base, big, page_h);
      MathSpan s;
      s.tex = std::move(tex);
      s.x0 = std::numeric_limits<float>::max();
      s.x1 = std::numeric_limits<float>::lowest();
      for (int x : sp) {
        s.x0 = std::min(s.x0, all[x].x0);
        s.x1 = std::max(s.x1, all[x].x1);
        s.chars.insert(s.chars.end(), all[x].src.begin(), all[x].src.end());
      }
      std::sort(s.chars.begin(), s.chars.end());
      s.chars.erase(std::unique(s.chars.begin(), s.chars.end()), s.chars.end());
      s.top = top;
      s.bottom = bottom;
      s.size = big;
      res.push_back(std::move(s));
    }
  }
  return res;
}

}  // namespace hp::tex::detail

namespace hp::tex {

// Whether the span carries structure plain text can't show: scripts, fractions,
// roots, accents, or a math alphabet. Note: a lone ± or μ (or plain Greek) stays
// text, so table cells like "0.11 ± 0.02" keep their form.
bool has_structure(const std::string &tex) {
  static constexpr std::string_view MARKS[19] = {
      R"(\frac)",  R"(\sqrt)", R"(\overline)", R"(\mathbb)", R"(\mathcal)", R"(\mathfrak)", R"(\mathscr)",
      R"(\hat)",   R"(\tilde)", R"(\bar)",     R"(\dot)",    R"(\ddot)",    R"(\acute)",    R"(\grave)",
      R"(\check)", R"(\breve)", R"(\vec)",     R"(\widehat)", R"(\widetilde)"};
  if (tex.find_first_of("_^") != std::string::npos) return true;
  for (auto m : MARKS)
    if (tex.find(m) != std::string::npos) return true;
  return false;
}

}  // namespace hp::tex
