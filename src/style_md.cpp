#include "style_md.h"

#include <algorithm>
#include <string_view>
#include <utility>

#include "text_layer.h"

namespace hp {
namespace {

bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n'; }

std::string_view trim(std::string_view s) {
  while (!s.empty() && is_ws(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_ws(s.back())) s.remove_suffix(1);
  return s;
}

std::string apply_style(std::string inner, uint8_t s) {
  if (s & kStyleMono) {
    size_t longest = 0, run = 0;
    for (char c : inner) longest = std::max(longest, run = c == '`' ? run + 1 : 0);
    if (longest == 0) {
      inner = "`" + inner + "`";
    } else {
      const std::string fence(longest + 1, '`');
      inner = fence + " " + inner + " " + fence;
    }
  } else if (!(s & (kStyleSup | kStyleSub))) {  // a script gets its tag alone
    const bool b = s & kStyleBold, i = s & kStyleItalic;
    if (b && i) inner = "***" + inner + "***";
    else if (b) inner = "**" + inner + "**";
    else if (i) inner = "*" + inner + "*";
  }
  if (s & kStyleSup) inner = "<sup>" + inner + "</sup>";
  else if (s & kStyleSub) inner = "<sub>" + inner + "</sub>";
  if (s & kStyleUnderline) inner = "<u>" + inner + "</u>";
  if (s & kStyleStrike) inner = "~~" + inner + "~~";
  return inner;
}

struct Piece {
  size_t b, e;    // byte range in the line's text, trimmed
  uint8_t style;  // StyleBit set; 0 = plain (also for math: never wrapped)
};

}  // namespace

std::string styled_markdown(const PageLine &line) {
  const std::string &t = line.text;
  if (line.styles.empty()) return t;
  std::vector<StyleRun> runs = line.styles;
  std::sort(runs.begin(), runs.end(), [](const StyleRun &a, const StyleRun &b) { return a.b < b.b; });
  // The line as pieces: styled runs and the plain text between them, each trimmed, empties dropped.
  std::vector<Piece> pieces;
  auto add = [&](size_t b, size_t e, uint8_t s) {
    b = std::min(b, t.size()), e = std::min(e, t.size());
    while (b < e && is_ws(t[b])) ++b;
    while (e > b && is_ws(t[e - 1])) --e;
    if (b < e) pieces.push_back({b, e, static_cast<uint8_t>((s & kStyleMath) ? 0 : s)});
  };
  size_t pos = 0;
  for (const StyleRun &r : runs) {
    if (r.b < pos) continue;  // overlapping run (should not happen): keep the first
    if (r.b > pos) add(pos, r.b, 0);
    add(r.b, r.e, r.style);
    pos = r.e;
  }
  if (pos < t.size()) add(pos, t.size(), 0);
  if (pieces.empty()) return t;
  const bool uniform = std::all_of(pieces.begin(), pieces.end(), [&](const Piece &p) { return p.style == pieces[0].style; });
  if (uniform) {
    if (!pieces[0].style) return t;
    return apply_style(std::string(trim(t)), pieces[0].style);
  }
  std::string out;
  for (size_t i = 0; i < pieces.size();) {
    size_t j = i + 1;
    while (j < pieces.size() && pieces[j].style == pieces[i].style) ++j;
    const std::string body(t, pieces[i].b, pieces[j - 1].e - pieces[i].b);
    // a space only where the page has one: a style that changes inside a word ("Taha" + raised
    // "1", "**Note**" + ":") abuts its neighbour
    if (!out.empty() && i > 0 && pieces[i].b > pieces[i - 1].e) out += ' ';
    out += pieces[i].style ? apply_style(body, pieces[i].style) : body;
    i = j;
  }
  return out;
}

void merge_adjacent_emphasis(std::string &md) {
  static constexpr std::pair<std::string_view, std::string_view> kJoins[] = {
      {"~~ ~~", " "}, {"</u> <u>", " "}, {"</u><u>", ""}, {"*** ***", " "}, {"** **", " "}};
  for (const auto &[from, to] : kJoins)
    for (size_t p = md.find(from); p != std::string::npos; p = md.find(from, p + to.size()))
      md.replace(p, from.size(), to);
}

}  // namespace hp
