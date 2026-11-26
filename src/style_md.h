#pragma once
// Inline Markdown for a line's styled runs (bold, italic, code, strike, underline, sup / sub),
// after liteparse markdown_layout/inline.rs render_line_inline + apply_style:
//   - a line whose runs all share one style gets one wrap around the whole line;
//   - otherwise consecutive runs of one style are grouped, and a group is wrapped on its own,
//     with a space between groups only where the page has one;
//   - mono wins over bold / italic (`code`), bold + italic is ***x***, a script is <sup> / <sub>
//     without emphasis, underline is <u>, strikethrough (~~) wraps outermost;
//   - TeX math runs ($...$, kStyleMath) are never wrapped.
// Only the Markdown view uses it: model inputs (OrderNet, LayoutNet) and the JSON keep plain text.
#include <string>

#include "assemble.h"

namespace hp {

std::string styled_markdown(const PageLine &line);

// Joins emphasis that a line break split: "~~a~~ ~~b~~" -> "~~a b~~" (also **, ***, <u>), as
// liteparse's append_inline_continuation does when it joins a paragraph's lines.
void merge_adjacent_emphasis(std::string &md);

}  // namespace hp
