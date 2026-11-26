// Two-dimensional math from rules: fractions, overlines and square roots. Port of
// LiteParse's tex_math/structures.rs.
//
// A rule is a thin horizontal rect, line, or stretched image (some PDFs draw TeX's
// \vrule as a 1x1 image). Rules are taken narrowest first, so an inner fraction is
// folded before the one holding it: its glyphs become one stand-in glyph whose
// text is the finished LaTeX.

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "detail.h"

namespace hp::tex::detail {
namespace {

std::string tex_of(std::vector<G> &all, const Idx &gs) {
  Idx sorted = gs;
  std::stable_sort(sorted.begin(), sorted.end(), [&](int a, int b) { return total_less(all[a].x0, all[b].x0); });
  const Idx kept = attach_accents(all, sorted);
  if (kept.empty()) return {};
  const float big = max_size(all, kept);
  std::vector<float> bs;
  for (int g : kept)
    if (all[g].size >= 0.92f * big) bs.push_back(all[g].base);
  return build(all, kept, median(std::move(bs)), big);
}

G stand_in(const std::vector<G> &all, const std::string &tex, const Idx &parts, float x0, float x1, float base,
           float size) {
  G g;
  g.t = std::string(1, PRE) + tex;
  g.role = Role::Mi;
  g.enc = Enc::None;
  g.fam = std::string(STAND_IN_FAM);
  g.size = size;
  g.base = base;
  g.x0 = x0;
  g.x1 = x1;
  g.top = std::numeric_limits<float>::max();
  g.bottom = std::numeric_limits<float>::lowest();
  g.loose_top = std::numeric_limits<float>::max();
  g.loose_bottom = std::numeric_limits<float>::lowest();
  for (int p : parts) {
    g.top = std::min(g.top, all[p].top);
    g.bottom = std::max(g.bottom, all[p].bottom);
    g.loose_top = std::min(g.loose_top, all[p].loose_top);
    g.loose_bottom = std::max(g.loose_bottom, all[p].loose_bottom);
    g.src.insert(g.src.end(), all[p].src.begin(), all[p].src.end());
  }
  return g;
}

// The glyphs stacked right above (or below) a rule, within its width: the
// baseline nearest the rule and the scripts riding on it. Glyphs much larger than
// the nearest ones are a neighbouring text line, not the fraction (a text-style
// fraction is set at script size).
Idx stack(const std::vector<G> &all, const Idx &live, const Rule &r, float h, bool above) {
  Idx cand;
  float s = std::numeric_limits<float>::lowest(), nb;
  if (above) {
    for (int g : live) {
      const G &x = all[g];
      if (r.x0 - 0.8f <= x.cx() && x.cx() <= r.x1 + 0.8f && (h - x.bottom) >= r.y - 0.6f && x.base - r.y < 1.6f * x.size)
        cand.push_back(g);
    }
    if (cand.empty()) return {};
    int g0 = cand[0];
    for (size_t k = 1; k < cand.size(); ++k)
      if (all[cand[k]].base < all[g0].base) g0 = cand[k];
    for (int g : cand)
      if (all[g].base <= all[g0].base + 0.4f * all[g].size) s = std::max(s, all[g].size);
    nb = std::numeric_limits<float>::max();
    for (int g : cand)
      if (all[g].size >= 0.85f * s) nb = std::min(nb, all[g].base);
  } else {
    for (int g : live) {
      const G &x = all[g];
      if (r.x0 - 0.8f <= x.cx() && x.cx() <= r.x1 + 0.8f && (h - x.top) <= r.y + 0.6f && r.y - (h - x.top) < 0.9f * x.size)
        cand.push_back(g);
    }
    if (cand.empty()) return {};
    int g0 = cand[0];
    for (size_t k = 1; k < cand.size(); ++k)
      if ((h - all[cand[k]].top) > (h - all[g0].top)) g0 = cand[k];
    for (int g : cand)
      if ((h - all[g].top) >= (h - all[g0].top) - 0.4f * all[g].size) s = std::max(s, all[g].size);
    nb = std::numeric_limits<float>::lowest();
    for (int g : cand)
      if (all[g].size >= 0.85f * s) nb = std::max(nb, all[g].base);
  }
  Idx out;
  for (int g : cand)
    if (all[g].size <= 1.15f * s && nb - 0.45f * s <= all[g].base && all[g].base <= nb + 0.6f * s) out.push_back(g);
  return out;
}

// Whether glyphs hold a word of running text (3+ text-font letters that aren't an
// operator name): table cells over and under a rule, not a fraction.
bool wordy(const std::vector<G> &all, const Idx &gs) {
  Idx sorted = gs;
  std::stable_sort(sorted.begin(), sorted.end(), [&](int a, int b) { return total_less(all[a].x0, all[b].x0); });
  std::string run;
  size_t run_len = 0;
  for (size_t k = 0; k <= sorted.size(); ++k) {
    if (k < sorted.size() && all[sorted[k]].role == Role::None && is_alpha(all[sorted[k]].t)) {
      run += all[sorted[k]].t;
      ++run_len;
      continue;
    }
    if (run_len >= 3 && !is_operator(run)) return true;
    run.clear();
    run_len = 0;
  }
  return false;
}

std::pair<float, float> width(const std::vector<G> &all, const Idx &gs) {
  float x0 = std::numeric_limits<float>::max(), x1 = std::numeric_limits<float>::lowest();
  for (int g : gs) {
    x0 = std::min(x0, all[g].x0);
    x1 = std::max(x1, all[g].x1);
  }
  return {x0, x1};
}

bool is_radical(std::string_view t) {
  using namespace std::literals;
  return t == "\0SQRT"sv || t == "\0\\surd "sv || t == "√"sv || t == "\0\\surd"sv;
}

// Replace `parts` in `live` with a new stand-in glyph.
Idx replace(std::vector<G> &all, const Idx &live, const Idx &parts, G sy) {
  const std::unordered_set<int> gone(parts.begin(), parts.end());
  all.push_back(std::move(sy));
  const int fresh = (int)all.size() - 1;
  Idx out;
  out.reserve(live.size() + 1);
  for (int g : live)
    if (!gone.count(g)) out.push_back(g);
  out.push_back(fresh);
  return out;
}

}  // namespace

Idx fold(std::vector<G> &all, Idx live, const std::vector<Rule> &rules_in, float h) {
  if (live.empty() || rules_in.empty()) return live;
  std::vector<float> sizes;
  for (int g : live) sizes.push_back(all[g].size);
  const float body = median(std::move(sizes));
  std::vector<Rule> rules = rules_in;
  std::stable_sort(rules.begin(), rules.end(), [](const Rule &a, const Rule &b) { return total_less(a.w, b.w); });
  for (const Rule &r : rules) {
    if (r.w > 0.6f * 612.0f) continue;
    // Square root: a radical glyph whose top-right meets the rule's left end.
    Idx rad;
    for (int g : live)
      if (is_radical(all[g].t) && std::abs(all[g].x1 - r.x0) < 1.5f &&
          std::abs((h - all[g].top) - r.y) < 0.35f * all[g].size + 1.5f)
        rad.push_back(g);
    if (!rad.empty()) {
      Idx under;
      for (int g : live) {
        if (std::find(rad.begin(), rad.end(), g) != rad.end()) continue;
        const G &x = all[g];
        if (r.x0 - 0.5f <= x.cx() && x.cx() <= r.x1 + 0.5f && (h - x.top) <= r.y + 0.6f && r.y - (h - x.top) < 1.6f * x.size)
          under.push_back(g);
      }
      if (!under.empty()) {
        const float s = max_size(all, under);
        std::vector<float> bs;
        for (int g : under)
          if (all[g].size >= 0.85f * s) bs.push_back(all[g].base);
        const float nb = median(std::move(bs));
        Idx kept;
        for (int g : under)
          if (nb - 0.6f * s <= all[g].base && all[g].base <= nb + 0.8f * s) kept.push_back(g);
        const std::string tex = R"(\sqrt{)" + tex_of(all, kept) + "}";
        Idx parts = kept;
        parts.insert(parts.end(), rad.begin(), rad.end());
        G sy = stand_in(all, tex, parts, all[rad[0]].x0, r.x1, nb, s);
        live = replace(all, live, parts, std::move(sy));
        continue;
      }
    }
    const Idx num = stack(all, live, r, h, true);
    const Idx den = stack(all, live, r, h, false);
    const float tol = std::max(0.3f * body, 3.0f);
    if (!num.empty() && !den.empty()) {
      const auto [nx0, nx1] = width(all, num);
      const auto [dx0, dx1] = width(all, den);
      const float wide = std::max(nx1 - nx0, dx1 - dx0);
      const float rc = (r.x0 + r.x1) / 2.0f;
      const bool centred = std::abs((nx0 + nx1) / 2.0f - rc) < 0.15f * r.w + 1.5f &&
                           std::abs((dx0 + dx1) / 2.0f - rc) < 0.15f * r.w + 1.5f;
      if (r.w - tol <= wide && wide <= r.w + 1.0f && centred && !wordy(all, num) && !wordy(all, den)) {
        std::string tex = R"(\frac{)" + tex_of(all, num);
        tex += "}{" + tex_of(all, den) + "}";
        Idx parts = num;
        parts.insert(parts.end(), den.begin(), den.end());
        const float part = max_size(all, parts);
        // Note: display fractions set their parts at full size, text-style ones
        // at script size (so the fraction's own size is ~1/0.7 of its parts').
        const float size = part >= 0.92f * body ? part : std::min(body, part / 0.7f);
        G sy = stand_in(all, tex, parts, r.x0, r.x1, r.y - 0.25f * size, size);
        live = replace(all, live, parts, std::move(sy));
        continue;
      }
    }
    // Overline: math glyphs right under the rule, as wide as it.
    if (!den.empty()) {
      const auto [dx0, dx1] = width(all, den);
      bool close = false, any_math = false;
      int top_g = den[0];
      for (int g : den) {
        close |= r.y - (h - all[g].top) < 0.45f * all[g].size + 1.0f;
        any_math |= all[g].role != Role::None;
      }
      for (size_t k = 1; k < den.size(); ++k)
        if (all[den[k]].top < all[top_g].top) top_g = den[k];
      if (close && any_math && std::abs((dx1 - dx0) - r.w) < std::max(0.3f * body, 2.5f) &&
          r.y - (h - all[top_g].top) < 0.5f * body) {
        const std::string tex = R"(\overline{)" + tex_of(all, den) + "}";
        const float s = max_size(all, den);
        std::vector<float> bs;
        for (int g : den)
          if (all[g].size >= 0.85f * s) bs.push_back(all[g].base);
        const float nb = median(std::move(bs));
        G sy = stand_in(all, tex, den, dx0, dx1, nb, s);
        live = replace(all, live, den, std::move(sy));
      }
    }
  }
  return live;
}

}  // namespace hp::tex::detail
