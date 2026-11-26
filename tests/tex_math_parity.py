#!/usr/bin/env python3
"""Parity of hp's TeX math pass (src/tex_math) against LiteParse's tex_math.

The reference is a LiteParse build that prints its math spans: in
crates/liteparse/src/extract.rs (extract_text closure, right after
`let spans = page_math_spans(..)`), add

    if std::env::var_os("LITEPARSE_MATH_DUMP").is_some() {
        for s in &spans {
            eprintln!("MATHSPAN {}", serde_json::json!({"page": page_number, "tex": s.tex,
                "x0": s.x0, "x1": s.x1, "top": s.top, "bottom": s.bottom, "size": s.size,
                "chars": s.chars, "emit": s.has_structure()}));
        }
    }

and build `cargo build --release --no-default-features -p liteparse --bin liteparse`.

    tests/tex_math_parity.py --liteparse <patched liteparse> --hp build/tex_math_parity [--list F] <pdf>...

Spans are matched per page by the char indices they consume; a match is exact
when the TeX strings are equal. Boxes are compared on matched spans.
"""
import argparse
import collections
import json
import os
import subprocess
import sys


def ref_spans(lp, pdf):
    env = dict(os.environ, LITEPARSE_MATH_DUMP="1")
    r = subprocess.run([lp, "parse", pdf, "--format", "json", "--no-ocr", "-q", "-o", os.devnull],
                       env=env, capture_output=True, text=True)
    out, seen = [], set()
    for line in r.stderr.splitlines():
        if not line.startswith("MATHSPAN "):
            continue
        s = json.loads(line[len("MATHSPAN "):])
        key = (s["page"], tuple(s["chars"]), s["tex"])
        if key in seen:  # pages with form widgets are extracted twice
            continue
        seen.add(key)
        out.append(s)
    return out


def hp_spans(hp, pdfs):
    r = subprocess.run([hp, "--all", *pdfs], capture_output=True, text=True, check=True)
    spans, ms = collections.defaultdict(list), {}
    for line in r.stdout.splitlines():
        s = json.loads(line)
        if "ms" in s:
            ms[s["file"]] = (s["pages"], s["ms"])
        else:
            spans[s["file"]].append(s)
    return spans, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liteparse", required=True)
    ap.add_argument("--hp", required=True)
    ap.add_argument("--show", type=int, default=40, help="diffs to print")
    ap.add_argument("--list", help="file with one PDF path per line")
    ap.add_argument("pdfs", nargs="*")
    a = ap.parse_args()
    if a.list:
        a.pdfs += [l.strip() for l in open(a.list) if l.strip()]

    hp, ms = hp_spans(a.hp, a.pdfs)
    tot = collections.Counter()
    diffs, box_err = [], []
    for pdf in a.pdfs:
        ref = ref_spans(a.liteparse, pdf)
        got = hp.get(pdf, [])
        for which in ("all", "emit"):
            r = [s for s in ref if which == "all" or s["emit"]]
            g = [s for s in got if which == "all" or s["emit"]]
            tot[which + "_ref"] += len(r)
            tot[which + "_hp"] += len(g)
            gi = collections.defaultdict(list)
            for s in g:
                gi[(s["page"], tuple(s["chars"]))].append(s)
            rest = collections.Counter((s["page"], s["tex"]) for s in g)
            for s in r:
                k = (s["page"], tuple(s["chars"]))
                m = gi[k].pop(0) if gi[k] else None
                if m and m["tex"] == s["tex"]:
                    tot[which + "_exact"] += 1
                    if which == "emit":
                        box_err.append(max(abs(m[f] - s[f]) for f in ("x0", "x1", "top", "bottom")))
                elif which == "emit":
                    diffs.append((pdf, s["page"], s["tex"], m["tex"] if m else None, s["chars"]))
                if rest[(s["page"], s["tex"])] > 0:
                    rest[(s["page"], s["tex"])] -= 1
                    tot[which + "_tex"] += 1
            if which == "emit":
                for k, left in gi.items():
                    for s in left:
                        diffs.append((pdf, k[0], None, s["tex"], list(k[1])))
        tot["files_with_ref_spans"] += bool(ref)
        tot["files_with_hp_spans"] += bool(got)

    for which in ("emit", "all"):
        n = tot[which + "_ref"]
        print(f"{which:5s} spans: liteparse {n}, hp {tot[which + '_hp']}, "
              f"exact (same chars + tex) {tot[which + '_exact']} = {100 * tot[which + '_exact'] / max(n, 1):.2f}%, "
              f"same tex on page {tot[which + '_tex']} = {100 * tot[which + '_tex'] / max(n, 1):.2f}%")
    if box_err:
        be = sorted(box_err)
        q = lambda p: be[min(len(be) - 1, int(p * len(be)))]
        print(f"box |diff| over matched emitted spans (pt): median {q(0.5):.4f}, p95 {q(0.95):.4f}, "
              f"max {be[-1]:.4f}, <=0.05pt {100 * sum(e <= 0.05 for e in be) / len(be):.2f}%")
    print(f"files with spans: liteparse {tot['files_with_ref_spans']}, hp {tot['files_with_hp_spans']} "
          f"of {len(a.pdfs)}")
    pages = sum(p for p, _ in ms.values())
    print(f"page_math CPU: {sum(m for _, m in ms.values()):.1f} ms over {pages} pages "
          f"({sum(m for _, m in ms.values()) / max(pages, 1):.3f} ms/page)")
    for pdf, page, want, got, chars in diffs[: a.show]:
        print(f"DIFF {os.path.basename(pdf)} p{page} chars={chars[:6]}{'...' if len(chars) > 6 else ''}\n"
              f"  liteparse: {want}\n  hp:        {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
