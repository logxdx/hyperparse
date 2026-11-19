"""OCR accuracy against the PDF's own text layer (ground truth on born-digital pages).

  hyperparse doc.pdf --ocr none --out text.jsonl     # text-layer lines
  hyperparse doc.pdf --ocr all  --out ocr.jsonl      # OCR lines for every page
  python tools/ocr_vs_textlayer.py text.jsonl ocr.jsonl

Bag-of-words F1 per page (case-folded, punctuation stripped), over pages whose text
layer has >= 50 words.
"""
import json
import re
import sys
from collections import Counter

WORD = re.compile(r"\w+", re.UNICODE)


def words(lines):
    return Counter(w.casefold() for l in lines for w in WORD.findall(l["text"]))


def main():
    ref = {r["page"]: r for r in map(json.loads, open(sys.argv[1]))}
    hyp = {r["page"]: r for r in map(json.loads, open(sys.argv[2]))}
    f1s, tp_all, n_ref, n_hyp = [], 0, 0, 0
    for p, r in sorted(ref.items()):
        if r["mode"] != "text" or p not in hyp:
            continue
        a, b = words(r["lines"]), words(hyp[p]["lines"])
        if sum(a.values()) < 50:
            continue
        tp = sum((a & b).values())
        pr, rc = tp / max(sum(b.values()), 1), tp / max(sum(a.values()), 1)
        f1s.append((2 * pr * rc / (pr + rc) if pr + rc else 0.0, p))
        tp_all, n_ref, n_hyp = tp_all + tp, n_ref + sum(a.values()), n_hyp + sum(b.values())
    pr, rc = tp_all / max(n_hyp, 1), tp_all / max(n_ref, 1)
    print(f"{len(f1s)} born-digital pages: micro word P {pr:.4f} R {rc:.4f} F1 {2 * pr * rc / (pr + rc):.4f}"
          f" | mean page F1 {sum(f for f, _ in f1s) / len(f1s):.4f}")
    print("  lowest pages (F1, page):", [(round(f, 3), p) for f, p in sorted(f1s)[:6]])


if __name__ == "__main__":
    main()
