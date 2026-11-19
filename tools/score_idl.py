"""Word-level F1 of hyperparse output vs the IDL dataset's Textract OCR (pixparse/idl-wds).
  python tools/score_idl.py <idl_scans_dir> <out.jsonl> [...]   (out.jsonl: hyperparse --out of the
  merged PDF of the sorted *.pdf in idl_scans_dir)
"""
import json, re, subprocess, sys, collections, glob, os
d = sys.argv[1]
pdfs = sorted(glob.glob(os.path.join(d, "*.pdf")))
gt = []
for p in pdfs:
    pages = json.load(open(p[:-4] + ".json"))["pages"]
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", p], capture_output=True, text=True).stdout).group(1))
    for i in range(n):
        gt.append(pages[i]["text"] if i < len(pages) else None)
W = lambda lines: collections.Counter(w.casefold() for t in lines for w in re.findall(r"\w+", t))
for f in sys.argv[2:]:
    R = {json.loads(l)["page"]: json.loads(l) for l in open(f)}
    tp = nh = ng = 0; pages = 0
    for p, r in R.items():
        if p >= len(gt) or gt[p] is None: continue
        g, h = W(gt[p]), W([l["text"] for l in r["lines"]])
        tp += sum((g & h).values()); nh += sum(h.values()); ng += sum(g.values()); pages += 1
    P, Rc = tp / max(nh, 1), tp / max(ng, 1)
    print(f"{os.path.basename(f):22s} {pages} pages: word P {P:.4f} R {Rc:.4f} F1 {2*P*Rc/(P+Rc):.4f}")
