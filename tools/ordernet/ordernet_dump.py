"""Parity dumps for hp's OrderNet port, from Indra's own Python (indra_ocr 1.1.2, order.onnx).

  python -I ordernet_dump.py <hp pages.jsonl> <dump_dir>

Per page (p<page>.json), the "v2c" assembly hp implements:
  lines   OrderNet input: hp's lines minus those whose centre lies in a D-FINE Page-header /
          Page-footer region or in a formula / table region, then one line per formula
          ("$<latex>$", box = region) and one per table ("TBLHOLD<k>X", box = region)
  wh      page size (px)
  feats   line_features [N, 16]
  a, cls, same   OrderNet outputs [N, N], [N, 10], [N, N] (omitted when N > 500)
  order   decode_blocks_order(a, same, boxes, cls)
  blocks  split_blocks(order, same)
  markdown  page_markdown(boxes, texts, order, cls, same, keep_furniture=False)
"""
import json, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from indra_ocr.order_post import line_features, decode_blocks_order
from indra_ocr.layout_post import page_markdown, split_blocks

MODEL = Path("/home/logx/.cache/huggingface/hub/models--datalab-to--indra-ocr/snapshots/"
             "bd02167fdf2005a19962cc15188f8496a2c87d3b/order.onnx")
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])


def inside(b, r):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]


def v2c_lines(p):
    lines, lay = p["lines"], p["layout"]
    keep = list(range(len(lines)))
    for r in lay:
        if r["cls"] in ("Page-header", "Page-footer"):
            keep = [i for i in keep if not inside(lines[i]["bbox"], r["bbox"])]
    pb, pt = [], []
    for f in p.get("formulas", []):
        r = lay[f["layout"]]["bbox"]
        keep = [i for i in keep if not inside(lines[i]["bbox"], r)]
        latex = f["latex"].replace("$", "").strip()
        if latex: pb.append(r); pt.append("$" + latex + "$")
    for ti, tb in enumerate(p.get("tables", [])):
        r = lay[tb["layout"]]["bbox"]
        keep = [i for i in keep if not inside(lines[i]["bbox"], r)]
        pb.append(r); pt.append(f"TBLHOLD{ti}X")
    boxes = [lines[i]["bbox"] for i in keep] + pb
    texts = [lines[i]["text"] for i in keep] + pt
    return boxes, texts


src, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
n_pages = 0
for line in open(src):
    p = json.loads(line)
    boxes, texts = v2c_lines(p)
    if len(texts) < 2: continue
    wh = (p["w"], p["h"])
    bx = np.array(boxes, np.float32).reshape(-1, 4)
    f = line_features(bx, wh, texts)
    a, cls, same, tab = (o[0] for o in sess.run(None, {"feats": f[None], "mask": np.ones((1, len(texts)), bool)}))
    order = decode_blocks_order(a, same, bx, cls)
    md = page_markdown(bx, texts, order, cls, same, keep_furniture=False)
    small = len(texts) <= 500
    d = {"page": p["page"], "wh": list(wh), "boxes": bx.tolist(), "texts": texts,
         "feats": f.ravel().tolist(), "order": [int(i) for i in order],
         "blocks": [[int(i) for i in b] for b in split_blocks(order, same)], "markdown": md}
    if small:
        d.update(a=a.ravel().tolist(), cls=cls.ravel().tolist(), same=same.ravel().tolist())
    json.dump(d, open(out / f"p{p['page']:04d}.json", "w"), ensure_ascii=False)
    n_pages += 1
print(f"{src}: {n_pages} pages dumped to {out}")
