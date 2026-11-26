"""Offline estimate of the table clean-up (src/table_fix.{h,cpp}, src/table_grid.cpp) on saved
ParseBench results, without the GPU: the tables in hp's cached server output are re-written by the
real C++ code (`table_fix_test --cli`), spliced into the saved Markdown, and re-scored with the
stock parse-bench evaluator.

  <parse-bench env>/bin/python -I tools/table_fix/rescore.py --bin build/table_fix_test \
      --root <nano-report dir> --bench pb1|pb2 [--variants base,full,...] [--out results.json]

Inputs per doc: the saved InferenceResult (raw_output = what hplit printed, i.e. hp's page
Markdown), and hp's raw server JSON (pb/cache/dedup/<sha256 of the source file>.json: per page
layout, lines, tables with `html` (LayoutNet) or `markdown` (geometry grid)).

What is reconstructed, and how faithfully:
  text               pages with a text layer: `table_fix_test --page` re-reads it on the CPU exactly
                     as the server does (native items, LayoutNet's unit words); OCR text comes from
                     the saved lines, which are merged lines (the server used the OCR boxes).
  LayoutNet tables   the HTML is parsed back into a tn::Grid (cells, spans, <th> rows). Cell boxes
                     are not saved: each cell's words are located on the page (row by row, on the
                     line most of the row's cells share), so they are close but not exact.
  geometry grids     table_markdown is re-run on the region's spans and used when it reproduces the
                     saved Markdown with the new passes off; otherwise the saved table's cells are
                     re-written by write_grid (the same header/clean-up code) with located boxes.
Variants: base (saved output; must reproduce the official score), repro (both writers with every
new pass off: measures the reconstruction itself), full (everything on), full_lp (header band
exactly as liteparse: top row must have an empty slot), and ablations model / model_lp /
model_band / model_band_lp / model_fix / grid / grid_stack / grid_fix.
"""

import argparse
import collections
import contextlib
import hashlib
import html as htmlmod
import io
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PB2_SUBSETS = ["tables", "complex_table_headers", "financial_tables", "insurance_filing_tables", "dense_tables"]
PB2_WEIGHTS = {"tables": 0.5, "complex_table_headers": 0.125, "financial_tables": 0.125,
               "insurance_filing_tables": 0.125, "dense_tables": 0.125}

# variant -> (model options or None = keep saved, grid options or None = keep saved)
OFF_M = {"band": False, "clean_up": False, "headless": False}
OFF_G = {"stack_header": False, "clean_up": False}
VARIANTS = {
    "base": (None, None),
    "repro": (OFF_M, OFF_G),
    "full": ({"band": True, "clean_up": True, "headless": True}, {"stack_header": True, "clean_up": True}),
    "full_lp": ({"band": True, "clean_up": True, "headless": False}, {"stack_header": True, "clean_up": True}),
    "model": ({"band": True, "clean_up": True, "headless": True}, None),
    "model_lp": ({"band": True, "clean_up": True, "headless": False}, None),
    "model_band": ({"band": True, "clean_up": False, "headless": True}, None),
    "model_band_lp": ({"band": True, "clean_up": False, "headless": False}, None),
    "model_fix": ({"band": False, "clean_up": True, "headless": False}, None),
    "grid": (None, {"stack_header": True, "clean_up": True}),
    "grid_stack": (None, {"stack_header": True, "clean_up": False}),
    "grid_fix": (None, {"stack_header": False, "clean_up": True}),
}

CELL = re.compile(r'<(th|td)((?: (?:rowspan|colspan)="\d+")*)>(.*?)</\1>', re.S)
ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
SPAN = re.compile(r'(rowspan|colspan)="(\d+)"')


def parse_hp_tables(html: str):
    """hp's LayoutNet HTML (tn::Grid::to_html, possibly several tables) -> list of grids (cells
    with slots; boxes filled later)."""
    grids = []
    for t in re.findall(r"<table>(.*?)</table>", html, re.S):
        rows = ROW.findall(t)
        occ = [set() for _ in rows]
        cells, head = [], []
        for r, row in enumerate(rows):
            c = 0
            is_head = False
            for tag, attrs, text in CELL.findall(row):
                is_head |= tag == "th"
                while c in occ[r]:
                    c += 1
                sp = {k: int(v) for k, v in SPAN.findall(attrs)}
                rs, cs = sp.get("rowspan", 1), sp.get("colspan", 1)
                for rr in range(r, min(r + rs, len(rows))):
                    occ[rr].update(range(c, c + cs))
                text = htmlmod.unescape(text)
                if text != "" or rs > 1 or cs > 1:
                    cells.append({"r0": r, "c0": c, "r1": r + rs - 1, "c1": c + cs - 1, "text": text})
                c += cs
            if is_head:
                head.append(r)
        n_cols = max((max(o) + 1 for o in occ if o), default=0)
        grids.append({"cells": cells, "head_rows": head, "n_rows": len(rows), "n_cols": n_cols})
    return grids


def centre_in(b, box, pad=0.0):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return box[0] - pad <= cx <= box[2] + pad and box[1] - pad <= cy <= box[3] + pad


def line_words(text, b):
    """Words of a line (box b): a word's x range is its character range within the box."""
    n = max(len(text), 1)
    return [(m.group(), [b[0] + (b[2] - b[0]) * m.start() / n, b[1], b[0] + (b[2] - b[0]) * m.end() / n, b[3]])
            for m in re.finditer(r"\S+", text)]


def page_source(page, dump):
    """The page's text as the server's table code saw it: spans (px, the items before line merging)
    and words with boxes (points). With a text-layer dump (table_fix_test --page) of a page that
    has native text, the native items and LayoutNet's unit words are exact and only the OCR lines
    come from the (merged) saved lines; otherwise everything comes from the saved lines."""
    k = page["dpi"] / 72.0
    lines = page.get("lines", [])
    if dump is not None and dump["route"] != 2 and page.get("route") != "full_ocr" and not page.get("rotation"):
        ocr = [l for l in lines if l.get("src") != "pdf"]
        spans = [[t, x0 * k, y0 * k, x1 * k, y1 * k] for t, x0, y0, x1, y1 in dump["items"]]
        spans += [[l["text"], *l["bbox"]] for l in ocr]
        words = [(t, [x0, y0, x1, y1]) for t, x0, y0, x1, y1 in dump["words"]]
        for l in ocr:
            words += line_words(l["text"], [v / k for v in l["bbox"]])
        return spans, words, True
    spans = [[l["text"], *l["bbox"]] for l in lines]
    words = [w for l in lines for w in line_words(l["text"], [v / k for v in l["bbox"]])]
    return spans, words, False


def page_words(words, region=None, scale=1.0):
    """The words near `region` (all words if None; in the words' units x scale), top-most then
    left-most first."""
    big = None
    if region is not None:
        rx = 0.05 * (region[2] - region[0])
        ry = 0.05 * (region[3] - region[1])
        big = [region[0] - rx, region[1] - ry, region[2] + rx, region[3] + ry]
    out = []
    for t, b in words:
        b = [v * scale for v in b]
        if big is None or centre_in(b, big):
            out.append({"t": t, "bx": b, "key": (b[1], b[0]), "used": False})
    out.sort(key=lambda w: w["key"])
    return out


def locate_cells(grid, words):
    """Give every cell a box: the union of its words found in `words` (top-most, left-most unused
    occurrence first, in row-major order); cells with no word found take their column's x range
    and their row's y range."""
    cells = sorted(grid["cells"], key=lambda c: (c["r0"], c["c0"]))
    # each grid row sits on the text line most of its cells' first words share (a page can repeat
    # a table's labels and values elsewhere), below the previous row's line
    by_tok = collections.defaultdict(list)
    for w in words:
        by_tok[w["t"]].append(w)
    prev_y = -1e9
    for r in sorted({c["r0"] for c in cells}):
        row = [c for c in cells if c["r0"] == r and c["text"].split()]
        ys = [[w["bx"][1] for w in by_tok[c["text"].split()[0]] if not w["used"]] for c in row]
        votes = collections.Counter()
        for y in {y for v in ys for y in v}:
            votes[y] = sum(1 for v in ys if any(abs(y - u) <= 3.0 for u in v))
        if votes:
            y_row = max(votes, key=lambda y: (votes[y], y > prev_y - 2, -y))
            prev_y = y_row
        else:
            y_row = prev_y
        for c in row:
            bx = None
            ref = (y_row, -1e9)
            for tok in c["text"].split():
                cand = [w for w in by_tok[tok] if not w["used"]]
                if not cand:
                    continue
                w = min(cand, key=lambda w: (abs(w["bx"][1] - ref[0]) + 0.01 * abs(w["bx"][0] - ref[1]) if ref[1] > -1e8
                                             else abs(w["bx"][1] - ref[0]), w["key"]))
                w["used"] = True
                b = w["bx"]
                ref = (b[1], b[2])
                bx = b[:] if bx is None else [min(bx[0], b[0]), min(bx[1], b[1]), max(bx[2], b[2]), max(bx[3], b[3])]
            c["bx"] = bx
    for c in cells:
        c.setdefault("bx", None)
    for c in cells:
        if c["bx"] is not None:
            continue
        xs = [d["bx"] for d in cells if d["bx"] is not None and d["c0"] == c["c0"] and d["c1"] == c["c1"]]
        ys = [d["bx"] for d in cells if d["bx"] is not None and d["r0"] == c["r0"]]
        near = [d["bx"] for d in cells if d["bx"] is not None]
        if not near:
            c["bx"] = [0.0, 0.0, 0.0, 0.0]
            continue
        x0, x1 = (min(b[0] for b in xs), max(b[2] for b in xs)) if xs else (near[0][0], near[0][2])
        y0, y1 = (min(b[1] for b in ys), max(b[3] for b in ys)) if ys else (near[0][1], near[0][3])
        c["bx_est"] = [x0, y0, x1, y1]
    for c in cells:
        if c["bx"] is None:
            c["bx"] = c.pop("bx_est")
    grid["cells"] = [c for c in grid["cells"]]


def parse_pipe(md: str):
    """A pipe table as table_grid writes it -> rows of cell text (the rule line dropped)."""
    rows = []
    for i, line in enumerate(md.split("\n")):
        if i == 1:
            continue
        parts = re.split(r"(?<!\\)\|", line)[1:-1]
        rows.append([p[1:-1].replace("\\|", "|") if len(p) >= 2 else "" for p in parts])
    return rows


def cells_request(md, spans, all_words, region, k):
    """A saved grid table's cells with boxes and row bands located in the region's words (px)."""
    rows = parse_pipe(md)
    words = page_words(all_words, region, k)
    hs = sorted(s[4] - s[2] for s in spans if centre_in(s[1:], region))
    h = hs[len(hs) // 2] if hs else 10.0
    boxes, bands = [], []
    for row in rows:
        brow = []
        for text in row:
            bx = None
            for tok in text.split():
                w = next((w for w in words if not w["used"] and w["t"] == tok), None)
                if w is None:
                    continue
                w["used"] = True
                b = w["bx"]
                bx = b[:] if bx is None else [min(bx[0], b[0]), min(bx[1], b[1]), max(bx[2], b[2]), max(bx[3], b[3])]
            brow.append(None if bx is None else [bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]])
        found = [b for b in brow if b]
        if found:
            bands.append([min(b[1] for b in found), max(b[1] + b[3] for b in found)])
        else:  # nothing located: one line below the previous row
            y = bands[-1][1] if bands else region[1]
            bands.append([y, y + h])
        boxes.append(brow)
    return {"kind": "cells", "rows": rows, "boxes": boxes, "bands": bands, "h": h}


def grid_box(g):
    bs = [c["bx"] for c in g["cells"] if c["text"].strip()]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def page_requests(page, dump=None):
    """CLI requests (without options) for each table of a cached server page."""
    k = page["dpi"] / 72.0
    spans_px, words_pt, exact = page_source(page, dump)
    out = []
    model = []
    for t in page.get("tables", []):
        region = page["layout"][t["layout"]]["bbox"]
        if t.get("html"):
            rp = [v / k for v in region]
            grids = parse_hp_tables(t["html"])
            words = page_words(words_pt)  # a LayoutNet table can reach beyond its region
            for g in grids:
                locate_cells(g, words)
            model.append((t, rp, grids))
        elif t.get("markdown"):
            spans = [s for s in spans_px if centre_in(s[1:], region)]
            out.append({"old": t["markdown"], "req": {"kind": "grid", "spans": spans}, "exact": exact,
                        "alt": cells_request(t["markdown"], spans_px, words_pt, region, k)})
    boxes = [b for _, _, gs in model for g in gs if (b := grid_box(g))]
    for t, rp, grids in model:
        spans, taken = [], []
        for s in spans_px:
            b = [v / k for v in s[1:]]
            if not centre_in(b, rp):
                continue
            spans.append([s[0], b[0], b[1], b[2] - b[0], b[3] - b[1]])
            taken.append(int(any(centre_in(b, gb, 1.0) for gb in boxes)))
        out.append({"old": t["html"], "req": {"kind": "model", "grids": grids, "spans": spans, "taken": taken},
                    "exact": exact})
    return out


def text_layer(binary, pdf, pages):
    """table_fix_test --page: {page index: dump} for a PDF (empty for images / on failure)."""
    if not pages or not str(pdf).lower().endswith(".pdf"):
        return {}
    p = subprocess.run([binary, "--page", str(pdf), *map(str, pages)], capture_output=True, text=True)
    if p.returncode != 0:
        return {}
    return {d["page"]: d for d in map(json.loads, p.stdout.splitlines())}


def run_cli(binary, reqs):
    if not reqs:
        return []
    p = subprocess.run([binary, "--cli"], input="\n".join(json.dumps(r) for r in reqs) + "\n",
                       capture_output=True, text=True, check=True)
    return [json.loads(l) for l in p.stdout.splitlines()]


def sha_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


BAND_LOG = []  # LayoutNet tables a header band joined (written next to --out)


def plan(binary, root, files):
    """Per doc: {variant: [(old, new), ...]} and reconstruction stats."""
    stats = collections.Counter()
    docs = {}
    jobs = []  # (doc, variant, old, request)
    for f in files:
        raw = json.loads(Path(f).read_text())
        try:
            cache = json.loads((root / "pb/cache/dedup" / f"{sha_of(raw['request']['source_file_path'])}.json").read_text())
        except FileNotFoundError:
            stats["no_cache"] += 1
            continue
        docs[f] = {v: [] for v in VARIANTS}
        pages = [p for p in cache["pages"] if "error" not in p and p.get("tables")]
        dumps = text_layer(binary, raw["request"]["source_file_path"], [p["page"] for p in pages])
        for page in pages:
            for item in page_requests(page, dumps.get(page["page"])):
                kind = item["req"]["kind"]
                stats[f"tables_{kind}"] += 1
                stats[f"tables_{kind}_{'text_layer' if item['exact'] else 'saved_lines'}"] += 1
                for v, (mo, go) in VARIANTS.items():
                    o = mo if kind == "model" else go
                    if o is None:
                        continue
                    jobs.append((f, v, item["old"], "req", {**item["req"], **o}))
                    if "alt" in item:
                        jobs.append((f, v, item["old"], "alt", {**item["alt"], **o}))
    outs = run_cli(binary, [j[4] for j in jobs])
    # per table: which reconstruction reproduces the saved text with every new pass off
    # (model: always the HTML parse; grid: the exact re-run on the lines, else the saved cells)
    res_of, how = {}, {}
    for (f, v, old, src, req), res in zip(jobs, outs):
        res_of[(f, v, old, src)] = res["html"] if req["kind"] == "model" else res["text"]
    for (f, v, old, src, req) in jobs:
        if v != "repro" or (f, old) in how:
            continue
        if req["kind"] == "model":
            how[(f, old)] = "req"
            stats["repro_model_" + ("same" if res_of[(f, v, old, "req")] == old else "diff")] += 1
        elif res_of[(f, v, old, "req")] == old:
            how[(f, old)] = "req"
            stats["grid_exact_rerun"] += 1
        elif res_of.get((f, v, old, "alt")) == old:
            how[(f, old)] = "alt"
            stats["grid_saved_cells"] += 1
        else:
            how[(f, old)] = None
            stats["grid_not_reconstructed"] += 1
    seen = set()
    for (f, v, old, src, req), res in zip(jobs, outs):
        if how.get((f, old)) != src or (f, v, old) in seen:
            continue
        seen.add((f, v, old))
        new = res_of[(f, v, old, src)]
        if req["kind"] == "model" and res.get("absorbed") and v in ("full", "full_lp"):
            stats[f"band_tables_{v}"] += 1
            BAND_LOG.append({"doc": Path(f).name, "variant": v, "old": old, "new": res_of[(f, v, old, src)]})
        if new != old:
            stats[f"changed_{v}_{req['kind']}"] += 1
        docs[f][v].append((old, new))
    return docs, stats


_CONV = None
_TC = {}
_BENCH = None


def _init(root, bench):
    global _CONV, _BENCH
    from parse_bench.inference.providers.parse.liteparse import LiteParseProvider
    from parse_bench.test_cases.loader import load_test_cases
    _CONV = LiteParseProvider._convert_md_tables_to_html
    _BENCH = bench
    if bench == "pb1":
        for tc in load_test_cases(root / "pb/data"):
            if tc.test_id.startswith("table/"):
                _TC[tc.test_id] = tc
    else:
        for s in PB2_SUBSETS:
            for tc in load_test_cases(root / "pb/data_parsebench-2" / s):
                _TC[tc.test_id] = tc


def splice(text, edits):
    for old, new in edits:
        if old != new:
            text = text.replace(old, new)
    return text


def score(args):
    path, edits_by_variant = args
    from parse_bench.schemas.pipeline_io import InferenceResult
    raw = json.loads(Path(path).read_text())
    tc = _TC[raw["request"]["example_id"]]
    out = {"id": raw["request"]["example_id"]}
    for v, edits in edits_by_variant.items():
        ir = InferenceResult.model_validate(raw)
        ro = raw["raw_output"]
        if v != "base":
            texts = [_CONV(splice(p.get("text", "") or "", edits)) for p in ro.get("pages", [])]
            for p, t in zip(ir.output.pages, texts):
                p.markdown = t
            full = ro.get("text") or ""
            ir.output.markdown = _CONV(splice(full, edits)) if full else "\n\n".join(texts)
        with contextlib.redirect_stdout(io.StringIO()):
            if _BENCH == "pb1":
                from parse_bench.evaluation.evaluators.parse import ParseEvaluator
                res = ParseEvaluator(enable_rule_based=False, grits_pair_workers=1).evaluate(ir, tc)
                m = {x.metric_name: x.value for x in res.metrics}
                out[v] = m.get("grits_trm_composite", 0.0)
                out[v + "_trm"] = m.get("table_record_match")
                out[v + "_con"] = m.get("grits_con")
            else:
                from parse_bench.evaluation.metrics.parse.next_metrics import compute_next_metrics
                ms = {m.metric_name: m for m in compute_next_metrics(ir, tc)}
                m = ms.get("table_next")
                out[v] = m.value if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--bench", choices=["pb1", "pb2"], default="pb1")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    variants = a.variants.split(",")
    if a.bench == "pb1":
        files = sorted((a.root / "pb/out_parsebench_dedup/liteparse_tesseract/table").glob("*.result.json"))
    else:
        files = []
        for s in PB2_SUBSETS:
            files += sorted((a.root / "pb/out_parsebench-2_dedup/parsebench-2" / s / "liteparse_tesseract" / s).glob("*.result.json"))
    if a.limit:
        files = files[: a.limit]
    docs, stats = plan(str(Path(a.bin).resolve()), a.root, files)
    print("reconstruction:", dict(sorted(stats.items())), flush=True)
    jobs = [(str(f), {v: docs.get(f, {}).get(v, []) for v in variants}) for f in files]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.root, a.bench)) as ex:
        rows = list(ex.map(score, jobs, chunksize=4))
    if a.out:
        a.out.write_text(json.dumps(rows))
        a.out.with_suffix(".bands.json").write_text(json.dumps(BAND_LOG, indent=1))
    if a.bench == "pb1":
        for v in variants:
            vals = [r[v] or 0.0 for r in rows]
            trm = [r[v + "_trm"] for r in rows if r.get(v + "_trm") is not None]
            con = [r[v + "_con"] for r in rows if r.get(v + "_con") is not None]
            print(f"{v:12s} n={len(vals)} grits_trm_composite={100 * sum(vals) / len(vals):.2f}"
                  f"  trm={100 * sum(trm) / max(len(trm), 1):.2f}  grits_con={100 * sum(con) / max(len(con), 1):.2f}"
                  f"  docs up/down vs base: {sum(1 for r in rows if (r[v] or 0) > (r['base'] or 0) + 1e-9)}"
                  f"/{sum(1 for r in rows if (r[v] or 0) < (r['base'] or 0) - 1e-9)}" if "base" in variants else "")
    else:
        for v in variants:
            dim = 0.0
            parts = []
            for s in PB2_SUBSETS:
                rs = [r for r in rows if r["id"].startswith(s + "/") and r.get("base", 0) is not None]
                mean = 100 * sum(r[v] or 0.0 for r in rs) / max(len(rs), 1)
                dim += PB2_WEIGHTS[s] * mean
                parts.append(f"{s}={mean:.1f}")
            print(f"{v:12s} tables_dimension={dim:.2f}  " + " ".join(parts))


if __name__ == "__main__":
    main()
