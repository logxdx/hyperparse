"""Calibration inputs (NPZ dirs, one batch each) for OrderNet and LayoutNet, from the parity dumps."""
import json, random
from pathlib import Path
import numpy as np

T = Path("/home/logx/.claude/jobs/784b7a0d/tmp")
Q = T / "quant"

# OrderNet: feats [1, N, 16], mask [1, N]
files = sorted((T / "ondumps").glob("*/p*.json"))
random.Random(0).shuffle(files)
out = Q / "calib_order"; out.mkdir(parents=True, exist_ok=True)
k = 0
for f in files:
    d = json.loads(f.read_text())
    n = len(d["texts"])
    if n < 8 or n > 600: continue
    np.savez(out / f"b{k:03d}.npz", feats=np.array(d["feats"], np.float32).reshape(1, n, 16), mask=np.ones((1, n), bool))
    k += 1
    if k == 64: break
print("ordernet calib batches:", k)

# LayoutNet: feats [1,N,68], mask, pairf [1,N,N,2], image [1,1,1024,1024], boxes [1,N,4], valid [1,2]
files = sorted((T / "dumps/omt_noproj").glob("p*.json")) + sorted((T / "dumps/olm_noproj").glob("p*.json"))
files = [f for f in files if not f.name.endswith("_img.u8")]
random.Random(0).shuffle(files)
out = Q / "calib_table"; out.mkdir(parents=True, exist_ok=True)
k = 0
for f in files:
    d = json.loads(f.read_text())
    n = d["n"]
    if d.get("pairf") is None or d.get("img") is None or n < 8 or n > 400: continue
    w, h = d["img"]["w"], d["img"]["h"]
    px = np.frombuffer((f.parent / (f.stem + "_img.u8")).read_bytes(), np.uint8).reshape(h, w)
    canvas = np.zeros((1, 1, 1024, 1024), np.float32)
    canvas[0, 0, :min(h, 1024), :min(w, 1024)] = (255.0 - px[:1024, :1024].astype(np.float32)) / 255.0
    np.savez(out / f"b{k:03d}.npz",
             feats=np.array(d["feats"], np.float32).reshape(1, n, 68), mask=np.ones((1, n), bool),
             pairf=np.array(d["pairf"], np.float32).reshape(1, n, n, 2), image=canvas,
             boxes=np.array(d["img_boxes"], np.float32).reshape(1, n, 4),
             valid=np.array(d["img_valid"], np.float32).reshape(1, 2))
    k += 1
    if k == 32: break
print("layoutnet calib batches:", k)
