"""INT8 (Q/DQ, rest FP16) versions of OrderNet and LayoutNet with ModelOpt, calibrated on CPU from the
padded calibration batches (padding lines masked out)."""
import glob, sys
import numpy as np
from modelopt.onnx.quantization import quantize

Q = "/home/logx/.claude/jobs/784b7a0d/tmp/quant"
N = 256
which = sys.argv[1]

if which == "order":
    feats, mask = [], []
    for f in sorted(glob.glob(f"{Q}/calib_order/*.npz")):
        d = np.load(f); n = min(d["feats"].shape[1], N)
        x = np.zeros((1, N, 16), np.float32); x[0, :n] = d["feats"][0, :n]
        m = np.zeros((1, N), bool); m[0, :n] = True
        feats.append(x); mask.append(m)
    calib = {"feats": np.concatenate(feats), "mask": np.concatenate(mask)}
    quantize("/home/logx/.cache/huggingface/hub/models--datalab-to--indra-ocr/snapshots/bd02167fdf2005a19962cc15188f8496a2c87d3b/order.onnx",
             quantize_mode="int8", calibration_data=calib, calibration_shapes=f"feats:1x{N}x16,mask:1x{N}",
             calibration_eps=["cpu"], high_precision_dtype="fp16", output_path=f"{Q}/order_int8.onnx")
else:
    cols = {k: [] for k in ["feats", "mask", "pairf", "image", "boxes", "valid"]}
    for f in sorted(glob.glob(f"{Q}/calib_table/*.npz")):
        d = np.load(f); n = min(d["feats"].shape[1], N)
        x = np.zeros((1, N, 68), np.float32); x[0, :n] = d["feats"][0, :n]
        x[0, n:, 66:68] = -1.0  # absent block / leaf ids, as real units
        m = np.zeros((1, N), bool); m[0, :n] = True
        p = np.zeros((1, N, N, 2), np.float32); p[0, :n, :n] = d["pairf"][0, :n, :n]
        b = np.zeros((1, N, 4), np.float32); b[0, :n] = d["boxes"][0, :n]
        for k, v in (("feats", x), ("mask", m), ("pairf", p), ("image", d["image"]), ("boxes", b), ("valid", d["valid"])):
            cols[k].append(v)
    calib = {k: np.concatenate(v) for k, v in cols.items()}
    quantize("/home/logx/.cache/huggingface/hub/models--logxdx--hyperparse-models/snapshots/f7ec3a8f436d999407be2ef8054502b4f3d38aac/tablenet.onnx".replace("/home/logx/.cache/huggingface/hub/models--logxdx--hyperparse-models/snapshots/f7ec3a8f436d999407be2ef8054502b4f3d38aac/tablenet.onnx", f"{Q}/tablenet_fixmask.onnx"),
             quantize_mode="int8", calibration_data=calib,
             calibration_shapes=f"feats:1x{N}x68,mask:1x{N},pairf:1x{N}x{N}x2,image:1x1x1024x1024,boxes:1x{N}x4,valid:1x2",
             calibration_eps=["cpu"], high_precision_dtype="fp16", output_path=f"{Q}/tablenet_fixmask_int8.onnx")
print("done", which)
