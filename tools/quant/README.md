# Reduced-precision LayoutNet / OrderNet

How `tablenet_{fp16,int8}.onnx` and `order_{fp16,int8}.onnx` in `logxdx/hyperparse-models` were
made, with NVIDIA ModelOpt 0.48 and ONNX Runtime on CPU (`CUDA_VISIBLE_DEVICES=`). The scripts
hold the paths they were run with at the top; edit them for another machine.

1. `make_calib.py`: calibration batches (NPZ) from the parity dumps (`tools/tablenet`,
   `tools/ordernet`): 64 OrderNet pages, 32 LayoutNet pages (OmniDocBench, olmOCR-bench).
2. `fix_mask.py`: LayoutNet's two attention-mask constants -1e9 -> -3e4 (-1e9 is infinite in
   FP16 and turns the softmax into NaN; -3e4 still gives masked entries a weight of exactly 0, and
   the FP32 model's tables are unchanged on every page).
3. FP16:
   `python -m modelopt.onnx.autocast --onnx_path <fp32.onnx> --output_path <fp16.onnx> -t fp16
   --calibration_data <calib dir> --keep_io_types --providers cpu`
4. INT8: `quant_int8.py order|table` (`modelopt.onnx.quantization.quantize`, int8, high precision
   FP16, lines / units padded to 256 and masked).

CPU checks against FP32 (ONNX Runtime; TensorRT numerics differ slightly):

| model | olmOCR-bench | OmniDocBench | identical to FP32 |
|---|---|---|---|
| OrderNet FP16 | 54.6 (FP32 54.1) | order tau 0.948 (0.947) | order 87%, Markdown 75% of pages |
| OrderNet INT8 | 54.1 | order tau 0.949 | order 76%, Markdown 51% |
| LayoutNet FP16 | | TEDS 0.511 (0.511) | tables on 94% / 99% of pages |
| LayoutNet INT8 | | TEDS 0.498 | tables on 48% / 89% of pages |
