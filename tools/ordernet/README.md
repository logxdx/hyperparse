# OrderNet parity tooling

`src/ordernet/` ports indra_ocr 1.1.2's `order_post.py` (line features, block-order decoding) and
`layout_post.py` (blocks -> Markdown) to C++; `order.onnx` itself runs on TensorRT.

- `ordernet_dump.py <hp pages.jsonl> <dump_dir>` (run with an env that has `indra_ocr` and
  onnxruntime): per page, the OrderNet input hp builds (`src/ordernet/page.cpp`: furniture
  dropped, formula / table regions as one line each), the features, the network outputs, the
  decoded order and blocks, and the Markdown, all from Indra's Python. Feed it hp's `--out` JSONL.
- `build/ordernet_parity_test <dump_dir>...`: features, order, blocks and Markdown against the
  dumps (CPU only; network outputs come from the dump).
- `gen_unicode_py.py`: regenerates `src/ordernet/unicode_py.inc` (Python's `str` classes, walked
  over every code point by the Python the reference ran on).

When ported: 334 pages (OmniDocBench, its table pages, olmOCR-bench subset): features within
2.4e-7 (numpy's float32 log), order, blocks and Markdown exact.
