# LayoutNet parity tooling

`src/tablenet/` is a C++ port of liteparse's table model (`crates/liteparse/src/table_model/`,
Apache-2.0). These files reproduce the checks it was validated with.

## Reference dumps

1. Get liteparse at commit `b2ca8f00e8c0b8e78c4e40bd0c2a60ce9d48f3f2` and, from
   `crates/liteparse/src`, run `python3 liteparse_dump_patch.py`. It adds two switches to the
   table model: `LP_TN_NOPROJ=1` (no LiteParse projection signals, as hp runs it) and
   `LP_TN_DUMP=<dir>` (per page: units, features, rules, image, raw logits, decoded tables).
2. `cargo build --release -p liteparse --bin lit --no-default-features --features ocria-nano`
   and install the model once (`lit models pull ocria-nano`).
3. `LP_TN_NOPROJ=1 LP_TN_DUMP=dumps/x lit parse doc.pdf --format markdown -o /dev/null -q`
   (for image-only PDFs add `--dpi 150`, the render hp's OCR pages use).

## Checks

- `build/tablenet_decode_test dumps/x ...`: decoding (`decode.cpp`) byte for byte.
- `build/tablenet_parity doc.pdf dumps/x <engine cache>/tablenet_3b913866.plan`: units, ruling
  lines, image branch, features, TensorRT logits, decoded tables and the whole page, per page.

Results when this was ported (OmniDocBench table pages as image PDFs; olmOCR-bench subset):
decoding 478/478 pages byte-exact; OCR pages: rules, image and features exact, logits within
0.021 of ONNX Runtime, tables 100/100; text-layer pages: units 75/76, features exact, logits
within 0.023, tables 140/140.

## Unicode tables

`src/tablenet/unicode_tables.inc` holds Rust's `char::is_alphabetic` / `is_uppercase` /
`is_lowercase` / `is_numeric` / `is_whitespace` as code point ranges (glibc's classes differ, and
the features notice). Regenerate with a `cargo new` project whose `main.rs` is
`gen_unicode_tables.rs`: `cargo run --release > src/tablenet/unicode_tables.inc`.
