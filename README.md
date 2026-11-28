# hyperparse

A GPU PDF parser in C++/CUDA. It turns PDFs into layout regions, text lines,
LaTeX formulas and Markdown, at 24–65 pages per second on a 6 GB laptop GPU (scans and
formula-heavy papers at the low end, born-digital documents at the high end).

## Why

Built for my GPU-poor self, so I can convince myself I have a professional-grade
document parsing pipeline at home. 😢

Hosted document-parsing APIs are pricey, and the open-source pipelines usually
assume a big GPU and a Python stack. hyperparse is what's left once you
squeeze that onto one consumer GPU: TensorRT everywhere, OCR only where the PDF
doesn't already have good text, and no Python on the hot path.

## What you get

- **Layout**: 12 region classes per page (text, title, table, picture, formula, caption, code, stamp, …)
- **Reading order**: OrderNet orders the page's lines and groups them into blocks (paragraphs,
  headings, lists), around D-FINE's tables and formulas; `--no-ordernet` falls back to XY-cut
- **Precision**: LayoutNet and OrderNet run as FP16 TensorRT engines (`--tables-precision` /
  `--order-precision int8` for INT8); see `tools/quant`
- **Text**: the PDF's own text layer when it is usable, OCR where it isn't (per page or per region);
  scans turned sideways or upside down are turned upright first (PP-LCNet page orientation)
- **Inline styles**: bold, italic, code, strikethrough, underline and super- / subscripts from the
  text layer, as Markdown
- **Formulas**: display formulas as LaTeX; inline math in TeX-made PDFs as `$...$`, read from the
  PDF's glyphs (a port of liteparse's tex_math)
- **Tables**: table structure from LayoutNet (rows, columns, spanning cells, header rows) as HTML
  inside the Markdown, cleaned up as liteparse does (currency symbols onto their numbers, wrapped
  rows folded, group headers spanned, header bands above the table joined); `--no-tables` falls
  back to pipe tables built from the text's geometry
- **Markdown**: per page and for the whole document
- **CLI**: `hyperparse doc.pdf --md doc.md --out pages.jsonl`
- **HTTP server**: `hyperparse_server`, with a queue, streaming (NDJSON), Prometheus metrics,
  health checks and its API docs at `GET /docs`

Measured on an RTX 4050 Laptop GPU (6 GB):

| Set | Pages | Throughput | Notes |
|---|---|---|---|
| Server, 300 one-page documents, 8 in flight | 300 | ~64 pages/s | p50 latency ~124 ms |
| Mixed documents (ParseBench-1 sample, one PDF) | 302 | 62–66 pages/s | born-digital and scanned pages |
| Scanned pages | 98 | ~24 pages/s | every page OCR'd |
| arXiv papers | 100 | ~24.5 pages/s | formula-heavy |

With the models loaded it peaks at about 3.4 GB of GPU memory. They unload after an
idle timeout and reload on the next request in about 0.6 s. Full numbers:
[Benchmarks](#benchmarks).

## Benchmarks

All on one RTX 4050 Laptop GPU (6 GB, sm89), TensorRT 11.5, 8 page workers, November 2026.

### Document-parsing benchmarks (full sets)

The five public benchmarks of LlamaIndex's
[OCRia-nano-1 report](https://www.llamaindex.ai/ocria/ocria-nano-1-report.pdf), each with its official
scorer, run at 250 dpi. ParseBench and OmniParseBench keep running headers / footers (their ground
truth has them; hp's page-header / page-footer text is put back around the page Markdown), olmOCR-bench
and BHL drop them (hp's default). The other columns are the report's numbers (CPU parsers, 16 vCPUs);
LiteParse + OCRia-nano-1 run here through the same harnesses reproduced them (ParseBench-1 52.7,
OmniParseBench 74.8, ParseBench-2 59.3).

| Benchmark | hp | LiteParse + OCRia-nano-1 | Indra 1.1.2 | Marker (fast) | LiteParse (Tesseract) |
|---|---|---|---|---|---|
| [ParseBench-2](https://huggingface.co/datasets/llamaindex/ParseBench-2) (4,317 pages) | **61.8** | 59.2 | 55.1 | 52.5 | 48.7 |
| [OmniParseBench](https://github.com/datalab-to/omni_parse_bench) v1.7 (2,937 pages) | **76.8** | 74.8 | 70.9 | 69.8 | 42.3 |
| [ParseBench-1](https://huggingface.co/datasets/llamaindex/ParseBench) (2,079 docs) | **56.5** | 52.7 | 42.6 | 40.4 | 40.5 |
| [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) (1,403 PDFs) | **65.9** | 60.9 | 60.8 | 60.8 | 45.7 |
| [BHL IMPACT](https://huggingface.co/datasets/finebooks/bhl-impact-gt) CER, lower is better (2,165 pages) | 0.050 | **0.048** | 0.062 | 0.067 | 0.275 |

ParseBench-2 by dimension: text 70.7, formatting 81.3, tables 61.0, charts 4.9, layout 82.4, forms 70.4
(LiteParse + OCRia-nano-1: 68.5 / 82.2 / 59.5 / 3.1 / 72.4 / 69.8). OmniParseBench: text 65.1, tables
73.4, layout 92.1. olmOCR-bench: arXiv math 61.9, multi-column 75.9, tables 75.5, headers / footers 94.6,
long tiny text 81.2, old scans 26.2, old-scan math 17.9. hp has no chart model (charts stay figures) and
its OCR is weakest on handwriting and historical print. GPU / cloud VLM parsers score 79–87 on
ParseBench-2 (OCRia-1, Reducto, Datalab, Chandra 2).

What moved these numbers, from the previous build (ParseBench-2 / ParseBench-1 / OmniParseBench /
olmOCR-bench): 59.5 / 51.9 / 70.9 / 60.1 before; D-FINE duplicate boxes dropped (a duplicate Table box
wrote its table twice); scans turned upright (OmniParseBench rotated pages 61 → 78.5); inline styles
(ParseBench formatting 75.3 → 81.2, 36.6 → 53.5); TeX math from the PDF's glyphs (olmOCR arXiv math
18 → 62); table clean-up and header bands (ParseBench tables 57.2 → 60.9, 63.2 → 68.9); detected OCR
lines merged only inside one layout region (BHL CER 0.085 → 0.050, olmOCR-bench 65.0 → 65.9).

Throughput through these harnesses (about one page per request, 4 requests in flight): 18–20 pages/s.
A saturated server parses about 64 pages/s of one-page documents (8 in flight, RTX 4050 laptop), as
fast as the same pages as one PDF: up to 8 documents share the GPU pipeline (`--concurrent-docs`).

### Stage choices

"Before" is hp with XY-cut reading order and geometry-only tables (`--no-ordernet --no-tables`).

### Accuracy

**[olmOCR-bench](https://github.com/allenai/olmocr)**, 140-page stratified subset (835 tests), its
own scorer:

| Category | Before | hp | | Reference systems | |
|---|---|---|---|---|---|
| **Overall** | 49.9 | **55.6** | | OXR-1.0 (4B VLM, 0.05 pages/s) | 86.1 |
| Order | 43.5 | **55.7** | | Indra OCR 1.1.2 (CPU) | 46.9 |
| Multi-column | 61.3 | **77.3** | | LiteParse + OCRia-nano-1 (CPU) | 49.8 |
| Tables | 53.4 | **72.8** | | | |
| Headers / footers removed | 91.1 | **94.6** | | | |
| Long tiny text | **79.3** | 75.0 | | | |
| Old scans | 25.5 | **30.0** | | | |
| arXiv math / old-scan math | 19.8 / 18.9 | 20.7 / 18.9 | | | |

**[OmniDocBench](https://github.com/opendatalab/OmniDocBench)**, 100-page subset (45 Chinese, 44
English, 11 mixed or other) and 100 extra table pages (135 tables):

| Metric | Before | hp |
|---|---|---|
| Reading order, Kendall tau over text blocks | 0.876 | **0.943** |
| Page text edit distance (lower is better) | 0.450 | **0.402** |
| Table TEDS / TEDS-structure (135 tables) | 0.480 / 0.592 | **0.658 / 0.764** |
| Text F1 (bag of words, CJK characters as words) | 0.933 | 0.933 |
| Layout recall / precision at IoU 0.5, class accuracy (D-FINE) | 0.920 / 0.947 / 0.966 | same |

**IDL scans**, 751 pages: word F1 0.894 against AWS Textract (P 0.887, R 0.902).

**Display formulas**, 300 from OmniDocBench, rendered with KaTeX and compared as olmOCR-bench does:
PP-FormulaNet-S 45.0% match (English pages 48.1%, Chinese 33.3%), LaTeX edit distance 0.239.

How the other candidates compared, per stage, on the same pages:

| Stage | Chosen | Alternatives |
|---|---|---|
| Layout (recall / class acc.) | D-FINE 0.920 / 0.966 | Indra 0.561 / 0.805, LayoutNet blocks 0.481 / 0.762 |
| OCR (IDL word F1, OmniDocBench text F1) | PP-OCRv6 0.899 / 0.933 | Indra 0.885 / 0.918, OCRia-nano-1 0.879 / 0.894 |
| Tables (TEDS) | LayoutNet 0.658 | geometry-only grid 0.480, Indra table heads 0.509 |
| Reading order (Kendall tau) | OrderNet 0.943 | XY-cut 0.876 |
| Formulas (render match) | PP-FormulaNet-S 45.0% | Indra math models 13.7–16.3% |

### Throughput

After the GPU optimization pass (November 2026), against the build before it, same machine:

| Set | Pages | Before | **Now** |
|---|---|---|---|
| Server, 300 one-page documents, 8 in flight | 300 | 20.0 pages/s (p50 333 ms) | **~64** (p50 ~124 ms) |
| Mixed documents (ParseBench-1 sample, one PDF) | 302 | 44.2 | **62–66** |
| Scanned pages | 98 | 16.4 | **~24** |
| arXiv papers (formula-heavy) | 100 | 10.3 | **~24.5** |
| Benchmark harnesses (olmOCR-bench, OmniParseBench, BHL) | – | 9–10 | **18–20** |

What did it:
- **Text detection post-processing:** warp-aggregated atomics, 5.7 → 0.25 ms per page. Detection and
  recognition are pipelined across pages.
- **Documents overlapped in one long-lived pipeline** in the server.
- **Formula decoder:** fixed batch 8, length-sorted cohorts with row compaction, asynchronous
  end-of-sequence checks, host-side shapes, and the key-value cache passed only up to the slots in use.
- **Text recognition:** a logits plan with a fused FP16 argmax.
- **D-FINE layout:** built with an all-FP16 backbone, 12.0 → 6.4 ms per page.
- **Threads and streams:** formulas on their own thread, 3 assembly threads, high-priority page
  streams. Server tail latency on mixed loads dropped from p99 13–15 s to 0.5–0.7 s.

The GPU is 95–100% busy during parses. Accuracy is unchanged or better (table above).

Per-page costs on these runs (the stages overlap: workers, GPU thread and assembly thread run
concurrently):

| Stage | Where | Cost |
|---|---|---|
| Text layer + routing | page workers | 0.5–8 ms per page |
| Render | page workers | 14–30 ms per page (80–90 ms for ~2000 px images) |
| Layout (D-FINE) | GPU | ~6.5 ms per page |
| OCR (det + rec) | GPU | ~35 ms per scanned page (det ~7.5, rec ~27) |
| Formulas | formula thread | batches of up to 8, across pages |
| LayoutNet (features + net + decode) | assembly thread | 8–24 ms per page with a table |
| OrderNet (features + net + decode + Markdown) | assembly thread | 0.9–3.2 ms per page |

Model load 1.0–1.1 s from the engine cache (warm-up 0.2 s); reload after an idle unload ~0.6 s.

Network latency in TensorRT, one page, including host-device copies (ms at N = 64 / 256 / 1024
lines or units):

| | FP32 | FP16 (shipped) | INT8 |
|---|---|---|---|
| OrderNet | 0.36 / 0.50 / 3.90 | 0.30 / 0.40 / 2.40 | 0.27 / 0.39 / 2.06 |
| LayoutNet | 1.27 / 1.70 / 9.50 | 0.81 / 1.18 / 6.56 | 0.84 / 1.25 / 7.41 |

FP16 matches FP32 on the benchmarks above; INT8 LayoutNet is slower and loses 1.4 TEDS points
(`tools/quant`).

## Why this stack

| Choice | Why |
|---|---|
| **C++ / CUDA end to end** | No Python on the hot path. Resizing, crop batching and preprocessing run as CUDA kernels; text recognition replays CUDA graphs. |
| **TensorRT** | The fastest inference on NVIDIA GPUs. Engines are built on your machine for your exact GPU from ONNX models on Hugging Face. |
| **Text layer first** | Most born-digital PDFs already contain correct text. Routing each page (native, partial or full OCR) skips OCR wherever it isn't needed, and that is the biggest speed-up. Routing and glyph recovery are ported from [liteparse](https://github.com/run-llama/liteparse). |
| **PDFium** ([run-llama's fork](https://github.com/run-llama/pdfium-binaries)) | Robust rendering and text extraction, plus the per-glyph font data that glyph recovery needs. |
| **D-FINE** (from [SoMark OXR-1.0](https://github.com/SoMarkAI)) | An accurate real-time layout detector: about 14 ms per page at 960² in FP16. |
| **PP-OCRv6 small, PP-FormulaNet-S** ([PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)) | Small, fast OCR and formula models that fit next to the layout model in 6 GB. |
| **OrderNet** (from [Indra OCR](https://huggingface.co/datalab-to/indra-ocr)) | A 0.73M-parameter reading-order model over text lines: pairwise order, block classes and same-block links, FP16 on TensorRT (~0.4 ms for 256 lines). Against XY-cut: olmOCR-bench 52.3 → 55.6 (order tests 43.5 → 55.7, multi-column 60 → 77, tables 68 → 72.8), OmniDocBench reading-order Kendall tau 0.87 → 0.94–0.95, at 1–3 ms per page on the assembly thread. Its decoding and Markdown are ported from indra_ocr and checked against it (`tests/ordernet_parity_test.cpp`). |
| **LayoutNet** (from [OCRia-nano-1](https://huggingface.co/llamaindex/OCRia-nano-1)) | A 4.3M-parameter table-structure model over text units and the page image, run on the pages where D-FINE finds a table. Against geometry-only grids: OmniDocBench TEDS 0.480 → 0.643 (135 tables), olmOCR-bench table tests 53.4 → 68.0, at about 25 ms per table page on the assembly thread. Its features and decoder are ported from liteparse and checked against it page by page (`tests/tablenet_parity.cpp`). |
| **[Drogon](https://github.com/drogonframework/drogon)** | A fast asynchronous C++ HTTP server. |

## Architecture

```
 PDF bytes
    │
    ▼
 page workers: forked processes running PDFium                      CPU
   text layer + glyph recovery ──► route each page: native | partial | full_ocr
   render only the pages that need pixels ──► shared-memory slot ring
    │
    ▼
 GPU pipeline: TensorRT FP16, CUDA streams                           GPU
   D-FINE layout (two lanes) ──┐
   PP-OCR det ──► rec ─────────┼──► assembly: reading order (XY-cut), formulas
   PP-FormulaNet-S ────────────┘    as LaTeX, LayoutNet tables, OrderNet order + blocks, Markdown
    │
    ▼
 per-page results, streamed as they finish
   ├─ hyperparse          JSON lines and/or Markdown files
   └─ hyperparse_server   JSON, Markdown or NDJSON over HTTP
```

- **Page workers are processes, not threads.** PDFium isn't thread-safe, and
  separate processes also isolate crashes: a watchdog times out stuck pages,
  and a failed page doesn't take down the document.
- **Documents overlap.** Concurrent parses (server requests) share one pipeline: one document
  renders while another is on the GPU and a third is assembled, and short documents overtake
  long ones. Errors stay with their document.
- **Models load lazily and can unload when idle,** giving the GPU back between bursts.
- **Model preparation runs in a child process.** It downloads from Hugging Face,
  checks SHA-256 and builds the engines, so the main process stays free of CUDA
  until it has forked its workers.

## Quick start

Linux, an NVIDIA GPU (Turing or newer), CUDA 13 and TensorRT 11. The full guide,
including the TensorRT options and every path setting, is in
[BUILDING.md](BUILDING.md).

```sh
uv sync --project trt-env                     # TensorRT 11.5 libraries (or use a system install)
cmake -S . -B build -G Ninja                  # first run fetches PDFium, Drogon and headers into build/_deps
ninja -C build
cmake --install build                         # dist/ + symlinks in ~/.local/bin

hyperparse --prepare-models                   # first run: download models, build engines (~12 min)
hyperparse doc.pdf --md doc.md
hyperparse_server --port 8080                 # then: curl localhost:8080/docs
```

The models are on Hugging Face at `logxdx/hyperparse-models`, which is private
for now. Its `tablenet.onnx` is OCRia-nano-1's LayoutNet, mirrored unmodified with its license.

## Layout

```
src/          the engine: page pipeline, models, text layer, tables, model store
src/core/     building blocks: TensorRT wrapper, OCR kernels and recognition,
              reading order, PDF text layer, Markdown export
server/       hyperparse_server (Drogon routes, queue, metrics, API docs)
tools/        Drogon build script, evaluation scripts
tests/        engine and HTTP end-to-end tests
trt-env/      optional pip-based TensorRT for building
```

## Acknowledgements and licenses

hyperparse stands on [liteparse](https://github.com/run-llama/liteparse),
SoMark OXR, PaddleOCR, PDFium and Drogon. Code under other licenses keeps them
(see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), and the models keep
theirs, listed with them on Hugging Face.
