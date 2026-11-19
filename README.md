# hyperparse

A GPU PDF parser in C++/CUDA. It turns PDFs into layout regions, text lines,
LaTeX formulas and Markdown, at about 16 pages per second on a 6 GB laptop GPU.

## Why

Built for my GPU-poor self, so I can convince myself I have a professional-grade
document parsing pipeline at home. 😢

Hosted document-parsing APIs are pricey, and the open-source pipelines usually
assume a big GPU and a Python stack. hyperparse is what's left once you
squeeze that onto one consumer GPU: TensorRT everywhere, OCR only where the PDF
doesn't already have good text, and no Python on the hot path.

## What you get

- **Layout**: 12 region classes per page (text, title, table, picture, formula, caption, code, stamp, …) with reading order
- **Text**: the PDF's own text layer when it is usable, OCR where it isn't (per page or per region)
- **Formulas**: display formulas as LaTeX
- **Tables**: table regions as Markdown pipe tables, built from the geometry of their text
- **Markdown**: per page and for the whole document
- **CLI**: `hyperparse doc.pdf --md doc.md --out pages.jsonl`
- **HTTP server**: `hyperparse_server`, with a queue, streaming (NDJSON), Prometheus metrics,
  health checks and its API docs at `GET /docs`

Measured on an RTX 4050 Laptop GPU (6 GB):

| Set | Pages | Throughput | Notes |
|---|---|---|---|
| IDL scans ([pixparse/idl-wds](https://huggingface.co/datasets/pixparse/idl-wds)) | 751 | 16.5 pages/s | word F1 0.894 against AWS Textract's OCR of the same pages |
| Scientific papers | 60 | 13.7 pages/s | 98 formulas |
| Mixed documents | 100 | 12.2 pages/s | 118 formulas |

With the models loaded it uses about 3 GB of GPU memory. They unload after an
idle timeout and reload on the next request in about 0.6 s.

## Why this stack

| Choice | Why |
|---|---|
| **C++ / CUDA end to end** | No Python on the hot path. Resizing, crop batching and preprocessing run as CUDA kernels; text recognition replays CUDA graphs. |
| **TensorRT** | The fastest inference on NVIDIA GPUs. Engines are built on your machine for your exact GPU from ONNX models on Hugging Face. |
| **Text layer first** | Most born-digital PDFs already contain correct text. Routing each page (native, partial or full OCR) skips OCR wherever it isn't needed, and that is the biggest speed-up. Routing and glyph recovery are ported from [liteparse](https://github.com/run-llama/liteparse). |
| **PDFium** ([run-llama's fork](https://github.com/run-llama/pdfium-binaries)) | Robust rendering and text extraction, plus the per-glyph font data that glyph recovery needs. |
| **D-FINE** (from [SoMark OXR-1.0](https://github.com/SoMarkAI)) | An accurate real-time layout detector: about 14 ms per page at 960² in FP16. |
| **PP-OCRv6 small, PP-FormulaNet-S** ([PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)) | Small, fast OCR and formula models that fit next to the layout model in 6 GB. |
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
   PP-FormulaNet-S ────────────┘    as LaTeX, table grids, Markdown
    │
    ▼
 per-page results, streamed as they finish
   ├─ hyperparse          JSON lines and/or Markdown files
   └─ hyperparse_server   JSON, Markdown or NDJSON over HTTP
```

- **Page workers are processes, not threads.** PDFium isn't thread-safe, and
  separate processes also isolate crashes: a watchdog times out stuck pages,
  and a failed page doesn't take down the document.
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
for now.

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
