#!/usr/bin/env bash
# Delete TensorRT builder resources for GPUs other than this one (and the Windows
# cross-build copies). Keeps the local SM and the PTX fallback. Re-run after `uv sync`
# reinstalls tensorrt-cu13-libs; `uv sync --reinstall-package tensorrt-cu13-libs` restores all.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
libs="$(ls -d "$here"/.venv/lib/python3*/site-packages/tensorrt_libs)"
sm="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
sm="${sm/_/}"   # sm_89 -> sm89, matching libnvinfer_builder_resource_sm89.so.*
before=$(du -sm "$libs" | cut -f1)
for f in "$libs"/libnvinfer_builder_resource_*.so.*; do
  base="$(basename "$f")"
  case "$base" in
    libnvinfer_builder_resource_${sm}.so.*|libnvinfer_builder_resource_ptx.so.*) ;;  # keep
    *) rm -f "$f" ;;
  esac
done
echo "tensorrt_libs: ${before} MB -> $(du -sm "$libs" | cut -f1) MB (kept ${sm} + ptx)"
