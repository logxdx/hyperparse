# Building hyperparse

hyperparse is built on the machine that runs it. It targets that machine's CPU
(`-march=native`) and GPU (`CMAKE_CUDA_ARCHITECTURES=native`). Installing it
gives one folder, `dist/`, holding the binaries and the TensorRT and PDFium
libraries they need. `hyperparse` and `hyperparse_server` are symlinked onto
your `PATH`. Models come from the Hugging Face cache, and TensorRT engines are
built on first use.

## Requirements

- Linux, x86_64 or aarch64
- NVIDIA GPU, Turing (RTX 20xx, T4) or newer, with driver R580 or newer
- CUDA Toolkit 13.x (`nvcc`) and a host compiler it supports
- TensorRT 11.x (see below)
- CMake 3.24+, Ninja, git, curl, libcurl headers, and network access on the
  first configure
- for `hyperparse_server`: zlib and libuuid headers

On Debian/Ubuntu:
`apt install cmake ninja-build git curl libcurl4-openssl-dev zlib1g-dev uuid-dev`.
CUDA comes from NVIDIA's repo or the runfile.

## 1. TensorRT

Any of these works. TensorRT is only needed while building: the install
copies the libraries hyperparse needs into `dist/lib`.

| Install | Configure with |
|---|---|
| pip wheel via [uv](https://docs.astral.sh/uv/): `uv sync --project trt-env` (1.6 GB; `./trt-env/prune_trt.sh` trims it) | nothing: found in `trt-env/.venv` |
| NVIDIA tarball, extracted to e.g. `/opt/TensorRT-11.5.0` | `-DTensorRT_ROOT=/opt/TensorRT-11.5.0` or env `TensorRT_ROOT` |
| NVIDIA apt/dnf packages (`libnvinfer-dev`, `libnvonnxparsers-dev`) | nothing: found in the system paths |

The pip wheel has no C++ headers; CMake then fetches the matching ones from
NVIDIA's TensorRT repo. It prints the library and header version it uses:
`-- TensorRT: .../libnvinfer.so.11 (headers 11.5.0 in ...)`.

## 2. Build and install

```sh
cmake -S . -B build -G Ninja [-DTensorRT_ROOT=<dir>]
ninja -C build
cmake --install build
```

The first configure fetches what the build needs into `build/_deps/`, checking
each download's SHA-256: run-llama's PDFium build (chromium/8058), the
nlohmann/json header, and, if TensorRT came from the wheel, its headers. It
also builds Drogon and jsoncpp as static libraries with `tools/build_drogon.sh`
(about a minute; `-DHP_SERVER=OFF` skips the server and Drogon). Nothing is
installed system-wide.

The install writes `dist/` (about 1 GB, mostly TensorRT). Pass `--prefix <dir>`
to install elsewhere:

```
bin/hyperparse, bin/hyperparse_server   find lib/ relative to themselves ($ORIGIN/../lib)
lib/libnvinfer.so.11                    TensorRT runtime and builder
lib/libnvonnxparser.so.11               ONNX parser, for building engines
lib/libnvinfer_builder_resource_sm<cc>.so.11.*   builder resource for this GPU
lib/libpdfium.so
share/licenses/
```

It also symlinks `hyperparse` and `hyperparse_server` into `~/.local/bin`, so
they run from any directory. `-DHP_LINK_DIR=<dir>` links somewhere else; an
empty value means no links. A real file is never replaced. If you move
`dist/`, re-run the install so the links follow.

The CUDA runtime is linked into the binaries; the NVIDIA driver provides the
rest. Beyond `dist/`, the binaries use only system libraries: libc and
libstdc++, plus zlib and libuuid for the server, and libcurl for the first
model download. The binaries in `build/` work too, but they load the libraries
from wherever the build found them.

### Paths

Every build input can be set with a CMake option or an environment variable.
The option wins over the variable.

| What | CMake option | Environment variable | Default |
|---|---|---|---|
| CUDA Toolkit | `-DCMAKE_CUDA_COMPILER=<dir>/bin/nvcc` | `CUDA_HOME` or `CUDA_PATH` (`<dir>`), `CUDACXX` (nvcc itself) | `nvcc` on `PATH`, else `/usr/local/cuda` and similar |
| TensorRT | `-DTensorRT_ROOT=<dir>` | `TensorRT_ROOT` or `TENSORRT_ROOT` | system paths, then `trt-env/.venv` |
| PDFium | `-DPDFIUM_DIR=<dir>` | `PDFIUM_DIR` | fetched into `build/_deps/pdfium` |
| Drogon | `-DDrogon_ROOT=<dir>` | `Drogon_ROOT`, also where `build_drogon.sh` installs | built into `build/_deps/drogon` |
| GPU for the builder resource | `-DHP_GPU_SM=89` | | detected with `nvidia-smi` |

CMake caches these on the first configure. To switch to a different CUDA
Toolkit, use a new build directory. For the others, pass the `-D` option again.

## 3. Models

```sh
hf auth login                  # the model repo is private for now
hyperparse --prepare-models
```

`--prepare-models` (or the first parse or server start) does two things:

1. **Download:** it downloads the ONNX models of `logxdx/hyperparse-models`
   into the standard Hugging Face cache, in the same layout as `hf download`,
   and checks each file's SHA-256 against `manifest.json`. A copy already in
   the cache from `hf download logxdx/hyperparse-models` is used as is.
2. **Build engines:** it builds the TensorRT engines into
   `~/.cache/hyperparse/engines/<manifest hash>-sm<cc>-trt<version>/`. This
   happens once per set of models, GPU and TensorRT version, and takes about
   12 minutes. Repo changes that don't touch the models (such as the model
   card) reuse the engines.

| Environment variable | Effect |
|---|---|
| `HF_TOKEN`, `HF_TOKEN_PATH` | Token for the private repo (default: the `hf auth login` token) |
| `HF_HUB_CACHE`, `HF_HOME` | Hugging Face cache location (default `~/.cache/huggingface/hub`) |
| `HF_HUB_OFFLINE=1` | Never download; fail if the models are not cached |
| `HP_MODEL_REVISION` | Branch, tag or commit of the model repo (default `main`). A cached copy is used without checking for updates; `hf download logxdx/hyperparse-models` updates it. |
| `HP_CACHE_DIR` | Engine cache (default `$XDG_CACHE_HOME/hyperparse`, else `~/.cache/hyperparse`) |
| `HP_MODELS`, `--models <dir>` | Use this folder instead of the caches. It holds the plans, or the ONNX to build them from (built in place). |

## 4. Run

```sh
hyperparse doc.pdf --out pages.jsonl --md doc.md     # options: hyperparse --help
hyperparse_server --port 8080                        # options and endpoints: hyperparse_server --help
curl localhost:8080/docs                             # full API docs, from the server itself
curl --data-binary @doc.pdf 'localhost:8080/parse?format=markdown'
```

## Tests

```sh
python3 tests/server_test.py --data <dir> --small a.pdf --medium b.pdf --long c.pdf --big d.pdf   # HTTP API end to end (see its --help)
build/engine_lifecycle_test lifecycle <doc.pdf>              # load/unload/idle-unload cycles
```

## When something changes

- **New TensorRT, CUDA Toolkit or GPU:** rebuild and reinstall (step 2). The
  engine cache key includes the GPU and TensorRT version, so the engines
  rebuild automatically on the next start.
- **New models on Hugging Face:** run `hf download logxdx/hyperparse-models` to
  update the cached copy. The engines then rebuild on the next start.
- **Driver update:** nothing to do.
