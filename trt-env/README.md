# trt-env

TensorRT 11.5 shared libraries from NVIDIA's pip index, for building
hyperparse on a machine without a system TensorRT install. CMake finds them in
`trt-env/.venv` automatically.

```sh
uv sync --project trt-env     # ~1.6 GB: libnvinfer, ONNX parser, builder resources for every GPU
./trt-env/prune_trt.sh        # optional: keep only this GPU's builder resource
```

The wheel has no C++ headers; see BUILDING.md for the matching header clone.
After `cmake --install`, the libraries hyperparse needs are copied into
`dist/lib`, so this environment can be deleted.
