cmake_minimum_required(VERSION 3.24)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
# CUDA: the toolkit installed on this machine (nothing bundled). nvcc comes from
# -DCMAKE_CUDA_COMPILER, else env CUDACXX, else env CUDA_HOME or CUDA_PATH
# (<dir>/bin/nvcc), else PATH. It is cached: a different toolkit needs a fresh build dir.
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  foreach(_v CUDA_HOME CUDA_PATH)
    if(DEFINED ENV{${_v}} AND EXISTS "$ENV{${_v}}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "$ENV{${_v}}/bin/nvcc")
      break()
    endif()
  endforeach()
endif()
project(hyperparse LANGUAGES CXX CUDA)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/dist CACHE PATH "install folder: binaries + their libraries" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Source paths in __FILE__ (error messages) relative to the repo, not this machine.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}/=>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${CMAKE_SOURCE_DIR}/=>)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA: ${CUDAToolkit_VERSION} in ${CUDAToolkit_BIN_DIR}")

# Third-party dependencies are found where they are installed, or else fetched
# into the build tree (build/_deps) on the first configure. Nothing lives in the
# source tree.
set(HP_DEPS ${CMAKE_BINARY_DIR}/_deps)

# TensorRT: -DTensorRT_ROOT=<tarball dir> (or env TensorRT_ROOT / TENSORRT_ROOT),
# else the system paths (apt/dnf packages), else the pip wheel in trt-env/.venv
# (uv sync --project trt-env). The wheel has no C++ headers: those then come
# from NVIDIA's TensorRT OSS repo at the matching tag.
set(_trt_root "$ENV{TensorRT_ROOT}")
if(NOT _trt_root)
  set(_trt_root "$ENV{TENSORRT_ROOT}")
endif()
set(TensorRT_ROOT "${_trt_root}" CACHE PATH "TensorRT install dir (holds include/ and lib/)")
file(GLOB _trt_wheel_dirs ${CMAKE_SOURCE_DIR}/trt-env/.venv/lib/python3*/site-packages/tensorrt_libs)
find_library(NVINFER NAMES nvinfer libnvinfer.so.11
  HINTS ${TensorRT_ROOT}/lib ${TensorRT_ROOT}/lib64 PATHS /usr/local/TensorRT/lib ${_trt_wheel_dirs})
if(NOT NVINFER)
  message(FATAL_ERROR "TensorRT not found: run `uv sync --project trt-env`, or install it and pass "
    "-DTensorRT_ROOT=<dir>; see BUILDING.md.")
endif()
get_filename_component(TRT_LIB_DIR ${NVINFER} DIRECTORY)
find_path(TRT_INCLUDE_DIR NvInfer.h
  HINTS ${TensorRT_ROOT}/include PATHS /usr/local/TensorRT/include ${HP_DEPS}/tensorrt-headers/include)
if(NOT TRT_INCLUDE_DIR)
  # Headers for the wheel: the OSS tag matching the library (from the builder
  # resource's file name, e.g. ...so.11.5.0 -> v11.5).
  file(GLOB _trt_res_any ${TRT_LIB_DIR}/libnvinfer_builder_resource*.so.*)
  set(_tag v11.5)
  if(_trt_res_any)
    list(GET _trt_res_any 0 _r)
    if(_r MATCHES "\\.so\\.([0-9]+)\\.([0-9]+)")
      set(_tag v${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
    endif()
  endif()
  message(STATUS "Fetching TensorRT ${_tag} headers into ${HP_DEPS}/tensorrt-headers")
  file(REMOVE_RECURSE ${HP_DEPS}/tensorrt-headers)
  execute_process(COMMAND git clone -q --depth 1 --filter=blob:none --sparse --branch ${_tag}
                          https://github.com/NVIDIA/TensorRT ${HP_DEPS}/tensorrt-headers RESULT_VARIABLE _rc)
  if(_rc EQUAL 0)
    execute_process(COMMAND git -C ${HP_DEPS}/tensorrt-headers sparse-checkout set include RESULT_VARIABLE _rc)
  endif()
  unset(TRT_INCLUDE_DIR CACHE)
  find_path(TRT_INCLUDE_DIR NvInfer.h PATHS ${HP_DEPS}/tensorrt-headers/include NO_DEFAULT_PATH)
  if(NOT TRT_INCLUDE_DIR)
    message(FATAL_ERROR "no TensorRT headers: fetching NVIDIA/TensorRT ${_tag} failed")
  endif()
endif()
find_library(NVONNXPARSER NAMES nvonnxparser libnvonnxparser.so.11 HINTS ${TRT_LIB_DIR} NO_DEFAULT_PATH)
if(NOT NVONNXPARSER)
  message(FATAL_ERROR "libnvonnxparser not found next to ${NVINFER}")
endif()
file(STRINGS ${TRT_INCLUDE_DIR}/NvInferVersion.h _trt_ver REGEX "define (TRT_(MAJOR|MINOR|PATCH)_ENTERPRISE|NV_TENSORRT_(MAJOR|MINOR|PATCH)) +[0-9]")
string(REGEX REPLACE "[^;]* ([0-9]+)" "\\1" _trt_ver "${_trt_ver}")
string(REPLACE ";" "." _trt_ver "${_trt_ver}")
message(STATUS "TensorRT: ${NVINFER} (headers ${_trt_ver} in ${TRT_INCLUDE_DIR})")

# PDFium: run-llama's fork (chromium/8058): the stock API plus per-char char codes
# and the font glyph-name/encoding/ToUnicode queries that glyph recovery uses.
# -DPDFIUM_DIR= or env PDFIUM_DIR points at an extracted release.
set(_pdfium_dir ${HP_DEPS}/pdfium)
if(DEFINED ENV{PDFIUM_DIR})
  set(_pdfium_dir "$ENV{PDFIUM_DIR}")
endif()
set(PDFIUM_DIR ${_pdfium_dir} CACHE PATH "PDFium release (include/, lib/libpdfium.so)")
if(NOT EXISTS ${PDFIUM_DIR}/lib/libpdfium.so)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_asset pdfium-linux-arm64 4915b5351640c1eea59e26023e0edcad8433df96dc8f4cb3d645a4d64a178b7c)
  else()
    set(_asset pdfium-linux-x64 2dd418a8974dbb373ad281f99b6f95ed02e01b1a2bb2e05e772c80b7587577b5)
  endif()
  list(GET _asset 0 _name)
  list(GET _asset 1 _sha)
  message(STATUS "Fetching PDFium (${_name}) into ${PDFIUM_DIR}")
  file(DOWNLOAD https://github.com/run-llama/pdfium-binaries/releases/download/chromium%2F8058/${_name}.tgz
       ${HP_DEPS}/${_name}.tgz EXPECTED_HASH SHA256=${_sha} TLS_VERIFY ON)
  file(ARCHIVE_EXTRACT INPUT ${HP_DEPS}/${_name}.tgz DESTINATION ${PDFIUM_DIR})
endif()

# nlohmann/json, single header.
set(JSON_INCLUDE_DIR ${HP_DEPS}/nlohmann/include)
if(NOT EXISTS ${JSON_INCLUDE_DIR}/nlohmann/json.hpp)
  file(DOWNLOAD https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp
       ${JSON_INCLUDE_DIR}/nlohmann/json.hpp
       EXPECTED_HASH SHA256=aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63 TLS_VERIFY ON)
endif()

# The engine: src/ (hyperparse) and src/core/ (TensorRT wrapper, OCR kernels and
# recognition, layout reading order, PDF text layer, Markdown export).
file(GLOB_RECURSE HP_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/core/*.cpp ${CMAKE_SOURCE_DIR}/src/core/*.cu)
set_source_files_properties(${HP_CORE_SOURCES} PROPERTIES COMPILE_OPTIONS "-w")
add_library(hp_core STATIC
  ${HP_CORE_SOURCES}
  src/multi_resize.cu
  src/dfine_layout.cpp
  src/text_det.cpp
  src/pdf_render.cpp
  src/render_pool.cpp
  src/crop_batch.cu
  src/text_layer.cpp
  src/glyph_decoder.cpp
  src/formula.cu
  src/rec_batcher.cpp
  src/text_recognizer.cpp
  src/model_set.cpp)
target_include_directories(hp_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(hp_core SYSTEM PUBLIC ${TRT_INCLUDE_DIR} ${PDFIUM_DIR}/include ${JSON_INCLUDE_DIR})
target_link_libraries(hp_core PUBLIC CUDA::cudart_static ${NVINFER} ${NVONNXPARSER} ${PDFIUM_DIR}/lib/libpdfium.so ${CMAKE_DL_LIBS})
target_compile_options(hp_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

add_executable(crop_batch_test tests/crop_batch_test.cu)
target_link_libraries(crop_batch_test PRIVATE hp_core)
target_compile_options(crop_batch_test PRIVATE -O3 --use_fast_math)
set_target_properties(crop_batch_test PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")

add_executable(pdf_chars tests/pdf_chars.cpp)
target_link_libraries(pdf_chars PRIVATE hp_core)
set_target_properties(pdf_chars PROPERTIES BUILD_RPATH "${PDFIUM_DIR}/lib")

add_executable(formula_golden_test tests/formula_golden_test.cpp)
target_link_libraries(formula_golden_test PRIVATE hp_core)
set_target_properties(formula_golden_test PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")

add_executable(render_pool_test tests/render_pool_test.cpp)
target_link_libraries(render_pool_test PRIVATE hp_core)
set_target_properties(render_pool_test PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")

