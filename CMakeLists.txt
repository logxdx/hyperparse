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

find_package(CURL REQUIRED)  # headers only: libcurl is dlopen'ed for model downloads

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
  src/orient.cpp
  src/style_md.cpp
  src/text_det.cpp
  src/pdf_render.cpp
  src/render_pool.cpp
  src/crop_batch.cu
  src/text_layer.cpp
  src/assemble.cpp
  src/table_grid.cpp
  src/tablenet/text.cpp
  src/tablenet/decode.cpp
  src/tablenet/units.cpp
  src/tablenet/features.cpp
  src/tablenet/tablenet.cpp
  src/tablenet/page.cpp
  src/ordernet/order.cpp
  src/ordernet/text.cpp
  src/ordernet/net.cpp
  src/ordernet/page.cpp
  src/glyph_decoder.cpp
  src/tex_math/tex_math.cpp
  src/tex_math/encodings.cpp
  src/tex_math/structures.cpp
  src/tex_math/collect.cpp
  src/formula.cu
  src/rec_batcher.cpp
  src/text_recognizer.cpp
  src/model_set.cpp
  src/ocr_stage.cpp
  src/page_pipeline.cpp
  src/page_json.cpp
  src/engine_build.cpp
  src/model_store.cpp
  src/engine.cpp)
target_include_directories(hp_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
# LayoutNet's features and decoder must round like the Rust reference (tests/tablenet_parity):
# no fused multiply-adds.
set_source_files_properties(src/tablenet/text.cpp src/tablenet/decode.cpp src/tablenet/units.cpp
  src/tablenet/features.cpp src/tablenet/page.cpp src/ordernet/order.cpp src/ordernet/text.cpp
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# The TeX math pass (LiteParse tex_math port) compares f32 sums against thresholds like the Rust reference.
set_source_files_properties(src/tex_math/tex_math.cpp src/tex_math/structures.cpp src/tex_math/collect.cpp
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_include_directories(hp_core SYSTEM PUBLIC ${TRT_INCLUDE_DIR} ${PDFIUM_DIR}/include ${JSON_INCLUDE_DIR})
target_include_directories(hp_core SYSTEM PRIVATE ${CURL_INCLUDE_DIRS})
target_link_libraries(hp_core PUBLIC CUDA::cudart_static ${NVINFER} ${NVONNXPARSER} ${PDFIUM_DIR}/lib/libpdfium.so ${CMAKE_DL_LIBS})
target_compile_options(hp_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

add_executable(hyperparse src/hyperparse.cpp)
target_link_libraries(hyperparse PRIVATE hp_core)
target_compile_options(hyperparse PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(hyperparse PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")

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

add_executable(tablenet_decode_test tests/tablenet_decode_test.cpp)
target_link_libraries(tablenet_decode_test PRIVATE hp_core)
target_compile_options(tablenet_decode_test PRIVATE -O2 -Wall -Wextra)
set_target_properties(tablenet_decode_test PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")

add_executable(ordernet_parity_test tests/ordernet_parity_test.cpp)
target_link_libraries(ordernet_parity_test PRIVATE hp_core)
target_compile_options(ordernet_parity_test PRIVATE -O2 -Wall -Wextra)
set_target_properties(ordernet_parity_test PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")

add_executable(ordernet_trt_check tests/ordernet_trt_check.cpp)
target_link_libraries(ordernet_trt_check PRIVATE hp_core)
set_target_properties(ordernet_trt_check PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")

add_executable(tablenet_parity tests/tablenet_parity.cpp)
target_link_libraries(tablenet_parity PRIVATE hp_core)
target_compile_options(tablenet_parity PRIVATE -O2 -Wall -Wextra)
set_target_properties(tablenet_parity PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")

add_executable(tex_math_parity tests/tex_math_parity.cpp)
target_link_libraries(tex_math_parity PRIVATE hp_core)
target_compile_options(tex_math_parity PRIVATE -O2 -Wall -Wextra)
set_target_properties(tex_math_parity PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")

add_executable(engine_lifecycle_test tests/engine_lifecycle_test.cpp)
target_link_libraries(engine_lifecycle_test PRIVATE hp_core)
set_target_properties(engine_lifecycle_test PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")

# HTTP server over Drogon 1.9.12 (static, with jsoncpp): -DDrogon_ROOT= or env
# Drogon_ROOT points at an install; otherwise tools/build_drogon.sh builds it into
# build/_deps/drogon on the first configure (~1 min). -DHP_SERVER=OFF skips it.
option(HP_SERVER "build hyperparse_server" ON)
if(HP_SERVER)
  list(APPEND CMAKE_PREFIX_PATH ${HP_DEPS}/drogon)
  find_package(Drogon CONFIG QUIET)
  if(NOT Drogon_FOUND)
    message(STATUS "Building Drogon into ${HP_DEPS}/drogon (log: ${HP_DEPS}/drogon-build.log)")
    execute_process(COMMAND ${CMAKE_COMMAND} -E env Drogon_ROOT=${HP_DEPS}/drogon
                            bash ${CMAKE_SOURCE_DIR}/tools/build_drogon.sh ${HP_DEPS}/drogon-src
                    OUTPUT_FILE ${HP_DEPS}/drogon-build.log ERROR_FILE ${HP_DEPS}/drogon-build.log RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "building Drogon failed; see ${HP_DEPS}/drogon-build.log")
    endif()
    find_package(Drogon CONFIG REQUIRED)
  endif()
endif()
if(Drogon_FOUND)
  add_executable(hyperparse_server
    server/main.cpp
    server/server_config.cpp
    server/bootstrap.cpp
    server/request.cpp
    server/metrics.cpp
    server/parse_routes.cpp
    server/model_routes.cpp
    server/admin_routes.cpp
    server/api_docs.cpp)
  target_link_libraries(hyperparse_server PRIVATE hp_core Drogon::Drogon)
  target_compile_options(hyperparse_server PRIVATE -O3 -march=native -Wall -Wextra)
  set_target_properties(hyperparse_server PROPERTIES BUILD_RPATH "${TRT_LIB_DIR};${PDFIUM_DIR}/lib")
endif()

# Self-contained install: `cmake --install build` (prefix dist unless --prefix) gives
#   <dir>/bin/hyperparse, hyperparse_server   (RUNPATH $ORIGIN/../lib: nothing absolute)
#   <dir>/lib/                      the TensorRT, ONNX-parser, builder-resource (this GPU) and PDFium libraries
# Models come from the Hugging Face cache and the engine cache (model_store.h).
set(_hp_bins hyperparse)
if(TARGET hyperparse_server)
  list(APPEND _hp_bins hyperparse_server)
endif()
set_target_properties(${_hp_bins} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS ${_hp_bins} RUNTIME DESTINATION bin)
# Each library under its SONAME, resolving the symlink chains of system installs.
foreach(_lib ${NVINFER} ${NVONNXPARSER} ${PDFIUM_DIR}/lib/libpdfium.so)
  file(REAL_PATH ${_lib} _real)
  execute_process(COMMAND readelf -d ${_real} OUTPUT_VARIABLE _dyn)
  string(REGEX MATCH "Library soname: \\[([^]]+)\\]" _m "${_dyn}")
  set(_soname ${CMAKE_MATCH_1})
  if(NOT _soname)
    get_filename_component(_soname ${_lib} NAME)
  endif()
  install(FILES ${_real} DESTINATION lib RENAME ${_soname})
endforeach()
# libnvinfer dlopens its builder resource from its own directory when building
# engines: ship the one for this GPU (HP_GPU_SM, detected). The 241 MB PTX
# resource is only the fallback when that is missing (engines built without it
# measured identical: IDL751 F1 and pages/s).
set(HP_GPU_SM "" CACHE STRING "GPU compute capability for the builder resource, e.g. 89 (empty = detect)")
if(NOT HP_GPU_SM)
  execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                  OUTPUT_VARIABLE _cc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  string(REGEX MATCH "^[0-9]+\\.[0-9]+" _cc "${_cc}")
  string(REPLACE "." "" HP_GPU_SM "${_cc}")
endif()
set(_trt_res)
if(HP_GPU_SM)
  file(GLOB _trt_res ${TRT_LIB_DIR}/libnvinfer_builder_resource_sm${HP_GPU_SM}.so.*)
endif()
if(NOT _trt_res)
  file(GLOB _trt_res ${TRT_LIB_DIR}/libnvinfer_builder_resource_ptx.so.*)
  message(WARNING "no TensorRT builder resource for sm${HP_GPU_SM} in ${TRT_LIB_DIR}; installing the PTX fallback: ${_trt_res}")
endif()
install(FILES ${_trt_res} DESTINATION lib)
install(FILES ${PDFIUM_DIR}/LICENSE DESTINATION share/licenses/pdfium)
# Symlinks into a directory on PATH (empty = none). $ORIGIN resolves through the
# link to <prefix>/bin, so the libraries are still found. Real files are never replaced.
set(HP_LINK_DIR "$ENV{HOME}/.local/bin" CACHE PATH "where `cmake --install` links hyperparse/hyperparse_server (empty = no links)")
if(HP_LINK_DIR)
  install(CODE "
    foreach(_b ${_hp_bins})
      set(_link \"${HP_LINK_DIR}/\${_b}\")
      if(EXISTS \"\${_link}\" AND NOT IS_SYMLINK \"\${_link}\")
        message(WARNING \"not linking \${_link}: a real file is there\")
      else()
        file(MAKE_DIRECTORY \"${HP_LINK_DIR}\")
        file(CREATE_LINK \"\${CMAKE_INSTALL_PREFIX}/bin/\${_b}\" \"\${_link}\" SYMBOLIC)
        message(STATUS \"Linked: \${_link} -> \${CMAKE_INSTALL_PREFIX}/bin/\${_b}\")
      endif()
    endforeach()")
endif()
