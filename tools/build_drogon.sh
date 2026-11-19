#!/usr/bin/env bash
# Builds Drogon v1.9.12 (+ its Trantor submodule) and jsoncpp from source as
# static libraries into $Drogon_ROOT (default build/_deps/drogon, where
# CMakeLists.txt looks; the first configure runs this script itself). No sudo,
# nothing installed system-wide. ORM, drogon_ctl and examples are off.
# TRANTOR_USE_TLS=none: Trantor v1.5.x's OpenSSLProvider.cc fails to compile
# against OpenSSL 4.0 (X509_get_subject_name now returns const, and the build
# uses -Werror on deprecations). hyperparse_server speaks plain HTTP; put TLS in
# a reverse proxy. c-ares (async DNS) and brotli (response compression) are left
# out: the server needs neither, so it links only zlib and libuuid from the system.
#
#   tools/build_drogon.sh [work_dir]
set -euo pipefail

HP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${Drogon_ROOT:-$HP_DIR/build/_deps/drogon}"
WORK="${1:-$(mktemp -d)}"
JOBS="$(nproc)"
mkdir -p "$WORK" "$PREFIX"
cd "$WORK"

[ -d jsoncpp ] || git clone --depth 1 --branch 1.9.6 https://github.com/open-source-parsers/jsoncpp.git
cmake -S jsoncpp -B jsoncpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DBUILD_OBJECT_LIBS=OFF -DJSONCPP_WITH_TESTS=OFF \
  -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF -DJSONCPP_WITH_EXAMPLE=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
ninja -C jsoncpp/build -j"$JOBS" install

if [ ! -d drogon ]; then
  git clone --depth 1 --branch v1.9.12 https://github.com/drogonframework/drogon.git
  git -C drogon submodule update --init --depth 1
fi
cmake -S drogon -B drogon/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="$PREFIX" -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_ORM=OFF -DBUILD_POSTGRESQL=OFF -DBUILD_MYSQL=OFF \
  -DBUILD_SQLITE=OFF -DBUILD_REDIS=OFF -DBUILD_TESTING=OFF -DBUILD_YAML_CONFIG=OFF -DTRANTOR_USE_TLS=none \
  -DCMAKE_DISABLE_FIND_PACKAGE_c-ares=ON -DCMAKE_DISABLE_FIND_PACKAGE_Brotli=ON
ninja -C drogon/build -j"$JOBS" install
echo "Drogon installed to $PREFIX"
