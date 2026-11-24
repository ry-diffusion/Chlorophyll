#!/bin/bash
# SPDX-License-Identifier: MIT
# libm2v.dylib: metal2vulkan's Metal device on Chlorophyll's Turnip, for the inject's per-process trial (m2v/m2v.mm).
#   GPU_METAL_WORK=... BERGAMOTA_GPU=.../gpu m2v/build.sh
# Sources, fetched into $GPU_METAL_WORK/m2v (never stored here):
#   guac/      coreycusick/ios-guacamole (ios27): userland/metal2 (frontend + bridge, MIT) and tools/build-m2v-bridge.sh
#   metal2vulkan/  steelbrain/metal2vulkan 720ef1ce + guac's userland/metal2/bridge/patches (GPL-2.0+ OR LGPL-3.0+)
# then m2v/patches/*.patch applied to guac (the dma-buf IOSurface import). The bridge is built for arm64e with
# -Zbuild-std (RUSTC_BOOTSTRAP: rust 1.98 has the target, not its prebuilt std); everything PAC-stripped (Kryo 385).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BG_GPU="${BERGAMOTA_GPU:-$HERE/..}"
WORK="${GPU_METAL_WORK:?set GPU_METAL_WORK}"
M="$WORK/m2v"
G="$M/guac"
SRC="$M/metal2vulkan"
OUT="${1:-$WORK/out-m2v}"
export PATH=/opt/homebrew/opt/rustup/bin:$PATH
if [ ! -d "$G/userland/metal2" ]; then
  git clone -q --depth 1 --branch ios27 --filter=blob:none --sparse https://github.com/coreycusick/ios-guacamole.git "$G"
  git -C "$G" sparse-checkout set --no-cone /userland/metal2/ /tools/build-m2v-bridge.sh
  for p in "$HERE"/m2v/patches/0*.patch; do git -C "$G" apply "$p"; done
fi
if [ ! -d "$SRC/crates" ]; then
  git init -q "$SRC"
  git -C "$SRC" fetch -q --depth 1 https://github.com/steelbrain/metal2vulkan.git 720ef1ce9ae2a2068999a8675717418b9416283e
  git -C "$SRC" checkout -q FETCH_HEAD
  for p in "$G"/userland/metal2/bridge/patches/*.patch "$HERE"/m2v/patches/metal2vulkan/*.patch; do git -C "$SRC" apply "$p"; done
fi
B="$M/m2v-bridge"
mkdir -p "$B" "$OUT"
LLVM_AS=$(ls /opt/homebrew/opt/llvm*/bin/llvm-as | head -1)
"$LLVM_AS" -o "$B/runtime_examples.air" "$SRC/crates/metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll"
"$LLVM_AS" -o "$B/function_constant.air" "$SRC/crates/metal2vulkan-validation/fixtures/public/kernel_function_constant_gate_fold.ll"
cat > "$B/Cargo.toml" <<EOT
[package]
name = "native-metal-bridge"
version = "0.1.0"
edition = "2024"
rust-version = "1.98"
[workspace]
[lib]
crate-type = ["staticlib"]
path = "$G/userland/metal2/bridge/NativeMetalBridge.rs"
[dependencies]
metal2vulkan = { path = "$SRC/crates/metal2vulkan", features = ["ash"] }
metal2vulkan-runtime = { path = "$SRC/crates/metal2vulkan-runtime" }
libc = "0.2"
sha2 = { version = "0.10", features = ["asm"] }
[profile.release]
panic = "unwind"
debug = 1
lto = "fat"
codegen-units = 1
EOT
triple=arm64e-apple-ios
RUSTC_BOOTSTRAP=1 NMB_FIXTURE_AIR=$B/runtime_examples.air NMB_FC_AIR=$B/function_constant.air IPHONEOS_DEPLOYMENT_TARGET=18.0 \
  cargo +1.98.0 build --release -Zbuild-std=std,panic_unwind --manifest-path "$B/Cargo.toml" --target $triple \
  --target-dir "$B/target" > "$B/build-arm64e.log" 2>&1 || { grep -E "^error" -A8 "$B/build-arm64e.log" | head -60; exit 1; }
BRIDGE="$B/target/$triple/release/libnative_metal_bridge.a"
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CC="xcrun --sdk iphoneos clang -arch arm64e -miphoneos-version-min=18.0 -isysroot $SDK"
$CC -x objective-c++ -std=c++17 -fobjc-arc -O2 -g -Wall -Wno-deprecated-declarations -Wno-unused-function \
  -Wno-unguarded-availability-new -I"$G/userland/metal2" -dynamiclib -o "$OUT/libm2v.dylib" \
  "$HERE/m2v/m2v.mm" "$G/userland/metal2/NativeBridgeMetalFrontend.mm" -x none "$BRIDGE" -Wl,-dead_strip -lc++ \
  -framework Foundation -framework Metal -framework QuartzCore -framework IOSurface -framework IOKit -framework CoreFoundation \
  -install_name /usr/local/lib/bgmtl/libm2v.dylib -Wl,-rpath,@loader_path
xcrun strip -x "$OUT/libm2v.dylib"
python3 "$BG_GPU/../tools/renegade/pacstrip.py" --sign "$OUT/libm2v.dylib" >/dev/null
ls -l "$OUT/libm2v.dylib"
