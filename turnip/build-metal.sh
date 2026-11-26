#!/bin/bash
# SPDX-License-Identifier: MIT
# Chlorophyll's Metal device on Turnip for the phone: bgmtl.o (src/bgmtl.m over air/chl.h), the Metal-level probes of test/ and
# the injection dylib, linked with libchlorophyll.dylib (turnip/build-turnip.sh) and the libdrm stand-in. Into OUT (default
# $GPU_METAL_WORK/out-metal); test tools are signed with turnip/entitlements.plist, everything PAC-stripped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BG_GPU="${BERGAMOTA_GPU:-$HERE/..}"
WORK="${GPU_METAL_WORK:?set GPU_METAL_WORK}"
D="${TURNIP_TREE:-$WORK/mesa-turnip26}"
OUT="${1:-$WORK/out-metal}"
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CC="xcrun --sdk iphoneos clang -arch arm64e -miphoneos-version-min=18.0 -isysroot $SDK"
OBJC="-O1 -g -fno-objc-arc -Wall -Wno-deprecated-declarations -Wno-unguarded-availability-new"
mkdir -p "$OUT"
cp -f "$D/build-ios/src/freedreno/vulkan/libchlorophyll.dylib" "$OUT/"
xcrun strip -x "$OUT/libchlorophyll.dylib"
cp -f "$D/drm-ios/libdrm.2.arm64e.dylib" "$OUT/libdrm.2.dylib"
LIBS=("$OUT/libchlorophyll.dylib" "$OUT/libdrm.2.dylib")
$CC $OBJC -I"$HERE/air" -c "$HERE/src/bgmtl.m" -o "$OUT/bgmtl.o"
for m in bg_kernels bg_render; do
  xcrun -sdk iphoneos metal -mios-version-min=18.0 -c "$HERE/test/$m.metal" -o "$OUT/$m.air"
  xcrun -sdk iphoneos metallib "$OUT/$m.air" -o "$OUT/$m.metallib"
done
probe() {   # probe NAME [frameworks...]
  local n=$1; shift
  $CC $OBJC -c "$HERE/test/$n.m" -o "$OUT/$n.o"
  $CC -o "$OUT/$n" "$OUT/bgmtl.o" "$OUT/$n.o" "${LIBS[@]}" -framework Foundation -framework Metal -framework IOSurface "$@" \
      -Wl,-rpath,@executable_path
  codesign -f -s - --entitlements "$HERE/turnip/entitlements.plist" "$OUT/$n" 2>/dev/null
}
probe bgmtl_probe
probe bgmtl_render_probe
probe bgmtl_layer_probe -framework QuartzCore
probe bgmtl_carenderer_probe -framework QuartzCore -framework CoreGraphics
probe bgmtl_xr2p_test -framework QuartzCore
probe bgmtl_adddev
probe bgmtl_texpool_test
# The injection dylib moved to Crimson (ry-diffusion/Crimson ios/inject: crimson-inject.dylib, 2026-11-26).
for f in "$OUT"/*; do
  case "$(file -b "$f")" in *Mach-O*) python3 "$BG_GPU/../tools/renegade/pacstrip.py" --sign "$f" >/dev/null ;; esac
done
ls -l "$OUT"
