#!/bin/bash
# SPDX-License-Identifier: MIT
# Turnip (Mesa's Vulkan driver for Adreno) for iOS arm64e, in a Mesa 26.2.5 tree of its own ($WORK/mesa-turnip26): Chlorophyll's compiler
# patches (AIR frontend, ir3 fixes) plus turnip-ios.patch, all from patches/mesa-26.2/, no Gallium. The msm kernel backend runs over Bergamota's libdrm
# stand-in (gpu/freedreno-sweet), which emulates DRM syncobjs and device enumeration on SweetAdrenoUserClient. See docs/TURNIP.md.
#   GPU_METAL_WORK=/Users/zesmoi/git/iAndroid/work-chlorophyll gpu/metal/turnip/build-turnip.sh
# Needs: TURNIP_MESA (pristine 26.2.5 checkout, default $WORK/mesa-src-26.2.5, cloned when missing), GPU_EAGL_LIBDRM (libdrm 2.4.124 source, default
# $WORK/libdrm-src), GPU_EAGL_VENV (mako, pyyaml, meson; default $WORK/venv), brew bison and glslang (glslangValidator).
# Produces $WORK/mesa-turnip26/build-ios/src/freedreno/vulkan/libchlorophyll.dylib (Turnip + the AIR frontend + Chlorophyll's runtime,
# air/chl.h), libvulkan_freedreno.dylib (the plain ICD, for turnip/test/tu_smoke.c) and the libdrm stand-in in drm-ios/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BG_GPU="${BERGAMOTA_GPU:-$HERE/..}"
[ -d "$BG_GPU/freedreno-sweet" ] || { echo "no $BG_GPU/freedreno-sweet: set BERGAMOTA_GPU to a Bergamota checkout's gpu/" >&2; exit 1; }
WORK="${GPU_METAL_WORK:?set GPU_METAL_WORK}"
MESA="${TURNIP_MESA:-$WORK/mesa-src-26.2.5}"
MESA_TAG=mesa-26.2.5
LIBDRM="${GPU_EAGL_LIBDRM:-$WORK/libdrm-src}"
VENV="${GPU_EAGL_VENV:-$WORK/venv}"
command -v glslangValidator >/dev/null || { echo "glslangValidator missing: brew install glslang" >&2; exit 1; }
export PATH="$(brew --prefix bison)/bin:$VENV/bin:$PATH"
D="${TURNIP_TREE:-$WORK/mesa-turnip26}"
[ -d "$MESA" ] || git clone -q --depth 1 --branch "$MESA_TAG" https://gitlab.freedesktop.org/mesa/mesa.git "$MESA"
mkdir -p "$D"
cd "$D"
if [ ! -d mesa ]; then
  git clone -q --shared "$MESA" mesa
  for p in air-frontend ir3-round-default ir3-ssbo-subdword ir3-context-error-return ir3-disk-cache-macho ir3-opt-jump-linear turnip-ios; do
    git -C mesa apply "$HERE/patches/mesa-26.2/$p.patch"
  done
  mkdir -p mesa/src/compiler/air
  for f in "$HERE"/air/*; do ln -sf "$f" "mesa/src/compiler/air/$(basename "$f")"; done     # edit in the repo, build here
fi
"$BG_GPU/freedreno-sweet/build.sh" "$D/drm-ios" >/dev/null
mkdir -p pc-ios drm-ios-link
cp drm-ios/libdrm.2.arm64e.dylib drm-ios-link/libdrm.2.dylib; ln -sf libdrm.2.dylib drm-ios-link/libdrm.dylib
cat > pc-ios/libdrm.pc <<EOP
Name: libdrm
Description: Bergamota libdrm stand-in (gpu/freedreno-sweet)
Version: 2.4.124
Cflags: -I$LIBDRM/include/drm -I$LIBDRM
Libs: -L$D/drm-ios-link -ldrm
EOP
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
# pkg_config_libdir (not only _path): Homebrew's host SPIRV-Tools would otherwise be found and linked into the arm64e dylib
cat > ios.cross <<EOT
[binaries]
c = ['xcrun','--sdk','iphoneos','clang','-arch','arm64e']
cpp = ['xcrun','--sdk','iphoneos','clang++','-arch','arm64e']
objc = ['xcrun','--sdk','iphoneos','clang','-arch','arm64e']
ar = ['xcrun','--sdk','iphoneos','ar']
strip = ['xcrun','--sdk','iphoneos','strip']
ranlib = ['xcrun','--sdk','iphoneos','ranlib']
bison = '$(brew --prefix bison)/bin/bison'
pkg-config = '$(command -v pkg-config)'

[built-in options]
pkg_config_path = ['$D/pc-ios']
c_args = ['-miphoneos-version-min=18.0','-isysroot','$SDK']
cpp_args = ['-miphoneos-version-min=18.0','-isysroot','$SDK']
c_link_args = ['-miphoneos-version-min=18.0','-isysroot','$SDK']
cpp_link_args = ['-miphoneos-version-min=18.0','-isysroot','$SDK']

[host_machine]
system = 'darwin'
subsystem = 'ios'
cpu_family = 'aarch64'
cpu = 'arm64e'
endian = 'little'

[properties]
pkg_config_libdir = ['$D/pc-ios']
needs_exe_wrapper = true
EOT
[ -d build-ios ] || meson setup build-ios mesa --cross-file ios.cross --buildtype=release -Dgallium-drivers= -Dvulkan-drivers=freedreno \
  -Dfreedreno-kmds=msm -Dvulkan-layers= -Dplatforms= -Dllvm=disabled -Dopengl=false -Dgles1=disabled -Dgles2=disabled -Dglx=disabled \
  -Degl=disabled -Dgbm=disabled -Dtools= -Dvalgrind=disabled -Dlibunwind=disabled -Dzstd=disabled -Dxmlconfig=disabled -Dexpat=disabled \
  -Dglvnd=disabled -Dlmsensors=disabled -Dbuild-tests=false -Dshader-cache=enabled -Dvideo-codecs= -Dperfetto=false -Dspirv-tools=disabled \
  -Ddisplay-info=disabled -Dprefix=/ > setup-ios.log
ninja -j"${JOBS:-8}" -C build-ios src/freedreno/vulkan/libvulkan_freedreno.dylib src/freedreno/vulkan/libchlorophyll.dylib
ls -l build-ios/src/freedreno/vulkan/libvulkan_freedreno.dylib build-ios/src/freedreno/vulkan/libchlorophyll.dylib
