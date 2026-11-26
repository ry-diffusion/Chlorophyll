# Chlorophyll

> **Archived (2026-11-26).** Chlorophyll is retired. Its successor is [Crimson](https://github.com/ry-diffusion/Crimson) (branch `chlorophyll`, `ios/`): a fork of metal2vulkan that runs iOS 26 CoreAnimation on Turnip for Bergamota. The inject (`ios/inject`) and the Turnip build (`ios/turnip`, Mesa 26.2.5 without the AIR frontend) moved there; the phone no longer runs any code from this repository. Last working branches: `turnip` (iOS 26) and `main`.


Chlorophyll is the Metal driver of [Bergamota](https://github.com/ry-diffusion/Bergamota) (iOS 18 on the Poco F1, Adreno 630).
iOS apps and CoreAnimation ship their shaders as AIR (Apple's LLVM-based GPU IR) inside `.metallib` files; Chlorophyll
translates AIR to Mesa's NIR and compiles it with freedreno's ir3 backend for the Adreno 630, and implements the
Metal API (`MTLDevice`, command buffers, encoders, textures, IOSurfaces) over Mesa's freedreno Gallium driver.

| Part | Where |
| --- | --- |
| AIR → NIR frontend (bitcode reader, layout, intrinsics, `air2nir`) | `air/` |
| Metal runtime over Gallium (`libbgair.dylib`, `air_metal.c`) | `air/air_metal.c` |
| `BGMTLDevice` and the Metal classes | `src/bgmtl.m` |
| Injection into processes (backboardd's compositor profile and watchdog) | `inject/` |
| Mesa patches (ir3 fixes, the AIR frontend hook) | `patches/` |
| Probes and tests (CoreAnimation renderer harness, kernel tests) | `test/` |
| Corpus and comparison tools | `tools/`, `results/` |
| Rendering bugs fixed so far: symptom, cause, fix, how to check | `docs/FIXES.md` |
| Moving to Turnip (Vulkan), replacing Gallium: build, phone results, port plan | `docs/TURNIP.md`, `turnip/`, `patches/turnip-ios.patch` |

## Checkout and build

Chlorophyll is normally checked out as Bergamota's `gpu/metal` submodule, next to the pieces of Bergamota it builds
against (`gpu/freedreno-sweet`, the libdrm stand-in over the Bergamota kernel, and `gpu/eagl/mesa`, the Bergamota Mesa patch):

```bash
git clone --recurse-submodules https://github.com/ry-diffusion/Bergamota
```

The `mesa` (25.0.7) and `libdrm` (2.4.124) submodules are shallow. For a standalone checkout, point `BERGAMOTA_GPU` at a
Bergamota checkout's `gpu/` directory. Then:

```bash
GPU_METAL_WORK=/scratch GPU_EAGL_VENV=/path/to/venv ./build-host.sh
GPU_METAL_WORK=/scratch GPU_EAGL_VENV=/path/to/venv GPU_EAGL_CROSS=/path/to/ios-hw.cross ./build-ios.sh
```

`build-host.sh` makes a patched copy of the Mesa tree in `$GPU_METAL_WORK/mesa-air` (the submodule is never modified) and
builds the host tools; `build-ios.sh` cross-builds `libbgair.dylib` for iOS arm64e.

## Licence

MIT. The AIR front end and the AIR-to-ir3 library in `air/` (including `air_metal.c`/`air_metal.h`) and the Mesa patches
`air-frontend`, `air-ir3`, `ir3-*` come from the hackios userland repository by OOSEX GANG / Darwin on redwood contributors
(https://github.com/oosexgang/hackios-xnu-userland, MIT); see `NOTICE.md` and `LICENSE.hackios-userland`. Thanks to them.
