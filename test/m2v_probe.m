// SPDX-License-Identifier: MIT
// m2v_probe: the metal2vulkan trial device (libm2v.dylib) outside any system process. Opens it, fills a buffer with a blit,
// clears a texture over an IOSurface (the dma-buf import) through a render pass, and reads both back.
//   /var/root/m2v_probe            (log lines also in /private/var/tmp/m2v.log when that file exists)
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurfaceRef.h>
#include <dlfcn.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

@interface NSObject (M2VPrivate)
- (id)newTextureWithBytesNoCopy:(void *)p length:(NSUInteger)l descriptor:(MTLTextureDescriptor *)d deallocator:(id)dealloc;
@end

int
main(void)
{
   @autoreleasepool {
      void *h = dlopen("/usr/local/lib/bgmtl/libm2v.dylib", RTLD_NOW);
      if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
      id (*create)(void) = (id(*)(void))dlsym(h, "m2v_create_device");
      id<MTLDevice> dev = create ? create() : nil;
      printf("device: %s\n", dev ? [[dev name] UTF8String] : "refused (see /private/var/tmp/m2v.log)");
      if (!dev) return 1;
      id<MTLCommandQueue> q = [dev newCommandQueue];
      // audit: _MTLDevice methods the device class does not override (a call to one runs Apple's base version, often nil)
      if (getenv("M2V_AUDIT")) {
         Class base = objc_getClass("_MTLDevice"), mine = object_getClass(dev);
         unsigned nb = 0, nm = 0;
         Method *bm = class_copyMethodList(base, &nb), *mm = class_copyMethodList(mine, &nm);
         for (unsigned i = 0; i < nb; i++) {
            SEL sel = method_getName(bm[i]);
            int own = 0;
            for (unsigned j = 0; j < nm && !own; j++) own = method_getName(mm[j]) == sel;
            if (!own) printf("AUDIT device-base %s\n", sel_getName(sel));
         }
         free(bm); free(mm);
         return 0;
      }
      // M2V_SWIZZLE=1: RenderBox's glyph layers (iconservicesagent): an RG8Unorm texture (grey, alpha) read through the swizzle RRRG.
      // Texels (255, 128) drawn unblended must read back grey in every colour channel with alpha 128: 0x80ffffff (unswizzled:
      // 0xffff8000, the yellow glyph in an opaque box the home screen's icons had).
      if (getenv("M2V_SWIZZLE")) {
         NSError *err = nil;
         id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"/var/root/bg_render.metallib"] error:&err];
         MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
         pd.vertexFunction = [lib newFunctionWithName:@"bg_vs"];
         pd.fragmentFunction = [lib newFunctionWithName:@"bg_fs_tex"];
         pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
         id<MTLRenderPipelineState> ps = pd.vertexFunction ? [dev newRenderPipelineStateWithDescriptor:pd error:&err] : nil;
         MTLTextureDescriptor *sd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG8Unorm width:4 height:4 mipmapped:NO];
         sd.swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleGreen);
         id<MTLTexture> st = [dev newTextureWithDescriptor:sd];
         uint8_t la[32];
         for (int k = 0; k < 32; k += 2) { la[k] = 255; la[k + 1] = 128; }
         [st replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:la bytesPerRow:8];
         MTLTextureDescriptor *rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:8 height:8 mipmapped:NO];
         rd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
         id<MTLTexture> rt = [dev newTextureWithDescriptor:rd];
         static const float quad[48] = {-1, -1, 0, 1, 1, 1, 1, 1, 3, -1, 0, 1, 1, 1, 1, 1, -1, 3, 0, 1, 1, 1, 1, 1};   // one triangle over the target
         id<MTLBuffer> vb = [dev newBufferWithBytes:quad length:sizeof quad options:MTLResourceStorageModeShared];
         uint32_t px[64] = {0};
         if (ps && st && rt) {
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture = rt; rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
            [re setRenderPipelineState:ps];
            [re setVertexBuffer:vb offset:0 atIndex:0];
            [re setFragmentTexture:st atIndex:0];
            [re setFragmentSamplerState:[dev newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]] atIndex:0];
            [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [re endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            [rt getBytes:px bytesPerRow:32 fromRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0];
         }
         MTLTextureSwizzleChannels z = st.swizzle;
         printf("swizzle: pipeline %d texture %d, its swizzle %u%u%u%u (want 2223), pixel 0x%08x 0x%08x, want 0x80ffffff: %s\n", ps != nil, st != nil, z.red,
                z.green, z.blue, z.alpha, px[0], px[63], px[0] == 0x80ffffff && px[63] == 0x80ffffff ? "PASS" : "FAIL");
      }
      // M2V_LINEAR16F=1: an icon's image as SpringBoard's RenderBox samples it: a 128x128 RGBA16Float texture over a buffer (rows of
      // 1024 bytes, at M2V_LINEAR_OFFSET, default 0), white texels written before (and, a second texture, after) the texture is made.
      // Drawn into a 128x128 BGRA8 target, every row must read back white.
      if (getenv("M2V_LINEAR16F")) {
         NSError *err = nil;
         id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"/var/root/bg_render.metallib"] error:&err];
         MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
         pd.vertexFunction = [lib newFunctionWithName:@"bg_vs"];
         pd.fragmentFunction = [lib newFunctionWithName:@"bg_fs_tex"];
         pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
         id<MTLRenderPipelineState> ps = pd.vertexFunction ? [dev newRenderPipelineStateWithDescriptor:pd error:&err] : nil;
         const NSUInteger off = getenv("M2V_LINEAR_OFFSET") ? atoi(getenv("M2V_LINEAR_OFFSET")) : 0, N = 128, ROW = N * 8;
         MTLTextureDescriptor *ld = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:N height:N mipmapped:NO];
         ld.usage = MTLTextureUsageShaderRead; ld.storageMode = MTLStorageModeShared;
         MTLTextureDescriptor *rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:N height:N mipmapped:NO];
         rd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
         static const float tri[24] = {-1, -1, 0, 1, 1, 1, 1, 1, 3, -1, 0, 1, 1, 1, 1, 1, -1, 3, 0, 1, 1, 1, 1, 1};
         id<MTLBuffer> vb = [dev newBufferWithBytes:tri length:sizeof tri options:MTLResourceStorageModeShared];
         for (int late = 0; late < 2 && ps; late++) {
            id<MTLBuffer> buf = [dev newBufferWithLength:off + ROW * N options:MTLResourceStorageModeShared];
            uint16_t *h = (uint16_t *)((uint8_t *)buf.contents + off);
            if (!late) for (NSUInteger k = 0; k < N * N * 4; k++) h[k] = 0x3c00;   // 1.0
            id<MTLTexture> lt = [buf newTextureWithDescriptor:ld offset:off bytesPerRow:ROW];
            if (late) for (NSUInteger k = 0; k < N * N * 4; k++) h[k] = 0x3c00;
            id<MTLTexture> rt = [dev newTextureWithDescriptor:rd];
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture = rt; rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLRenderCommandEncoder> re = lt ? [cb renderCommandEncoderWithDescriptor:rp] : nil;
            [re setRenderPipelineState:ps];
            [re setVertexBuffer:vb offset:0 atIndex:0];
            [re setFragmentTexture:lt atIndex:0];
            [re setFragmentSamplerState:[dev newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]] atIndex:0];
            [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [re endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            static uint32_t px[128 * 128];
            memset(px, 0, sizeof px);
            if (lt) [rt getBytes:px bytesPerRow:N * 4 fromRegion:MTLRegionMake2D(0, 0, N, N) mipmapLevel:0];
            NSUInteger white = 0, first = N;
            for (NSUInteger y = 0; y < N; y++) {
               bool all = true;
               for (NSUInteger x = 0; x < N; x++) all = all && px[y * N + x] == 0xffffffff;
               if (all) white++; else if (first == N) first = y;
            }
            printf("linear16f (offset %lu, texels written %s the texture): texture %d, %lu of %lu rows white, first other row %lu (0x%08x): %s\n", (unsigned long)off,
                   late ? "after" : "before", lt != nil, (unsigned long)white, (unsigned long)N, (unsigned long)first, first < N ? px[first * N] : 0, white == N ? "PASS" : "FAIL");
         }
      }
      // M2V_ASTC=1: RenderBox's glyph layers in SpringBoard ("rb:nocopy-data-texture"): a 132x132 ASTC 4x4 texture over the caller's
      // blocks (newTextureWithBytesNoCopy:), every block a void-extent of opaque white. Drawn into a BGRA8 target, every row must
      // read back white.
      if (getenv("M2V_ASTC")) {
         NSError *err = nil;
         id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"/var/root/bg_render.metallib"] error:&err];
         MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
         pd.vertexFunction = [lib newFunctionWithName:@"bg_vs"];
         pd.fragmentFunction = [lib newFunctionWithName:@"bg_fs_tex"];
         pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
         id<MTLRenderPipelineState> ps = pd.vertexFunction ? [dev newRenderPipelineStateWithDescriptor:pd error:&err] : nil;
         const NSUInteger N = getenv("M2V_ASTC_N") ? MIN(atoi(getenv("M2V_ASTC_N")), 2048) : 132, B = (N + 3) / 4, LEN = B * B * 16, MAP = (LEN + 16383) & ~(NSUInteger)16383;
         uint8_t *mem = NULL; posix_memalign((void **)&mem, 16384, MAP); memset(mem, 0, MAP);
         static const uint8_t white[16] = {0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
         for (NSUInteger k = 0; k < B * B; k++) memcpy(mem + k * 16, white, 16);
         if (getenv("M2V_ASTC_RAMP"))   // red is the block's row, green its column (both from 1), to see where each block lands
            for (NSUInteger k = 0; k < B * B; k++) {
               mem[k * 16 + 8] = mem[k * 16 + 9] = (uint8_t)(k / B + 1);
               mem[k * 16 + 10] = mem[k * 16 + 11] = (uint8_t)(k % B + 1);
            }
         MTLTextureDescriptor *ad = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatASTC_4x4_LDR width:N height:N mipmapped:NO];
         ad.usage = MTLTextureUsageShaderRead; ad.storageMode = MTLStorageModeShared;
         id<MTLTexture> at = [(id)dev newTextureWithBytesNoCopy:mem length:getenv("M2V_ASTC_EXACT") ? LEN : MAP descriptor:ad deallocator:nil];
         if (getenv("M2V_ASTC_CONTROL")) {   // the same draw from a plain BGRA8 texture: what the target and the draw do by themselves
            uint32_t *w8 = malloc(N * N * 4);
            memset(w8, 0xff, N * N * 4);
            at = [dev newTextureWithDescriptor:[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:N height:N mipmapped:NO]];
            [at replaceRegion:MTLRegionMake2D(0, 0, N, N) mipmapLevel:0 withBytes:w8 bytesPerRow:N * 4];
         }
         MTLTextureDescriptor *rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:N height:N mipmapped:NO];
         rd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
         id<MTLTexture> rt = [dev newTextureWithDescriptor:rd];
         static const float tri[24] = {-1, -1, 0, 1, 1, 1, 1, 1, 3, -1, 0, 1, 1, 1, 1, 1, -1, 3, 0, 1, 1, 1, 1, 1};
         id<MTLBuffer> vb = [dev newBufferWithBytes:tri length:sizeof tri options:MTLResourceStorageModeShared];
         uint32_t *px = calloc(N * N, 4);
         if (ps && at && rt) {
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture = rt; rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
            [re setRenderPipelineState:ps];
            [re setVertexBuffer:vb offset:0 atIndex:0];
            [re setFragmentTexture:at atIndex:0];
            [re setFragmentSamplerState:[dev newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]] atIndex:0];
            [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [re endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            [rt getBytes:px bytesPerRow:N * 4 fromRegion:MTLRegionMake2D(0, 0, N, N) mipmapLevel:0];
         }
         NSUInteger white_rows = 0, first = N;
         for (NSUInteger y = 0; y < N; y++) {
            bool all = true;
            for (NSUInteger x = 0; x < N; x++) all = all && px[y * N + x] == 0xffffffff;
            if (all) white_rows++; else if (first == N) first = y;
         }
         if (getenv("M2V_ASTC_MAP"))   // one character a block: W all white, . none, + some
            for (NSUInteger by = 0; by < B; by++) {
               char line[520];
               for (NSUInteger bx = 0; bx < B; bx++) {
                  int w = 0, n = 0;
                  for (NSUInteger y = by * 4; y < by * 4 + 4 && y < N; y++)
                     for (NSUInteger x = bx * 4; x < bx * 4 + 4 && x < N; x++) { n++; w += px[y * N + x] == 0xffffffff; }
                  line[bx] = w == n ? 'W' : w ? '+' : '.';
               }
               line[B] = 0;
               printf("astc %2lu %s %08x %08x %08x\n", (unsigned long)by, line, px[by * 4 * N], px[by * 4 * N + N / 2], px[by * 4 * N + N - 1]);
            }
         if (getenv("M2V_ASTC_RAMP") && !getenv("M2V_ASTC_CONTROL") && N % 4 == 0 && B < 256) {   // every block where it belongs (the draw flips y)
            NSUInteger wrong = 0;
            for (NSUInteger by = 0; by < B; by++)
               for (NSUInteger bx = 0; bx < B; bx++) wrong += px[(by * 4 + 1) * N + bx * 4 + 1] != (0xff0000ff | (uint32_t)(B - by) << 16 | (uint32_t)(bx + 1) << 8);
            printf("astc ramp %lux%lu: %lu of %lu blocks misplaced: %s\n", (unsigned long)N, (unsigned long)N, (unsigned long)wrong, (unsigned long)(B * B), wrong ? "FAIL" : "PASS");
         }
         if (!getenv("M2V_ASTC_RAMP"))
         printf("astc: pipeline %d texture %d (%lu bytes of blocks), %lu of %lu rows white, first other row %lu (0x%08x 0x%08x): %s\n", ps != nil, at != nil, (unsigned long)LEN,
                (unsigned long)white_rows, (unsigned long)N, (unsigned long)first, first < N ? px[first * N] : 0, first < N ? px[first * N + N - 1] : 0, white_rows == N ? "PASS" : "FAIL");
      }
      // M2V_RGB10A2=1: RenderBox's glass mask case (iconservicesagent): an RGB10A2Unorm render target over a no-copy buffer of plain
      // memory (rows 256 bytes, Metal's own 240), 60x60. (a) a clear pass: every texel must read back 0xC00003FF (R=1023, A=3);
      // (b) a load+store pass with no draw: the CPU's pattern must come back unchanged.
      if (getenv("M2V_RGB10A2")) {
         const NSUInteger W = 60, H = 60, ROW = 256, LEN = 16384;
         void *mem = NULL; posix_memalign(&mem, 16384, LEN);
         id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:mem length:LEN options:MTLResourceStorageModeShared deallocator:nil];
         MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGB10A2Unorm width:W height:H mipmapped:NO];
         td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; td.storageMode = MTLStorageModeShared;
         id<MTLTexture> t = [buf newTextureWithDescriptor:td offset:0 bytesPerRow:ROW];
         printf("rgb10a2: buffer %s, texture %s\n", buf ? "ok" : "refused", t ? "ok" : "refused");
         if (!t) return 1;
         for (int step = 0; step < 2; step++) {
            uint32_t *w = mem;
            for (NSUInteger k = 0; k < LEN / 4; k++) w[k] = 0x12345678u;
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture = t;
            rp.colorAttachments[0].loadAction = step == 0 ? MTLLoadActionClear : MTLLoadActionLoad;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
            id<MTLCommandBuffer> cb = [q commandBuffer];
            [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            const uint32_t want = step == 0 ? 0xC00003FFu : 0x12345678u;
            unsigned bad = 0, badRows = 0, firstBadRow = 999;
            for (NSUInteger y = 0; y < H; y++) {
               unsigned rb = 0;
               for (NSUInteger x = 0; x < W; x++) if (((uint32_t *)((uint8_t *)mem + y * ROW))[x] != want) { bad++; rb++; }
               if (rb) { badRows++; if (firstBadRow == 999) firstBadRow = (unsigned)y; }
            }
            uint32_t *r0 = mem, *rl = (uint32_t *)((uint8_t *)mem + (H - 1) * ROW);
            printf("rgb10a2 %s: %u of %lu texels wrong in %u rows (first row %u); row0[0]=%08x row59[0]=%08x row59[59]=%08x want %08x\n",
                   step == 0 ? "clear" : "load+store", bad, (unsigned long)(W * H), badRows, firstBadRow, r0[0], rl[0], rl[W - 1], want);
         }
         // (c) a draw over the top half only (/var/root/half.metallib: half_vs/half_fs, solid green), the pass loading the CPU's
         // pattern: the top 30 rows must read 0xC00FFC00 (G=1023, A=3), the bottom 30 the pattern
         NSError *err = nil;
         id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"/var/root/half.metallib"] error:&err];
         MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
         pd.vertexFunction = [lib newFunctionWithName:@"half_vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"half_fs"];
         pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGB10A2Unorm;
         id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
         if (!ps) { printf("rgb10a2 draw: pipeline refused: %s\n", [[err description] UTF8String]); return 1; }
         uint32_t *w = mem;
         for (NSUInteger k = 0; k < LEN / 4; k++) w[k] = 0x12345678u;
         MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
         rp.colorAttachments[0].texture = t; rp.colorAttachments[0].loadAction = MTLLoadActionLoad; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
         id<MTLCommandBuffer> cb = [q commandBuffer];
         id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
         [re setRenderPipelineState:ps]; [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6]; [re endEncoding];
         [cb commit]; [cb waitUntilCompleted];
         unsigned topBad = 0, botBad = 0;
         for (NSUInteger y = 0; y < H; y++)
            for (NSUInteger x = 0; x < W; x++) {
               uint32_t v = ((uint32_t *)((uint8_t *)mem + y * ROW))[x];
               if (y < H / 2) topBad += v != 0xC00FFC00u; else botBad += v != 0x12345678u;
            }
         printf("rgb10a2 half draw: top %u of 1800 wrong, bottom %u of 1800 wrong; row0[0]=%08x row29[0]=%08x row30[0]=%08x row59[0]=%08x\n",
                topBad, botBad, w[0], ((uint32_t *)((uint8_t *)mem + 29 * ROW))[0], ((uint32_t *)((uint8_t *)mem + 30 * ROW))[0],
                ((uint32_t *)((uint8_t *)mem + 59 * ROW))[0]);
         // (d) RenderBox's way of drawing into such a target (iconservicesagent's icon layers): the same top-half quad under a
         // scissor (10, 6, 30 x 40), then blended, then with RenderBox's other attachments (RG16Float, RGBA16Float, depth and
         // stencil) beside it. Only x 10..39, y 6..29 may change.
         for (int variant = 0; variant < 3; variant++) {
            MTLRenderPipelineDescriptor *sd = [MTLRenderPipelineDescriptor new];
            sd.vertexFunction = pd.vertexFunction; sd.fragmentFunction = pd.fragmentFunction;
            sd.colorAttachments[0].pixelFormat = MTLPixelFormatRGB10A2Unorm;
            if (variant >= 1) {
               sd.colorAttachments[0].blendingEnabled = YES;
               sd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne; sd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
               sd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
               sd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            }
            MTLRenderPassDescriptor *sp = [MTLRenderPassDescriptor renderPassDescriptor];
            sp.colorAttachments[0].texture = t; sp.colorAttachments[0].loadAction = MTLLoadActionLoad; sp.colorAttachments[0].storeAction = MTLStoreActionStore;
            if (variant == 2) {
               const MTLPixelFormat extra[2] = {MTLPixelFormatRG16Float, MTLPixelFormatRGBA16Float};
               for (int k = 0; k < 2; k++) {
                  MTLTextureDescriptor *ed = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:extra[k] width:W height:H mipmapped:NO];
                  ed.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; ed.storageMode = MTLStorageModePrivate;
                  sp.colorAttachments[k + 1].texture = [dev newTextureWithDescriptor:ed];
                  sp.colorAttachments[k + 1].loadAction = MTLLoadActionLoad; sp.colorAttachments[k + 1].storeAction = MTLStoreActionStore;
                  sd.colorAttachments[k + 1].pixelFormat = extra[k]; sd.colorAttachments[k + 1].writeMask = MTLColorWriteMaskNone;
               }
               MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:W height:H mipmapped:NO];
               dd.usage = MTLTextureUsageRenderTarget; dd.storageMode = MTLStorageModePrivate;
               id<MTLTexture> ds = [dev newTextureWithDescriptor:dd];
               sp.depthAttachment.texture = ds; sp.stencilAttachment.texture = ds;
               sp.depthAttachment.loadAction = MTLLoadActionClear; sp.stencilAttachment.loadAction = MTLLoadActionClear;
               sd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8; sd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
            }
            id<MTLRenderPipelineState> sps = [dev newRenderPipelineStateWithDescriptor:sd error:&err];
            if (!sps) { printf("rgb10a2 scissor %d: pipeline refused: %s\n", variant, [[err description] UTF8String]); continue; }
            for (NSUInteger k = 0; k < LEN / 4; k++) w[k] = 0x12345678u;
            id<MTLCommandBuffer> scb = [q commandBuffer];
            id<MTLRenderCommandEncoder> se = [scb renderCommandEncoderWithDescriptor:sp];
            [se setScissorRect:(MTLScissorRect){10, 6, 30, 40}];
            [se setRenderPipelineState:sps]; [se drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6]; [se endEncoding];
            [scb commit]; [scb waitUntilCompleted];
            unsigned inBad = 0, outBad = 0; NSUInteger x0 = W, x1 = 0, y0 = H, y1 = 0;
            for (NSUInteger y = 0; y < H; y++)
               for (NSUInteger x = 0; x < W; x++) {
                  uint32_t v = ((uint32_t *)((uint8_t *)mem + y * ROW))[x];
                  const bool inside = x >= 10 && x < 40 && y >= 6 && y < 30;
                  if (inside) inBad += v != 0xC00FFC00u; else outBad += v != 0x12345678u;
                  if (v != 0x12345678u) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
               }
            printf("rgb10a2 scissor%s%s: %u of 720 inside wrong, %u outside wrong; changed x %lu..%lu y %lu..%lu (want 10..39, 6..29), status %ld\n",
                   variant >= 1 ? " + blend" : "", variant == 2 ? " + RenderBox's attachments" : "", inBad, outBad,
                   (unsigned long)x0, (unsigned long)x1, (unsigned long)y0, (unsigned long)y1, (long)scb.status);
         }
         return 0;
      }
      // an unknown selector is answered 0 and logged MISSING (libm2v's forwarding), never an abort
      long z1 = (long)[(id)dev performSelector:NSSelectorFromString(@"chlNoSuchDeviceMethod")];
      long z2 = (long)[(id)q performSelector:NSSelectorFromString(@"chlNoSuchQueueMethod")];
      printf("unknown selectors answered %ld %ld (want 0 0)\n", z1, z2);
      // 0. CPU bandwidth over a GPU buffer's mapping (write-combined) against malloc'd memory
      {
         const size_t N = 8 << 20;
         id<MTLBuffer> big = [dev newBufferWithLength:N options:MTLResourceStorageModeShared];
         uint8_t *g = [big contents], *m = malloc(N);
         for (int pass = 0; pass < 2; pass++) {
            uint8_t *p = pass ? m : g;
            uint64_t t0 = mach_absolute_time();
            memset(p, pass + 1, N);
            uint64_t t1 = mach_absolute_time();
            volatile uint64_t sum = 0;
            for (size_t k = 0; k < N; k += 8) sum += *(uint64_t *)(p + k);
            uint64_t t2 = mach_absolute_time();
            mach_timebase_info_data_t tb; mach_timebase_info(&tb);
            double w = (double)(t1 - t0) * tb.numer / tb.denom / 1e9, r = (double)(t2 - t1) * tb.numer / tb.denom / 1e9;
            printf("%s: write %.0f MB/s, read %.0f MB/s\n", pass ? "malloc " : "GPU buf", N / w / 1e6, N / r / 1e6);
         }
         free(m);
         // an IOSurface's own CPU mapping (what CoreGraphics draws into)
         NSDictionary *sp = @{(id)kIOSurfaceWidth : @1024, (id)kIOSurfaceHeight : @1024, (id)kIOSurfaceBytesPerElement : @4,
                              (id)kIOSurfacePixelFormat : @((uint32_t)'BGRA')};
         IOSurfaceRef bs = IOSurfaceCreate((__bridge CFDictionaryRef)sp);
         IOSurfaceLock(bs, 0, NULL);
         uint8_t *q = IOSurfaceGetBaseAddress(bs);
         size_t qn = IOSurfaceGetAllocSize(bs);
         uint64_t t0 = mach_absolute_time();
         memset(q, 7, qn);
         uint64_t t1 = mach_absolute_time();
         volatile uint64_t sum = 0;
         for (size_t k = 0; k < qn; k += 8) sum += *(uint64_t *)(q + k);
         uint64_t t2 = mach_absolute_time();
         IOSurfaceUnlock(bs, 0, NULL);
         mach_timebase_info_data_t tb; mach_timebase_info(&tb);
         printf("IOSurface: write %.0f MB/s, read %.0f MB/s (%zu bytes)\n", qn / ((double)(t1 - t0) * tb.numer / tb.denom / 1e9) / 1e6,
                qn / ((double)(t2 - t1) * tb.numer / tb.denom / 1e9) / 1e6, qn);
         CFRelease(bs);
      }
      // 1. a blit fill of a shared buffer
      id<MTLBuffer> b = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
      [bl fillBuffer:b range:NSMakeRange(0, 4096) value:0xab];
      [bl endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      const uint8_t *p = [b contents];
      int bad = 0;
      for (int i = 0; i < 4096; i++) bad += p[i] != 0xab;
      printf("blit fill: status %ld, %d of 4096 bytes wrong%s\n", (long)cb.status, bad, cb.error ? [[cb.error description] UTF8String] : "");
      // 2. a render-pass clear of a texture over an IOSurface (64x64 BGRA)
      NSDictionary *props = @{(id)kIOSurfaceWidth : @64, (id)kIOSurfaceHeight : @64, (id)kIOSurfaceBytesPerElement : @4,
                              (id)kIOSurfaceBytesPerRow : @256, (id)kIOSurfacePixelFormat : @((uint32_t)'BGRA')};
      IOSurfaceRef sf = IOSurfaceCreate((__bridge CFDictionaryRef)props);
      MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:64 height:64
                                                                               mipmapped:NO];
      td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
      td.storageMode = MTLStorageModeShared;
      id<MTLTexture> t = [dev newTextureWithDescriptor:td iosurface:sf plane:0];
      printf("IOSurface %u texture: %s\n", IOSurfaceGetID(sf), t ? "made" : "refused");
      if (t) {
         MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
         rp.colorAttachments[0].texture = t;
         rp.colorAttachments[0].loadAction = MTLLoadActionClear;
         rp.colorAttachments[0].storeAction = MTLStoreActionStore;
         rp.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
         cb = [q commandBuffer];
         [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
         [cb commit];
         [cb waitUntilCompleted];
         IOSurfaceLock(sf, kIOSurfaceLockReadOnly, NULL);
         const uint32_t *px = IOSurfaceGetBaseAddress(sf);
         printf("clear: status %ld, pixel (0,0) 0x%08x (0,63) 0x%08x (63,63) 0x%08x, want 0xffff0000%s\n", (long)cb.status, px[0],
                px[63 * 64], px[63 * 64 + 63], cb.error ? [[cb.error description] UTF8String] : "");
         IOSurfaceUnlock(sf, kIOSurfaceLockReadOnly, NULL);
      }
      CFRelease(sf);
      // 4. per-draw CPU cost: 1000 textured draws in one pass (vertex buffer offset, texture, sampler rebound each draw)
      {
         NSError *err = nil;
         id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"/var/root/bg_render.metallib"] error:&err];
         MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
         pd.vertexFunction = [lib newFunctionWithName:@"bg_vs"];
         pd.fragmentFunction = [lib newFunctionWithName:@"bg_fs_tex"];
         pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
         pd.colorAttachments[0].blendingEnabled = YES;
         pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
         pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
         id<MTLRenderPipelineState> ps = pd.vertexFunction ? [dev newRenderPipelineStateWithDescriptor:pd error:&err] : nil;
         if (!ps) printf("bench: no pipeline (%s)\n", err ? err.description.UTF8String : "no library");
         else {
            MTLTextureDescriptor *rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:828 height:1792 mipmapped:NO];
            rd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; rd.storageMode = MTLStorageModePrivate;
            id<MTLTexture> rt = [dev newTextureWithDescriptor:rd];
            MTLTextureDescriptor *td2 = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:64 height:64 mipmapped:NO];
            id<MTLTexture> tex[4]; for (int k = 0; k < 4; k++) tex[k] = [dev newTextureWithDescriptor:td2];
            id<MTLSamplerState> smp = [dev newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
            id<MTLBuffer> vb = [dev newBufferWithLength:1000 * 96 options:MTLResourceStorageModeShared];
            float *v = [vb contents]; for (int k = 0; k < 1000 * 24; k++) v[k] = (k % 8 < 2) ? ((k % 3) * 0.1f - 0.1f) : 1.0f;
            mach_timebase_info_data_t tb; mach_timebase_info(&tb);
            for (int round = 0; round < 3; round++) {
               MTLRenderPassDescriptor *rp2 = [MTLRenderPassDescriptor renderPassDescriptor];
               rp2.colorAttachments[0].texture = rt; rp2.colorAttachments[0].loadAction = MTLLoadActionClear; rp2.colorAttachments[0].storeAction = MTLStoreActionStore;
               uint64_t t0 = mach_absolute_time();
               id<MTLCommandBuffer> bcb = [q commandBuffer];
               id<MTLRenderCommandEncoder> re = [bcb renderCommandEncoderWithDescriptor:rp2];
               [re setRenderPipelineState:ps];
               [re setVertexBuffer:vb offset:0 atIndex:0];
               for (int k = 0; k < 1000; k++) {
                  [re setVertexBufferOffset:k * 96 atIndex:0];
                  [re setFragmentTexture:tex[k & 3] atIndex:0];
                  [re setFragmentSamplerState:smp atIndex:0];
                  [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
               }
               uint64_t t1 = mach_absolute_time();
               [re endEncoding];
               uint64_t t2 = mach_absolute_time();
               [bcb commit];
               uint64_t t3 = mach_absolute_time();
               [bcb waitUntilCompleted];
               uint64_t t4 = mach_absolute_time();
               double f = (double)tb.numer / tb.denom / 1e3;
               printf("bench %d: 1000 draws encode %.0f us (%.1f us/draw), endEncoding %.0f us, commit %.0f us, GPU %.0f us, status %ld\n", round,
                      (t1 - t0) * f, (t1 - t0) * f / 1000, (t2 - t1) * f, (t3 - t2) * f, (t4 - t3) * f, (long)bcb.status);
            }
            // CoreAnimation-like: indexed quads, per-draw vertex and fragment bytes, two textures, 8 passes of 22 draws
            uint16_t idx[6] = {0, 1, 2, 2, 1, 3};
            id<MTLBuffer> ib = [dev newBufferWithBytes:idx length:sizeof idx options:MTLResourceStorageModeShared];
            float vbytes[32], fbytes[8];
            for (int k = 0; k < 32; k++) vbytes[k] = (k % 8 < 2) ? 0.1f * (k % 4) : 1.0f;
            for (int k = 0; k < 8; k++) fbytes[k] = 0.5f;
            for (int round = 0; round < 3; round++) {
               uint64_t t0 = mach_absolute_time();
               id<MTLCommandBuffer> bcb = [q commandBuffer];
               for (int pass = 0; pass < 8; pass++) {
                  MTLRenderPassDescriptor *rp2 = [MTLRenderPassDescriptor renderPassDescriptor];
                  rp2.colorAttachments[0].texture = rt; rp2.colorAttachments[0].loadAction = pass ? MTLLoadActionLoad : MTLLoadActionClear;
                  rp2.colorAttachments[0].storeAction = MTLStoreActionStore;
                  id<MTLRenderCommandEncoder> re = [bcb renderCommandEncoderWithDescriptor:rp2];
                  [re setRenderPipelineState:ps];
                  for (int k = 0; k < 22; k++) {
                     [re setVertexBytes:vbytes length:sizeof vbytes atIndex:0];
                     [re setFragmentBytes:fbytes length:sizeof fbytes atIndex:1];
                     [re setFragmentTexture:tex[k & 3] atIndex:0];
                     [re setFragmentTexture:tex[(k + 1) & 3] atIndex:3];
                     [re setFragmentSamplerState:smp atIndex:0];
                     [re drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:6 indexType:MTLIndexTypeUInt16 indexBuffer:ib indexBufferOffset:0];
                  }
                  [re endEncoding];
               }
               uint64_t t1 = mach_absolute_time();
               [bcb commit];
               [bcb waitUntilCompleted];
               uint64_t t2 = mach_absolute_time();
               double f = (double)tb.numer / tb.denom / 1e3;
               printf("bench CA-like %d: 8 passes x 22 indexed draws: encode+end %.0f us (%.1f us/draw), commit+GPU %.0f us, status %ld\n", round,
                      (t1 - t0) * f, (t1 - t0) * f / 176, (t2 - t1) * f, (long)bcb.status);
            }
         }
      }
      // 3. a texture over plain memory with length 0 (SpringBoard's icon images): rows packed
      void *mem = NULL;
      posix_memalign(&mem, 16384, 64 * 64 * 4);
      memset(mem, 0x5a, 64 * 64 * 4);
      MTLTextureDescriptor *nd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:64 height:64 mipmapped:NO];
      id<MTLTexture> nt = [(id)dev newTextureWithBytesNoCopy:mem length:0 descriptor:nd deallocator:nil];
      uint32_t got = 0;
      if (nt) [nt getBytes:&got bytesPerRow:256 fromRegion:MTLRegionMake2D(5, 5, 1, 1) mipmapLevel:0];
      printf("no-copy texture (length 0): %s, texel 0x%08x (want 0x5a5a5a5a), usage %lu storage %lu\n", nt ? "made" : "refused", got,
             (unsigned long)nd.usage, (unsigned long)nd.storageMode);
   }
   return 0;
}
