// SPDX-License-Identifier: MIT
// m2v_probe: the metal2vulkan trial device (libm2v.dylib) outside any system process. Opens it, fills a buffer with a blit,
// clears a texture over an IOSurface (the dma-buf import) through a render pass, and reads both back.
//   /var/root/m2v_probe            (log lines also in /private/var/tmp/m2v.log when that file exists)
#import <Foundation/Foundation.h>
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
