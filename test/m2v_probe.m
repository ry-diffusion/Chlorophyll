// SPDX-License-Identifier: MIT
// m2v_probe: the metal2vulkan trial device (libm2v.dylib) outside any system process. Opens it, fills a buffer with a blit,
// clears a texture over an IOSurface (the dma-buf import) through a render pass, and reads both back.
//   /var/root/m2v_probe            (log lines also in /private/var/tmp/m2v.log when that file exists)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurfaceRef.h>
#include <dlfcn.h>
#include <stdio.h>

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
   }
   return 0;
}
