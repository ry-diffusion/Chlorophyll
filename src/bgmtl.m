// SPDX-License-Identifier: MIT
//
// BGMTLDevice: Chlorophyll's Metal device, on Turnip.
//
// Design after hackios's `_MTLDevice` subclass (hackios-xnu-userland userland/src/hkmetal.m, MIT; read and re-written
// here, nothing copied wholesale): a subclass of Metal's private `_MTLDevice`, `MTLDeviceSPI` added to it with
// class_addProtocol. Every Metal object is a chl object (air/chl.h, libchlorophyll.dylib): buffers, textures, samplers and
// pipelines are Vulkan objects of Turnip on the Adreno 630, and the encoders of a command buffer record straight into one
// Vulkan command buffer (render passes, dispatches, copies), submitted at commit and completed on chl's completion thread.
//
// THIS FILE NEVER REGISTERS THE DEVICE. It does not call MTLAddDevice, does not touch the device array that
// MTLCreateSystemDefaultDevice / MTLCopyAllDevices read, and does not publish MetalPluginName on any IOKit service. A test
// process makes the object itself with bgmtl_create_device() and uses it directly. A half-working device must never be
// visible to backboardd/CoreAnimation (docs/METAL_ADRENO630.md, nil-or-complete rule).
//
// Scope: command queues and buffers, render / compute / blit encoders, buffers, textures (own memory, IOSurfaces, views, over
// buffers or caller memory), samplers, depth/stencil states, libraries and functions through Metal's own _MTLLibrary, render and
// compute pipelines. No events, indirect draws, tile shaders or Metal 4.
#import <Foundation/Foundation.h>
#include <libkern/OSByteOrder.h>
#include <os/lock.h>
#include <pthread.h>
#include <Block.h>
#include <sys/stat.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <time.h>
#import <IOSurface/IOSurfaceRef.h>
#include <dlfcn.h>
#include <mach/mach.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "chl.h"

// ---- the private bases (names only; their layouts are Metal's) -----------------------------------------------------
@interface _MTLDevice : NSObject
- (instancetype)init;
@end
@interface _MTLObjectWithLabel : NSObject
- (instancetype)init;
@end
@interface _MTLResource : _MTLObjectWithLabel
@end
@interface _MTLCommandQueue : _MTLObjectWithLabel
- (instancetype)initWithDevice:(id)device descriptor:(id)descriptor;
@end
@interface _MTLCommandBuffer : _MTLObjectWithLabel
- (instancetype)initWithQueue:(id)queue retainedReferences:(BOOL)retained synchronousDebugMode:(BOOL)synchronous;
@end
@interface _MTLCommandEncoder : _MTLObjectWithLabel
- (instancetype)initWithCommandBuffer:(id)commandBuffer;
- (void)endEncoding;
@end
@interface _MTLComputePipelineState : NSObject
- (instancetype)initWithDevice:(id)device pipelineStateDescriptor:(id)descriptor;
@end
@interface _MTLRenderPipelineState : NSObject
- (instancetype)initWithDevice:(id)device pipelineStateDescriptor:(id)descriptor;
@end
@interface _MTLSamplerState : NSObject
- (instancetype)initWithDevice:(id)device samplerDescriptor:(id)descriptor;
@end
@interface _MTLDepthStencilState : NSObject
- (instancetype)initWithDevice:(id)device depthStencilDescriptor:(id)descriptor;
@end

#define BG_LOG(...) bg_log(__VA_ARGS__)
static int bg_trace(void);
static void
bg_log(const char *fmt, ...)
{
   static FILE *tf;
   static int tf_tried;
   va_list ap;
   va_start(ap, fmt);
   fprintf(stderr, "bgmtl[%d]: ", getpid());
   vfprintf(stderr, fmt, ap);
   va_end(ap);
   // /private/var/tmp/bgmtl.trace: the log also goes to bgmtl-trace-<pid>.txt there (daemons and extensions whose stderr
   // nobody reads and whose environment launchd sets, such as the wallpaper poster)
   (void)tf_tried;
   // a trace file deleted while open (rm between two traces) would take every later line with it: reopen it then
   static time_t tf_checked;
   time_t now = time(NULL);
   if (tf && now != tf_checked) {
      struct stat st;
      tf_checked = now;
      if (fstat(fileno(tf), &st) == 0 && st.st_nlink == 0) { fclose(tf); tf = NULL; }
   }
   if (bg_trace() && !tf) {          // the trace file follows the live switch (bg_trace)
      char path[96];
      snprintf(path, sizeof(path), "/private/var/tmp/bgmtl-trace-%d.txt", getpid());
      tf = fopen(path, "a");
   } else if (!bg_trace() && tf) {
      fclose(tf);
      tf = NULL;
   }
   if (tf) {
      va_start(ap, fmt);
      vfprintf(tf, fmt, ap);
      va_end(ap);
      fflush(tf);
   }
}

// /private/var/tmp/bgmtl.faildump: a pipeline Chlorophyll refuses leaves its reason in bgmtl-fail/refused.txt and the bitcode of
// its functions in bgmtl-fail/<pid>-<name>.<stage>.bc, without the per-frame volume of the trace (backboardd's glass, the poster
// extensions' CoreImage kernels: callers that drop the NSError silently). air2nir on the Mac takes the .bc as it is.
static void
bg_fail_dump(const char *reason, NSData *vair, NSString *vname, NSData *air, NSString *name, const char *stage)
{
   if (access("/private/var/tmp/bgmtl.faildump", F_OK) != 0) return;
   mkdir("/private/var/tmp/bgmtl-fail", 0777);
   NSString *dir = @"/private/var/tmp/bgmtl-fail";
   if (vair) [vair writeToFile:[dir stringByAppendingFormat:@"/%d-%@.v.bc", getpid(), vname] atomically:NO];
   if (air) [air writeToFile:[dir stringByAppendingFormat:@"/%d-%@.%s.bc", getpid(), name, stage] atomically:NO];
   FILE *f = fopen("/private/var/tmp/bgmtl-fail/refused.txt", "a");
   if (!f) return;
   fprintf(f, "[%d %s] %s%s%s (%s): %s\n", getpid(), getprogname(), vname ? vname.UTF8String : "", vname ? " + " : "",
           name ? name.UTF8String : "?", stage, reason);
   fclose(f);
}

// /private/var/tmp/bgmtl.refuse: fragment-function name prefixes, one per line. A render pipeline whose fragment function starts
// with one is not built: its AIR (vertex, fragment and the linked functions) goes to /private/var/tmp/bgmtl-refused/ and the
// pipeline is refused. A diagnostic for a shader that hangs the GPU: the process keeps running and the shader can be examined
// on the Mac (air_vk_check) without spending one of the kernel's GPU recoveries. Read once per process.
static BOOL
bg_prefix_listed(NSString *path, NSString *fname, NSArray **cache, dispatch_once_t *once)
{
   dispatch_once(once, ^{
      NSString *t = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:NULL];
      NSMutableArray *a = [NSMutableArray array];
      for (NSString *l in [t componentsSeparatedByString:@"\n"]) {
         NSString *w = [l stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
         if (w.length && ![w hasPrefix:@"#"]) [a addObject:w];
      }
      *cache = [a copy];
   });
   for (NSString *p in *cache)
      if (fname && [fname hasPrefix:p]) return YES;
   return NO;
}

static BOOL
bg_refused(NSString *fname)
{
   static NSArray *prefixes;
   static dispatch_once_t once;
   return bg_prefix_listed(@"/private/var/tmp/bgmtl.refuse", fname, &prefixes, &once);
}

// /private/var/tmp/bgmtl.dumpair: the same kind of list; those pipelines' AIR (with the linked functions) goes to
// /private/var/tmp/bgmtl-refused/ too, but they are built and drawn as usual (the AIR RenderBox specializes at run time)
static BOOL
bg_dump_listed(NSString *fname)
{
   static NSArray *prefixes;
   static dispatch_once_t once;
   return bg_prefix_listed(@"/private/var/tmp/bgmtl.dumpair", fname, &prefixes, &once);
}

static void
bg_refuse_dump(NSData *vair, NSString *vname, NSData *fair, NSString *fname, const struct chl_linked_function *lk, unsigned nvl, unsigned nfl)
{
   NSString *dir = @"/private/var/tmp/bgmtl-refused";
   mkdir(dir.UTF8String, 01777);
   chmod(dir.UTF8String, 01777);
   NSString *stem = [dir stringByAppendingFormat:@"/%@+%@", vname, fname];
   if (access([stem stringByAppendingString:@".f.bc"].UTF8String, F_OK) == 0) return;   // one copy of each pipeline
   [vair writeToFile:[stem stringByAppendingString:@".v.bc"] atomically:NO];
   [fair writeToFile:[stem stringByAppendingString:@".f.bc"] atomically:NO];
   for (unsigned i = 0; i < nvl + nfl; i++)
      [[NSData dataWithBytes:lk[i].bitcode length:lk[i].size]
         writeToFile:[stem stringByAppendingFormat:@".%s.%s.bc", i < nvl ? "lv" : "lf", lk[i].name ? lk[i].name : "linked"] atomically:NO];
}

// BGMTL_TRACE=1 or /private/var/tmp/bgmtl.trace (or bgmtl.tracing): one line per command buffer, render pass, pipeline build and commit
// (what a client such as CoreAnimation sends)
static int
bg_trace(void)
{
   // re-read at most once a second, so a long-lived process (backboardd, which must not be restarted by hand) can be traced
   // for a few seconds by creating and removing the file
   static int t = -1, env = -1;
   static time_t next;
   if (env < 0) env = getenv("BGMTL_TRACE") != NULL;
   if (env) return 1;
   time_t now = time(NULL);
   // bgmtl.tracing: the same switch under a name the kernel's sweetdenypath=bgmtl.trace (the guard boot images carry since a
   // stale trace switch crash-looped backboardd) does not hide. Remove it after use like the other one.
   if (t < 0 || now >= next) {
      t = access("/private/var/tmp/bgmtl.trace", F_OK) == 0 || access("/private/var/tmp/bgmtl.tracing", F_OK) == 0;
      next = now + 1;
   }
   return t;
}
#define BG_TRACE(...) do { if (bg_trace()) bg_log(__VA_ARGS__); } while (0)

// /private/var/tmp/bgmtl.xr10log: into bgmtl.log, one line per two-plane XR10 surface wrapped, written back and decoded (surface
// ID, texture, seed, texels with alpha), and per IOSurface texture refused. Its own switch: the trace's can be hidden by the
// kernel's sweetdenypath (and traces a lot more).
static int
bg_xr10log(void)
{
   static int t = -1;
   static time_t next;
   time_t now = time(NULL);
   if (t < 0 || now >= next) { t = access("/private/var/tmp/bgmtl.xr10log", F_OK) == 0; next = now + 1; }
   return t;
}

void bgmtl_note_line(const char *line);
// one bgmtl.xr10log line into bgmtl.log (bg_log only reaches stderr and the trace file, which daemons do not have)
static void __attribute__((format(printf, 1, 2)))
bg_xr10_note(const char *fmt, ...)
{
   char line[320];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(line, sizeof(line), fmt, ap);
   va_end(ap);
   bgmtl_note_line(line);
}

// texels of a BGRA8 image whose alpha is not 0
static unsigned
bg_alpha_count(const uint8_t *px, size_t n)
{
   unsigned c = 0;
   for (size_t i = 0; i < n; i++) c += px[4 * i + 3] != 0;
   return c;
}


// counts command buffers: a buffer-backed texture is filled from its buffer at most once per command buffer (BGMTLTexture _linSeq)
static unsigned long g_bg_commit_seq, g_bg_cb_seq;
// Chlorophyll: what a process holds in Metal objects, for the idle-trim line (bgmtl.perf): textures with memory of their
// own and buffers (count, bytes; texture bytes estimated from the shape), heap allocations, makeAliasable calls and
// setPurgeableState requests by state (0..4: KeepCurrent is 1, NonVolatile 2, Volatile 3, Empty 4)
static long g_bg_tex_n, g_bg_tex_bytes, g_bg_buf_n, g_bg_buf_bytes;
static unsigned long g_bg_heap_allocs, g_bg_aliasable, g_bg_purge[5];
static long
bg_tex_bytes(const struct chl_texture_desc *s)
{
   long layers = s->depth > s->array_length ? s->depth : s->array_length;
   long b = (long)s->width * s->height * (layers ? layers : 1) * 4 * (s->sample_count ? s->sample_count : 1);
   return s->mip_levels > 1 ? b + b / 3 : b;
}
static void bg_count_purge(NSUInteger state) { __atomic_add_fetch(&g_bg_purge[state < 5 ? state : 0], 1, __ATOMIC_RELAXED); }

// Chlorophyll: the compositor's own pool of released plain textures. CoreAnimation makes its offscreen targets (blur pyramid,
// glass passes, full-screen RGBA16F copies) anew every frame; a released texture with no IOSurface, view, buffer or caller
// memory behind it is kept here, and a new texture of exactly the same shape takes it back. Its contents are undefined, as
// a new Metal texture's are; a command buffer that still uses it holds its own reference (chl retains what it records).
// Off unless BGMTL_TEXPOOL_MB is set (the inject sets it for backboardd); entries older than 5 s go.
#define BG_TEXPOOL_SLOTS 64
static struct { struct chl_device *dev; struct chl_texture *tex; struct chl_texture_desc desc; long bytes; uint64_t t; } g_texpool[BG_TEXPOOL_SLOTS];
static long g_texpool_bytes;
static unsigned long g_texpool_hits, g_texpool_misses;
static os_unfair_lock g_texpool_lock = OS_UNFAIR_LOCK_INIT;

static long
bg_texpool_cap(void)
{
   static long cap = -1;
   if (cap < 0) cap = getenv("BGMTL_TEXPOOL_MB") ? atol(getenv("BGMTL_TEXPOOL_MB")) << 20 : 0;
   return cap;
}

static void
bg_texpool_drop(unsigned k)
{
   chl_texture_release(g_texpool[k].tex);
   g_texpool_bytes -= g_texpool[k].bytes;
   memset(&g_texpool[k], 0, sizeof(g_texpool[k]));
}

static void
bg_texpool_expire(uint64_t now)
{
   for (unsigned k = 0; k < BG_TEXPOOL_SLOTS; k++)
      if (g_texpool[k].tex && now - g_texpool[k].t > 5000000000ull) bg_texpool_drop(k);
}

static struct chl_texture *
bg_texpool_take(struct chl_device *dev, const struct chl_texture_desc *desc)
{
   if (!bg_texpool_cap()) return NULL;
   struct chl_texture *t = NULL;
   os_unfair_lock_lock(&g_texpool_lock);
   bg_texpool_expire(clock_gettime_nsec_np(CLOCK_MONOTONIC));
   for (unsigned k = 0; k < BG_TEXPOOL_SLOTS && !t; k++)
      if (g_texpool[k].tex && g_texpool[k].dev == dev && !memcmp(&g_texpool[k].desc, desc, sizeof(*desc))) {
         t = g_texpool[k].tex;
         g_texpool_bytes -= g_texpool[k].bytes;
         memset(&g_texpool[k], 0, sizeof(g_texpool[k]));
      }
   if (t) g_texpool_hits++; else g_texpool_misses++;
   os_unfair_lock_unlock(&g_texpool_lock);
   return t;
}

// keep a released texture; NO: the caller releases it
static BOOL
bg_texpool_put(struct chl_device *dev, struct chl_texture *tex, const struct chl_texture_desc *desc)
{
   long cap = bg_texpool_cap(), bytes = bg_tex_bytes(desc) * 2;   // the estimate counts 4 bytes a texel; RGBA16F has 8
   if (!cap || bytes > cap / 4) return NO;
   uint64_t now = clock_gettime_nsec_np(CLOCK_MONOTONIC);
   os_unfair_lock_lock(&g_texpool_lock);
   bg_texpool_expire(now);
   int free_k = -1;
   for (;;) {
      int old_k = -1;
      free_k = -1;
      for (unsigned k = 0; k < BG_TEXPOOL_SLOTS; k++) {
         if (!g_texpool[k].tex) { if (free_k < 0) free_k = (int)k; }
         else if (old_k < 0 || g_texpool[k].t < g_texpool[old_k].t) old_k = (int)k;
      }
      if (free_k >= 0 && g_texpool_bytes + bytes <= cap) break;
      if (old_k < 0) { os_unfair_lock_unlock(&g_texpool_lock); return NO; }
      bg_texpool_drop((unsigned)old_k);   // the oldest makes room
   }
   g_texpool[free_k].dev = dev;
   g_texpool[free_k].tex = tex;
   g_texpool[free_k].desc = *desc;
   g_texpool[free_k].bytes = bytes;
   g_texpool[free_k].t = now;
   g_texpool_bytes += bytes;
   os_unfair_lock_unlock(&g_texpool_lock);
   return YES;
}

#define BG_MAX_BINDINGS 31
#define BG_MAX_TEX 32

// provided by the injection dylib (gpu/metal/inject), absent in the test tools
__attribute__((weak)) void bgmtl_note_failure(const char *why) { (void)why; }   // overridden by the injection dylib
__attribute__((weak)) void bgmtl_note_unimplemented(const char *sel) { (void)sel; }
__attribute__((weak)) void bgmtl_note_completed(void) {}   // a command buffer finished without error (the injection's dead-man reset)
__attribute__((weak)) void bgmtl_note_committed(void) {}   // a command buffer was committed (starts the injection's dead-man clock)
__attribute__((weak)) void bgmtl_note_line(const char *line) { (void)line; }   // one diagnostic line, logged by the injection dylib
__attribute__((weak)) void bgmtl_note_present(const char *line) { (void)line; }   // per-present diagnostics, logged (rate-limited) by the injection dylib

// Chlorophyll: where a slow start goes (/private/var/tmp/bgmtl.perf at process start): pipeline builds and single commits that
// take long. Totals are logged every 16 pipelines and any single wait over 100 ms is logged at once.
static int
bg_perf_on(void)
{
   static int on = -1;
   if (on < 0) on = access("/private/var/tmp/bgmtl.perf", F_OK) == 0;
   return on;
}
static void
bg_perf_slow(const char *what, const char *name, uint64_t ns)
{
   if (!bg_perf_on() || ns < 100000000ull) return;
   char line[256];
   snprintf(line, sizeof(line), "slow: %s %s: %.1f ms", what, name ? name : "", ns / 1e6);
   bgmtl_note_line(line);
}

@class BGMTLDevice;
@class BGMTLComputePipeline;
@class BGMTLCommandBuffer;
@class BGMTLTexture;
@class BGMTLBuffer;

// Unknown selectors on the queue, command buffers, encoders and resources: logged once each and answered with zero / a no-op,
// as the device does, so a client (CoreAnimation) shows every gap in one run instead of dying on the first. The signature comes
// from Metal's protocols (MTL*), where every such selector is declared.
static NSMethodSignature *
bg_proto_signature(SEL sel)
{
   unsigned n = 0;
   Protocol *__unsafe_unretained *all = objc_copyProtocolList(&n);
   NSMethodSignature *sig = nil;
   for (unsigned i = 0; i < n && !sig; i++) {
      if (strncmp(protocol_getName(all[i]), "MTL", 3) != 0) continue;
      for (int req = 0; req < 2 && !sig; req++) {
         struct objc_method_description d = protocol_getMethodDescription(all[i], sel, req, YES);
         if (d.name && d.types) sig = [NSMethodSignature signatureWithObjCTypes:d.types];
      }
   }
   free(all);
   if (!sig) {     // SPI declared nowhere at run time (setResponsibleTaskIDs:count:): void, one pointer-sized argument per colon
      char types[64] = "v@:";
      for (const char *c = sel_getName(sel); *c && strlen(types) < sizeof(types) - 1; c++)
         if (*c == ':') strcat(types, "Q");
      sig = [NSMethodSignature signatureWithObjCTypes:types];
   }
   return sig;
}

// Chlorophyll: a forwarded selector's signature, worked out once per selector. bg_proto_signature walks every protocol in
// the process (objc_copyProtocolList), and CoreAnimation sends a few unimplemented selectors every frame
// (setResponsibleProcess: on each command buffer): with that walk per call, forwarding was 57 % of backboardd's CPU
// while swiping.
static NSMethodSignature *
bg_cached_signature(SEL sel, NSMethodSignature *(^make)(void))
{
   static CFMutableDictionaryRef cache;   // SEL -> NSMethodSignature (retained)
   static os_unfair_lock lk = OS_UNFAIR_LOCK_INIT;
   os_unfair_lock_lock(&lk);
   if (!cache) cache = CFDictionaryCreateMutable(NULL, 0, NULL, &kCFTypeDictionaryValueCallBacks);
   NSMethodSignature *sig = (NSMethodSignature *)CFDictionaryGetValue(cache, sel);
   os_unfair_lock_unlock(&lk);
   if (sig) return sig;
   sig = make();
   if (!sig) return nil;
   os_unfair_lock_lock(&lk);
   CFDictionarySetValue(cache, sel, sig);
   os_unfair_lock_unlock(&lk);
   return sig;
}

// The answer to a forwarded selector whose result comes back in x0 (nothing, an integer, a BOOL, an object or a pointer):
// zero. Installed on the class after the first forwarded call, so later calls are ordinary method calls and skip the
// NSInvocation altogether; selectors returning floats or structs keep going through forwardInvocation:.
static void *bg_zero_imp(id self, SEL _cmd) { (void)self; (void)_cmd; return NULL; }

static void
bg_install_zero(Class c, SEL sel, NSMethodSignature *sig)
{
   const char *r = [sig methodReturnType];
   while (*r && strchr("rnNoORV", *r)) r++;           // type qualifiers (const, in, out, oneway ...)
   if (!*r || strchr("vcCsSiIlLqQB@#:*^", *r) == NULL) return;
   char types[8] = "^v@:";                            // a pointer result
   if (*r != '^' && *r != '*') snprintf(types, sizeof(types), "%c@:", *r);
   class_addMethod(c, sel, (IMP)bg_zero_imp, types);
}

static NSMethodSignature *
bg_fwd_sig(id self, SEL _cmd, SEL sel)
{
   (void)_cmd;
   Method m = class_getInstanceMethod(object_getClass(self), sel);     // implemented somewhere up the chain
   if (m) return [NSMethodSignature signatureWithObjCTypes:method_getTypeEncoding(m)];
   // the base classes' own -methodSignatureForSelector: is not asked: it may throw
   return bg_cached_signature(sel, ^NSMethodSignature *(void) { return bg_proto_signature(sel); });
}

static void
bg_fwd_inv(id self, SEL _cmd, NSInvocation *inv)
{
   (void)_cmd;
   static NSMutableSet *seen;
   static os_unfair_lock lk = OS_UNFAIR_LOCK_INIT;
   NSString *key = [NSString stringWithFormat:@"%s %s", class_getName(object_getClass(self)), sel_getName([inv selector])];
   os_unfair_lock_lock(&lk);
   if (!seen) seen = [[NSMutableSet alloc] init];
   BOOL fresh = ![seen containsObject:key];
   if (fresh) [seen addObject:key];
   os_unfair_lock_unlock(&lk);
   if (fresh) {
      BG_LOG("%s: unimplemented, answered with zero\n", key.UTF8String);
      bgmtl_note_unimplemented(sel_getName([inv selector]));
      bg_install_zero(object_getClass(self), [inv selector], [inv methodSignature]);
   }
   NSUInteger len = [[inv methodSignature] methodReturnLength];
   if (len) {
      void *z = calloc(1, len);
      [inv setReturnValue:z];
      free(z);
   }
}

static void
bg_install_fallback(Class c)
{
   class_addMethod(c, @selector(methodSignatureForSelector:), (IMP)bg_fwd_sig, "@@::");
   class_addMethod(c, @selector(forwardInvocation:), (IMP)bg_fwd_inv, "v@:@");
}

// ---- device ----------------------------------------------------------------------------------------------------------
@interface BGMTLDevice : _MTLDevice {
   struct chl_device *_dev;
   char _why[512];
   BOOL _assertions;
}
- (struct chl_device *)chl;
- (NSString *)bgWhy;
- (id)bgFail:(NSError **)error with:(NSString *)reason;
@end

@interface BGMTLBuffer : _MTLResource {
   BGMTLDevice *_owner;
   struct chl_buffer *_buf;
   void *_contents;
   NSUInteger _length;
   void *_mapped;                               // the GPU memory when _contents is a caller's (newBufferWithBytesNoCopy)
   void (^_deallocator)(void *, NSUInteger);
}
- (instancetype)initWithBGDevice:(BGMTLDevice *)device length:(NSUInteger)length;
- (instancetype)initWithBGDevice:(BGMTLDevice *)device bytesNoCopy:(void *)p length:(NSUInteger)length deallocator:(void (^)(void *, NSUInteger))dealloc;
- (struct chl_buffer *)chlBuffer;
- (void)bgRefresh;   // a no-copy buffer: the caller's bytes into the GPU memory (at commit)
- (BOOL)bgIsNoCopy;
- (void *)contents;
@end

@interface BGMTLComputePipeline : _MTLComputePipelineState {
 @public
   BGMTLDevice *_owner;
   struct chl_compute_pipeline *_p;
   struct chl_pipeline_info _info;
}
@end

@interface BGMTLCommandQueue : _MTLCommandQueue {
   BGMTLDevice *_owner;
}
- (instancetype)initWithBGDevice:(BGMTLDevice *)device descriptor:(id)descriptor;
@end

@interface BGMTLCommandBuffer : _MTLCommandBuffer {
 @public
   BGMTLCommandQueue *_queue;
   struct chl_cmdbuf *_ccb;
   unsigned long _seq;                                     // g_bg_cb_seq at creation: once-per-command-buffer refreshes
   NSError *_error;
   NSInteger _status;
   NSMutableArray *_completed, *_scheduled, *_presents;   // handlers; drawables to present once the work has completed
   dispatch_group_t _done;                                 // left when the completion has run: waitUntilCompleted
   int _slotReturned;                                      // the base queue's in-flight slot given back (bg_return_slot)
   NSMutableSet *_linWritten;                              // buffer-backed textures this command buffer writes
   NSMutableSet *_written;                                 // textures rendered or resolved into (IOSurface sync, XR10 write-back)
   NSMutableSet *_noCopy;                                  // no-copy buffers it binds (refreshed at commit)
   unsigned _draws;
   uint64_t _t_first;                                      // the first encoder (encode time for bgmtl.perf)
}
- (instancetype)initWithBGQueue:(BGMTLCommandQueue *)queue;
- (struct chl_cmdbuf *)bgChl;                         // the Vulkan command buffer, noting when encoding began
- (void)bgWrote:(BGMTLTexture *)t;                    // a texture this command buffer renders or copies into
- (void)bgUses:(BGMTLBuffer *)b;                      // a buffer it binds (a no-copy one is refreshed at commit)
- (void)bgSkip:(const char *)what why:(const char *)why;
- (void)bgFailWith:(NSString *)why code:(NSInteger)code;
- (void)bgCompleteFrom:(uint64_t)t_begin;
@end

@interface BGMTLComputeEncoder : _MTLCommandEncoder {
   BGMTLCommandBuffer *_cb;
   BGMTLComputePipeline *_pipeline;
}
- (instancetype)initWithBGCommandBuffer:(BGMTLCommandBuffer *)cb;
@end

@interface BGMTLBlitEncoder : _MTLCommandEncoder {
   BGMTLCommandBuffer *_cb;
}
- (instancetype)initWithBGCommandBuffer:(BGMTLCommandBuffer *)cb;
@end

// MTLHeap (QuartzCore 26 asks for one every frame and falls back when it gets nil): every allocation is an ordinary one of the
// device's, so nothing aliases (more memory, the same pictures); /private/var/tmp/bgmtl.noheap makes the device answer nil again.
@interface BGMTLHeap : NSObject {
   @public
   id _owner;
   NSUInteger _size, _storage, _cache, _hazard, _type;
   NSString *_label;
}
@end

@interface BGMTLTexture : _MTLResource {
   BGMTLDevice *_owner;
   struct chl_texture *_tex;
   struct chl_texture_desc _desc;
   CFTypeRef _surface;   // retained IOSurface, when the memory is one
   NSUInteger _plane;
   @public
   id _linSrc;           // newLinearTextureWithDescriptor: the buffer whose bytes this texture shows (retained)
   NSUInteger _linOff, _linBpr;
   unsigned long _linSeq; // the command buffer (g_bg_cb_seq) it was last filled from its buffer for: once per command buffer
   int _linPending;       // command buffers in flight that write it: no fill from the buffer until they wrote back
   const void *_ncBytes; // newTextureWithBytesNoCopy: the caller's memory, refreshed from at each use like _linSrc
   BOOL _surfCopy;       // an IOSurface the GPU could not import: own memory, refreshed from the surface at each use
   BOOL _xr2p;           // ... a two-plane RGB10A8_2P_XR10 one ('b3a8': XR10 RGB + A8 alpha), decoded into BGRA8 at each refresh
   uint32_t _surfSeed;   // the surface's seed at the last refresh (an unchanged surface is not copied again)
   BOOL _surfFresh;      // _surfSeed is valid: the surface was decoded or written back at least once (a seed may be 0)
   uint32_t _xrUses;     // uses since the last XR10 decode (bgmtl.xr10log)
   NSUInteger _ncLength;
   void (^_ncDealloc)(void *, NSUInteger);
   BGMTLTexture *_parent;  // a texture view (newTextureViewWith...): the texture whose memory it shows (retained)
   NSUInteger _parentLevel, _parentSlice;
   MTLTextureSwizzleChannels _swizzle;
}
- (instancetype)initWithBGDevice:(BGMTLDevice *)device descriptor:(MTLTextureDescriptor *)d;
- (instancetype)initWithBGDevice:(BGMTLDevice *)device descriptor:(MTLTextureDescriptor *)d iosurface:(void *)surface plane:(NSUInteger)plane;
- (struct chl_texture *)chlTexture;
- (struct chl_texture *)chlTextureFor:(BGMTLCommandBuffer *)cb;   // refreshed from whatever its pixels come from, once per command buffer
- (BGMTLTexture *)bgRoot;
- (void *)iosurface;
- (void)bgLinearPending:(int)delta;
- (void)bgLinearWriteBack;
- (BOOL)bgXR10WriteBack;
@end

@interface BGMTLSampler : _MTLSamplerState {
   BGMTLDevice *_owner;
   struct chl_sampler *_smp;
}
- (instancetype)initWithBGDevice:(BGMTLDevice *)device descriptor:(MTLSamplerDescriptor *)d;
- (struct chl_sampler *)chlSampler;
@end

@interface BGMTLDepthStencil : _MTLDepthStencilState {
   BGMTLDevice *_owner;
   struct chl_depth_stencil *_ds;
   BOOL _readsDepth, _writesDepth, _readsStencil, _writesStencil;
}
- (instancetype)initWithBGDevice:(BGMTLDevice *)device descriptor:(MTLDepthStencilDescriptor *)d;
- (struct chl_depth_stencil *)chlState;
@end

@interface BGMTLRenderPipeline : _MTLRenderPipelineState {
 @public
   BGMTLDevice *_owner;
   struct chl_render_pipeline *_p;
   struct chl_pipeline_info _info;
   char _names[96];
}
@end

@interface BGMTLRenderEncoder : _MTLCommandEncoder {
   BGMTLCommandBuffer *_cb;
   BGMTLRenderPipeline *_pipeline;
   id _vbuf[BG_MAX_BINDINGS], _fbuf[BG_MAX_BINDINGS];   // what each slot holds (setVertexBufferOffset: moves only the offset)
   NSUInteger _cull, _winding;
   BOOL _open;
}
- (instancetype)initWithBGCommandBuffer:(BGMTLCommandBuffer *)cb descriptor:(MTLRenderPassDescriptor *)d;
@end

static void
ensure_protocol(Class cls, const char *proto)
{
   Protocol *p = objc_getProtocol(proto);
   if (!p) {
      BG_LOG("protocol %s is not in this process\n", proto);
      return;
   }
   if (!class_conformsToProtocol(cls, p) && !class_addProtocol(cls, p))
      BG_LOG("class_addProtocol(%s) refused for %s\n", proto, class_getName(cls));
}

@implementation BGMTLDevice

+ (void)load
{
   // MTLDeviceSPI is the private protocol Metal's registration asserts on; we never register, but a subclass that claims
   // to be a device has to conform, and the whole SPI surface is answered by _MTLDevice's own defaults or by methods below.
   ensure_protocol(self, "MTLDeviceSPI");
   ensure_protocol(self, "MTLDevice");
   // by class object, not by name: Chlorophyll.bundle builds this file with every class renamed (CHL_...), and a name
   // lookup there would patch the inject's copy instead of its own
   Class fb[] = { [BGMTLCommandQueue class], [BGMTLCommandBuffer class], [BGMTLRenderEncoder class], [BGMTLComputeEncoder class],
                  [BGMTLBlitEncoder class], [BGMTLBuffer class], [BGMTLTexture class], [BGMTLRenderPipeline class],
                  [BGMTLComputePipeline class], [BGMTLSampler class], [BGMTLDepthStencil class], [BGMTLHeap class] };
   for (unsigned i = 0; i < sizeof(fb) / sizeof(fb[0]); i++) {
      if (fb[i]) bg_install_fallback(fb[i]);
   }
}

- (instancetype)initWithAcceleratorPort:(uint32_t)port
{
   (void)port;
   self = [super init];
   if (!self) return nil;
   // Mesa's shader disk cache (ir3 binaries, keyed by the shader and libbgair's LC_UUID): one folder per user, since
   // backboardd (mobile), iconservicesagent (_iconservices) and the poster extensions cannot share files. This system has
   // no per-user cache dir (no dirhelper), so it lives in /private/var/tmp; /private/var/tmp/bgmtl.nocache turns it off.
   if (!getenv("MESA_SHADER_CACHE_DIR")) {
      if (access("/private/var/tmp/bgmtl.nocache", F_OK) == 0) {
         setenv("MESA_SHADER_CACHE_DISABLE", "true", 1);
      } else {
         char dir[96];
         mkdir("/private/var/tmp/chlorophyll-cache", 01777);
         chmod("/private/var/tmp/chlorophyll-cache", 01777);   // sticky and world-writable, like /tmp: every user makes its own
         snprintf(dir, sizeof(dir), "/private/var/tmp/chlorophyll-cache/%u", (unsigned)getuid());
         mkdir(dir, 0700);
         setenv("MESA_SHADER_CACHE_DIR", dir, 1);
      }
   }
   _dev = chl_device_open(_why, sizeof(_why));
   BG_LOG("chl_device_open: %s\n", _dev ? chl_device_name(_dev) : _why);
   return self;
}

- (void)dealloc
{
   if (_dev) chl_device_close(_dev);
   [super dealloc];
}

- (struct chl_device *)chl { return _dev; }

// Metal's private -targetDeviceInfo: a pointer to {u64, u32, const char *name at +0x10}, read when a library is loaded or
// compiled (Metal+0x24888, iOS 18.7.8: it dereferences +0x10 without a check). The fallback forwarding answered NULL and
// SpringBoard's Siri orb (SiriUICore) crashed there. The values are what Metal builds itself for a device that does not
// answer: zeros and the device name.
- (const void *)targetDeviceInfo
{
   static const struct { uint64_t a; uint32_t b; const char *name; } info = { 0, 0, "Chlorophyll" };
   return &info;
}
- (NSString *)bgWhy { return [NSString stringWithUTF8String:_why]; }
- (id)bgFail:(NSError **)error with:(NSString *)reason
{
   BG_LOG("refusing: %s\n", [reason UTF8String]);
   if (error) *error = [NSError errorWithDomain:@"BGMTL" code:1 userInfo:@{ NSLocalizedDescriptionKey : reason }];
   return nil;
}

// the registration trio of the hackios measurements (iOS 16.5); here they are what _MTLDevice's own tables read
- (NSUInteger)featureProfile { return 1; }
- (uint64_t)maxBufferLength { return 256ULL << 20; }
- (BOOL)supportsVertexAmplificationCount:(NSUInteger)count { return count <= 1; }

- (NSString *)name { return _dev ? [NSString stringWithFormat:@"%s (Bergamota)", chl_device_name(_dev)] : @"Adreno (Bergamota, no GPU)"; }
- (BOOL)hasUnifiedMemory { return YES; }
- (BOOL)isLowPower { return NO; }
- (BOOL)isHeadless { return NO; }
- (BOOL)isRemovable { return NO; }
- (uint64_t)recommendedMaxWorkingSetSize { return 512ULL << 20; }
- (uint64_t)currentAllocatedSize { return 0; }
- (MTLSize)maxThreadsPerThreadgroup { return MTLSizeMake(1024, 1024, 64); }
- (NSUInteger)maxThreadgroupMemoryLength { return 16384; }

// The device is a stand-in for a Metal device in a process that has none, so a client may ask it something nobody has implemented yet.
// Metal's stock behaviour for an unknown selector is to abort the process; for a selector that IS in the MTLDevice / MTLDeviceSPI
// protocols we answer zero (nil / NO / 0) and tell the injection dylib, which logs it. Anything outside the protocols still aborts.
- (uint64_t)registryID { return 0x10000b630ULL; }
- (BOOL)supportsSampleCount:(NSUInteger)count { return count == 1 || count == 4; }
- (BOOL)isDepth24Stencil8PixelFormatSupported { return NO; }
- (NSUInteger)minimumTextureBufferAlignmentForPixelFormat:(NSUInteger)format { (void)format; return 64; }
// Linear textures (and the buffers clients size for them, CoreGraphics' CGImageTextureData among them) have rows
// aligned to 64 bytes here (freedreno a6xx pitch alignment). The fallback answered 0 and SpringBoard's image upload
// wrote past an ImageIO buffer into its guard page (KERN_PROTECTION_FAILURE in CGImageTextureDataGetOffsetWithBlock).
- (NSUInteger)minimumLinearTextureAlignmentForPixelFormat:(NSUInteger)format { (void)format; return 64; }
// Heap sizing queries: page-aligned, and for textures an upper bound (16 bytes a texel, every mip level), never zero.
- (MTLSizeAndAlign)heapBufferSizeAndAlignWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
   (void)options;
   return (MTLSizeAndAlign){ (length + 0x3fff) & ~(NSUInteger)0x3fff, 0x4000 };
}
- (MTLSizeAndAlign)heapTextureSizeAndAlignWithDescriptor:(MTLTextureDescriptor *)d
{
   NSUInteger w = d.width ? d.width : 1, h = d.height ? d.height : 1, z = d.depth ? d.depth : 1;
   NSUInteger layers = (d.arrayLength ? d.arrayLength : 1) * (d.textureType == MTLTextureTypeCube || d.textureType == MTLTextureTypeCubeArray ? 6 : 1);
   NSUInteger total = 0;
   for (NSUInteger l = 0; l < (d.mipmapLevelCount ? d.mipmapLevelCount : 1); l++) {
      NSUInteger lw = w >> l ? w >> l : 1, lh = h >> l ? h >> l : 1, lz = z >> l ? z >> l : 1;
      total += ((lw * 16 + 63) & ~(NSUInteger)63) * lh * lz * layers * (d.sampleCount ? d.sampleCount : 1);
   }
   return (MTLSizeAndAlign){ (total + 0x3fff) & ~(NSUInteger)0x3fff, 0x4000 };
}
- (uint64_t)sharedMemorySize { return 2048ULL << 20; }
- (uint64_t)dedicatedMemorySize { return 0; }

- (NSMethodSignature *)methodSignatureForSelector:(SEL)sel
{
   NSMethodSignature *sig = [super methodSignatureForSelector:sel];
   if (sig) return sig;
   return bg_cached_signature(sel, ^NSMethodSignature *(void) {
      const char *names[] = { "MTLDevice", "MTLDeviceSPI" };
      for (unsigned i = 0; i < 2; i++) {
         Protocol *pr = objc_getProtocol(names[i]);
         if (!pr) continue;
         for (int req = 0; req < 2; req++) {
            struct objc_method_description d = protocol_getMethodDescription(pr, sel, req, YES);
            if (d.name && d.types) return [NSMethodSignature signatureWithObjCTypes:d.types];
         }
      }
      return (NSMethodSignature *)nil;
   });
}

- (void)forwardInvocation:(NSInvocation *)inv
{
   SEL sel = [inv selector];
   BG_LOG("device: unimplemented %s answered with zero\n", sel_getName(sel));
   bgmtl_note_unimplemented(sel_getName(sel));
   bg_install_zero(object_getClass(self), sel, [inv methodSignature]);
   NSUInteger len = [[inv methodSignature] methodReturnLength];
   if (len) {
      void *z = calloc(1, len);
      [inv setReturnValue:z];
      free(z);
   }
}

// traced pass-throughs of the capability queries CoreAnimation makes (what it chooses its renderer conventions from)
- (BOOL)supportsFamily:(NSInteger)family
{
   // Apple1..Apple3 and Common1..Common2, as UniXNU's native device answers (Apple4+ promise tile shaders and imageblocks,
   // which Chlorophyll does not run yet); everything else as _MTLDevice's tables say. BGMTL_FAMILY=N overrides the Apple ceiling.
   static long apple_max = -1;
   if (apple_max < 0) apple_max = getenv("BGMTL_FAMILY") ? atol(getenv("BGMTL_FAMILY")) : 3;
   BOOL r;
   if (family >= 1001 && family <= 1009) r = family - 1000 <= apple_max;          // MTLGPUFamilyApple1..9
   else if (family == 3001 || family == 3002) r = YES;                            // MTLGPUFamilyCommon1, Common2
   else if (family == 5002) r = NO;                                               // MTLGPUFamilyMetal4 (iOS 26): see below
   else r =((BOOL (*)(struct objc_super *, SEL, NSInteger))objc_msgSendSuper)(&(struct objc_super){ self, [_MTLDevice class] }, _cmd, family);
   BG_TRACE("supportsFamily %ld -> %d\n", (long)family, r);
   return r;
}
- (BOOL)supportsFeatureSet:(NSUInteger)set
{
   BOOL r = ((BOOL (*)(struct objc_super *, SEL, NSUInteger))objc_msgSendSuper)(&(struct objc_super){ self, [_MTLDevice class] }, _cmd, set);
   BG_TRACE("supportsFeatureSet %lu -> %d\n", (unsigned long)set, r);
   return r;
}

// No extended-range (XR) formats: the Adreno renders BGRA8, R8 and RGBA16Float natively and no 10-bit XR layout. With YES (the
// default tables of the profile) QuartzCore's render server built its deep-colour pipelines in RGB10A8_2P_XR10 (550), which
// Chlorophyll refuses, and aborted "Metal failed to build render pipeline" (first backboardd run). With NO it renders the SDR
// path only (air2ir3's FORMATS.md: WindowServer decides the same way).
- (BOOL)supportsExtendedXR10Formats { BG_TRACE("supportsExtendedXR10Formats -> NO\n"); return NO; }
- (BOOL)supportsExtendedColors { BG_TRACE("supportsExtendedColors -> NO\n"); return NO; }

// iOS 26: no Metal 4, tensors, machine learning encoders, texture view pools or placement sparse. iOS 26's _MTLDevice gained
// these queries and factories (74 methods over 18.7.8, read from the 23G90 shared cache); their answers come from the feature
// tables an Apple GPU driver fills in, which this device does not, and a Metal 4 queue made by the base class would not
// submit to this GPU. QuartzCore 26 references MTL4CommandQueue (signalEvent:value:/waitForEvent:value:), so every query
// says NO and every factory returns nil here. iOS 18 never sends these selectors.
- (BOOL)supportsMTL4CommandQueue { BG_TRACE("supportsMTL4CommandQueue -> NO\n"); return NO; }
- (BOOL)supportsMTL4CommandAllocator { return NO; }
- (BOOL)supportsMTL4Compiler { return NO; }
- (BOOL)supportsMTL4ComputeCommandEncoder { return NO; }
- (BOOL)supportsMTL4RenderCommandEncoder { return NO; }
- (BOOL)supportsMTL4Counters { return NO; }
- (BOOL)supportsMTL4LateBoundRenderTargets { return NO; }
- (BOOL)supportsMTL4PSOSpecialization { return NO; }
- (BOOL)supportsMTL4PlacementSparse { return NO; }
- (BOOL)supportsPlacementSparse { return NO; }
- (BOOL)supportsTensors { return NO; }
- (BOOL)supportsMachineLearningCommandEncoders { return NO; }
- (BOOL)supportsMTLTextureViewPools { return NO; }
- (BOOL)supportsTextureAccessPatterns { return NO; }
- (BOOL)validateMetal4SupportWithError:(NSError **)error
{
   [self bgFail:error with:@"Metal 4 is not supported on this device"];
   return NO;
}
- (id)newMTL4CommandQueue { BG_TRACE("newMTL4CommandQueue -> nil\n"); return nil; }
- (id)newMTL4CommandQueueWithDescriptor:(id)descriptor error:(NSError **)error
{
   (void)descriptor;
   BG_TRACE("newMTL4CommandQueueWithDescriptor -> nil\n");
   return [self bgFail:error with:@"Metal 4 command queues are not supported on this device"];
}
- (id)newCommandQueue4 { return nil; }
- (id)newCommandAllocator { return nil; }
- (id)newCommandAllocatorWithDescriptor:(id)descriptor error:(NSError **)error
{
   (void)descriptor;
   return [self bgFail:error with:@"Metal 4 command allocators are not supported on this device"];
}
- (id)newCompilerWithDescriptor:(id)descriptor error:(NSError **)error
{
   (void)descriptor;
   return [self bgFail:error with:@"Metal 4 compilers are not supported on this device"];
}
- (id)newArgumentTableWithDescriptor:(id)descriptor error:(NSError **)error
{
   (void)descriptor;
   return [self bgFail:error with:@"Metal 4 argument tables are not supported on this device"];
}
- (id)newCounterHeapWithDescriptor:(id)descriptor error:(NSError **)error
{
   (void)descriptor;
   return [self bgFail:error with:@"counter heaps are not supported on this device"];
}
- (id)newTensorWithDescriptor:(id)descriptor error:(NSError **)error
{
   (void)descriptor;
   return [self bgFail:error with:@"tensors are not supported on this device"];
}
- (id)newTextureViewPoolWithDescriptor:(id)descriptor error:(NSError **)error
{
   (void)descriptor;
   return [self bgFail:error with:@"texture view pools are not supported on this device"];
}

- (BOOL)metalAssertionsEnabled { return _assertions; }
- (void)setMetalAssertionsEnabled:(BOOL)enabled { _assertions = enabled; }

- (id)newCommandQueueWithDescriptor:(id)descriptor
{
   id use = descriptor ? [[descriptor retain] autorelease] : [[[MTLCommandQueueDescriptor alloc] init] autorelease];
   return [[BGMTLCommandQueue alloc] initWithBGDevice:self descriptor:use];
}

// -- libraries from files: system metallibs (QuartzCore's default.metallib) are fat wrappers, slice 0 the AIR container
// ("MTLB"), slice 1 precompiled Apple GPU code. _MTLDevice's own URL/bundle loaders pick the slice through MTLLoader by the
// device's target architecture, which this device does not have (assertion "Target device architecture is nil"), so the
// AIR slice is taken here and loaded with -newLibraryWithData:, the path every other library of this device takes --
static NSData *
bg_air_slice(NSData *d)
{
   const uint8_t *b = d.bytes;
   NSUInteger n = d.length;
   if (n >= 4 && memcmp(b, "MTLB", 4) == 0) return d;
   if (n < 8 || OSReadBigInt32(b, 0) != 0xcafebabeu) return nil;
   uint32_t count = OSReadBigInt32(b, 4);
   for (uint32_t i = 0; i < count && 8 + 20 * (NSUInteger)(i + 1) <= n; i++) {
      uint32_t off = OSReadBigInt32(b, 8 + 20 * i + 8), size = OSReadBigInt32(b, 8 + 20 * i + 12);
      if ((NSUInteger)off + size <= n && size >= 4 && memcmp(b + off, "MTLB", 4) == 0)
         return [d subdataWithRange:NSMakeRange(off, size)];
   }
   return nil;
}

- (id)newLibraryWithURL:(NSURL *)url error:(NSError **)error
{
   NSData *d = [NSData dataWithContentsOfURL:url options:NSDataReadingMappedIfSafe error:error];
   NSData *air = d ? bg_air_slice(d) : nil;
   if (!air) {
      BG_LOG("library %s: %s\n", url.path.UTF8String, d ? "no AIR (MTLB) slice" : "unreadable");
      if (error && d)
         *error = [NSError errorWithDomain:@"MTLLibraryErrorDomain" code:1
                                  userInfo:@{ NSLocalizedDescriptionKey : @"no AIR slice in the metallib" }];
      return nil;
   }
   [air retain];
   dispatch_data_t dd = dispatch_data_create(air.bytes, air.length, NULL, ^{ [air release]; });
   id lib = ((id (*)(id, SEL, dispatch_data_t, NSError **))objc_msgSend)(self, @selector(newLibraryWithData:error:), dd, error);
   dispatch_release(dd);
   BG_LOG("library %s: AIR slice %lu bytes -> %s\n", url.path.UTF8String, (unsigned long)air.length, lib ? "loaded" : "refused");
   return lib;
}

- (id)newLibraryWithFile:(NSString *)path error:(NSError **)error
{
   return [self newLibraryWithURL:[NSURL fileURLWithPath:path] error:error];
}

- (id)newDefaultLibraryWithBundle:(NSBundle *)bundle error:(NSError **)error
{
   NSURL *url = [bundle URLForResource:@"default" withExtension:@"metallib"];
   if (!url) {
      if (error)
         *error = [NSError errorWithDomain:@"MTLLibraryErrorDomain" code:1
                                  userInfo:@{ NSLocalizedDescriptionKey : @"no default.metallib in the bundle" }];
      return nil;
   }
   return [self newLibraryWithURL:url error:error];
}

// Binary archives (QuartzCore opens one with a URL: pipelines precompiled for Apple GPUs). Reading it asserts in MTLLoader for
// want of a target architecture, and nothing in it runs on an Adreno: the archive is created empty (no URL), so it only
// serves as the cache object the caller expects; pipelines are built from AIR as always.
- (id)newBinaryArchiveWithDescriptor:(MTLBinaryArchiveDescriptor *)descriptor error:(NSError **)error
{
   MTLBinaryArchiveDescriptor *d = [[descriptor copy] autorelease];
   if (d.url)
      BG_LOG("binary archive %s: opened empty (precompiled Apple GPU code)\n", d.url.path.UTF8String);
   d.url = nil;
   return [super newBinaryArchiveWithDescriptor:d error:error];
}

- (id)newDefaultLibrary
{
   return [self newDefaultLibraryWithBundle:[NSBundle mainBundle] error:NULL];
}

- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options
{
   (void)options;
   return [[BGMTLBuffer alloc] initWithBGDevice:self length:length];
}

- (id)newBufferWithBytesNoCopy:(void *)p length:(NSUInteger)length options:(NSUInteger)options deallocator:(void (^)(void *, NSUInteger))dealloc
{
   (void)options;
   return [[BGMTLBuffer alloc] initWithBGDevice:self bytesNoCopy:p length:length deallocator:dealloc];
}

- (id)newBufferWithBytes:(const void *)bytes length:(NSUInteger)length options:(NSUInteger)options
{
   BGMTLBuffer *b = [self newBufferWithLength:length options:options];
   if (b && bytes && [b contents]) memcpy([b contents], bytes, length);
   return b;
}

- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)d
{
   return [[BGMTLTexture alloc] initWithBGDevice:self descriptor:d];
}
- (id)newHeapWithDescriptor:(MTLHeapDescriptor *)d
{
   if (!d || access("/private/var/tmp/bgmtl.noheap", F_OK) == 0) return nil;
   BGMTLHeap *h = [[BGMTLHeap alloc] init];
   h->_owner = self;
   h->_size = [d size];
   h->_storage = [d storageMode];
   h->_cache = [d cpuCacheMode];
   h->_hazard = [d hazardTrackingMode];
   h->_type = [d type];
   BG_TRACE("heap %p: %lu bytes, type %lu\n", h, (unsigned long)h->_size, (unsigned long)h->_type);
   return h;
}

// A texture over the caller's memory (QuartzCore's CPU-drawn images): a texture of our own, refreshed from those bytes at each use
// (rows of length / height bytes), and the deallocator called when it is freed.
- (id)newTextureWithBytesNoCopy:(void *)p length:(NSUInteger)length descriptor:(MTLTextureDescriptor *)d deallocator:(void (^)(void *, NSUInteger))dealloc
{
   BGMTLTexture *t = [[BGMTLTexture alloc] initWithBGDevice:self descriptor:d];
   if (!t) return nil;
   t->_ncBytes = p;
   t->_ncLength = length;
   t->_ncDealloc = dealloc ? Block_copy(dealloc) : NULL;
   return t;
}

- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)d iosurface:(void *)surface plane:(NSUInteger)plane
{
   return [[BGMTLTexture alloc] initWithBGDevice:self descriptor:d iosurface:surface plane:plane];
}

- (id)newDepthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)d
{
   return [[BGMTLDepthStencil alloc] initWithBGDevice:self descriptor:d];
}

- (id)newSamplerStateWithDescriptor:(MTLSamplerDescriptor *)d
{
   return [[BGMTLSampler alloc] initWithBGDevice:self descriptor:d];
}

- (id)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor error:(NSError **)error
{
   static SEL bitcode;
   if (!bitcode) bitcode = sel_getUid("bitcodeData");
   if (!_dev) return [self bgFail:error with:[NSString stringWithFormat:@"no GPU: %s", _why]];
   id vertex = [descriptor vertexFunction], fragment = [descriptor fragmentFunction];
   BG_TRACE("render pipeline: vertex %s fragment %s\n", vertex ? [[vertex name] UTF8String] : "-", fragment ? [[fragment name] UTF8String] : "-");
   if (bg_trace()) {
      for (unsigned i = 0; i < 8; i++) {
         MTLRenderPipelineColorAttachmentDescriptor *ca = descriptor.colorAttachments[i];
         if (ca.pixelFormat)
            BG_LOG("  pipe color%u fmt %lu blend %d src %lu/%lu dst %lu/%lu mask %lu\n", i, (unsigned long)ca.pixelFormat, ca.blendingEnabled,
                   (unsigned long)ca.sourceRGBBlendFactor, (unsigned long)ca.sourceAlphaBlendFactor, (unsigned long)ca.destinationRGBBlendFactor,
                   (unsigned long)ca.destinationAlphaBlendFactor, (unsigned long)ca.writeMask);
      }
      BG_LOG("  pipe depth fmt %lu stencil fmt %lu\n", (unsigned long)descriptor.depthAttachmentPixelFormat, (unsigned long)descriptor.stencilAttachmentPixelFormat);
   }
   // no fragment function: a depth/stencil-only pipeline (RenderBox's path fills write stencil this way)
   if (!vertex) return [self bgFail:error with:@"a render pipeline needs a vertex function"];
   NSData *vair = ((NSData * (*)(id, SEL))objc_msgSend)(vertex, bitcode);
   NSData *fair = fragment ? ((NSData * (*)(id, SEL))objc_msgSend)(fragment, bitcode) : nil;
   if ((fragment && !fair.length) || !vair.length) return [self bgFail:error with:@"a function has no bitcode"];
   if (bg_trace() && getenv("BGMTL_DUMP")) {   // the exact modules this pipeline is built from, for air2nir on the Mac
      mkdir(getenv("BGMTL_DUMP"), 0755);
      NSString *d = [NSString stringWithUTF8String:getenv("BGMTL_DUMP")];
      [vair writeToFile:[d stringByAppendingFormat:@"/%@.v.bc", [vertex name]] atomically:NO];
      if (fair) [fair writeToFile:[d stringByAppendingFormat:@"/%@.f.bc", [fragment name]] atomically:NO];
   }
   struct chl_render_pipeline_desc pd;
   memset(&pd, 0, sizeof(pd));
   pd.sample_count = (unsigned)[descriptor sampleCount];
   if ([descriptor respondsToSelector:@selector(isAlphaToCoverageEnabled)]) pd.alpha_to_coverage = [descriptor isAlphaToCoverageEnabled];
   if ([descriptor respondsToSelector:@selector(inputPrimitiveTopology)]) pd.topology_class = (unsigned)[descriptor inputPrimitiveTopology];
   // Chlorophyll: the functions each stage links in (MTLLinkedFunctions: iOS 26's RenderBox supplies its effect functions this way
   // and the stage only declares them), as their AIR; the translator resolves the stage's declared calls against them
   struct chl_linked_function vlk[16], flk[16];
   unsigned nvl = 0, nfl = 0;
   for (int st = 0; st < 2; st++) {
      SEL which = st ? sel_getUid("fragmentLinkedFunctions") : sel_getUid("vertexLinkedFunctions");
      id lf = [descriptor respondsToSelector:which] ? ((id (*)(id, SEL))objc_msgSend)(descriptor, which) : nil;
      for (int kind = 0; lf && kind < 2; kind++) {
         SEL list = kind ? sel_getUid("privateFunctions") : sel_getUid("functions");
         NSArray *fns = [lf respondsToSelector:list] ? ((id (*)(id, SEL))objc_msgSend)(lf, list) : nil;
         for (id fn in fns) {
            NSData *b = [fn respondsToSelector:bitcode] ? ((NSData * (*)(id, SEL))objc_msgSend)(fn, bitcode) : nil;
            unsigned *n = st ? &nfl : &nvl;
            if (!b.length || *n >= 16) continue;
            (st ? flk : vlk)[(*n)++] = (struct chl_linked_function){ b.bytes, b.length, [[fn name] UTF8String] };
            BG_TRACE("  linked %s function %s: %lu bytes of AIR\n", st ? "fragment" : "vertex", [[fn name] UTF8String],
                     (unsigned long)b.length);
         }
      }
   }
   pd.vertex = (struct chl_function){ vair.bytes, vair.length, [[vertex name] UTF8String], NULL, 0, nvl ? vlk : NULL, nvl };
   if (fair) pd.fragment = (struct chl_function){ fair.bytes, fair.length, [[fragment name] UTF8String], NULL, 0, nfl ? flk : NULL, nfl };
   for (unsigned i = 0; i < 8; i++) {
      MTLRenderPipelineColorAttachmentDescriptor *a = [[descriptor colorAttachments] objectAtIndexedSubscript:i];
      if (!a) continue;
      pd.color_format[i] = (unsigned)[a pixelFormat];
      pd.blend[i] = (struct chl_blend){ [a isBlendingEnabled], (unsigned)[a rgbBlendOperation], (unsigned)[a alphaBlendOperation],
                                        (unsigned)[a sourceRGBBlendFactor], (unsigned)[a destinationRGBBlendFactor],
                                        (unsigned)[a sourceAlphaBlendFactor], (unsigned)[a destinationAlphaBlendFactor], (unsigned)[a writeMask] };
   }
   pd.depth_format = (unsigned)[descriptor depthAttachmentPixelFormat];
   pd.stencil_format = (unsigned)[descriptor stencilAttachmentPixelFormat];
   MTLVertexDescriptor *vd = [descriptor vertexDescriptor];
   if (vd) {
      pd.has_vertex_descriptor = true;
      for (unsigned i = 0; i < CHL_MAX_ATTRIBUTES; i++) {
         MTLVertexAttributeDescriptor *a = [[vd attributes] objectAtIndexedSubscript:i];
         if (!a || ![a format]) continue;
         pd.attribute[i] = (struct chl_vertex_attribute){ (unsigned)[a format], (unsigned)[a offset], (unsigned)[a bufferIndex] };
         BG_TRACE("  attribute %u: format %lu offset %lu buffer %lu\n", i, (unsigned long)[a format], (unsigned long)[a offset],
                  (unsigned long)[a bufferIndex]);
      }
      for (unsigned k = 0; k < CHL_MAX_BUFFERS; k++) {
         MTLVertexBufferLayoutDescriptor *l = [[vd layouts] objectAtIndexedSubscript:k];
         if (!l) continue;
         pd.layout[k] = (struct chl_vertex_layout){ (unsigned)[l stride], (unsigned)[l stepFunction], (unsigned)[l stepRate] };
      }
   }
   if (fragment && bg_dump_listed([fragment name])) {
      struct chl_linked_function all[32];
      memcpy(all, vlk, nvl * sizeof(vlk[0]));
      memcpy(all + nvl, flk, nfl * sizeof(flk[0]));
      bg_refuse_dump(vair, [vertex name], fair, [fragment name], all, nvl, nfl);
   }
   if (fragment && bg_refused([fragment name])) {
      struct chl_linked_function all[32];
      memcpy(all, vlk, nvl * sizeof(vlk[0]));
      memcpy(all + nvl, flk, nfl * sizeof(flk[0]));
      bg_refuse_dump(vair, [vertex name], fair, [fragment name], all, nvl, nfl);
      char line[512], bwhy[256] = "";
      // /private/var/tmp/bgmtl.refuse.build: build it all the same (how long it takes, whether it compiles), never draw it
      if (access("/private/var/tmp/bgmtl.refuse.build", F_OK) == 0) {
         struct chl_pipeline_info binfo;
         uint64_t b0 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
         struct chl_render_pipeline *br = chl_render_pipeline_create(_dev, &pd, &binfo, bwhy, sizeof(bwhy));
         uint64_t b1 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
         if (br) chl_render_pipeline_release(br);
         snprintf(line, sizeof(line), "refused pipeline %s + %s built in %.1f ms: %s", [[vertex name] UTF8String],
                  [[fragment name] UTF8String], (b1 - b0) / 1e6, br ? "ok" : bwhy);
         bgmtl_note_line(line);
      }
      snprintf(line, sizeof(line), "render pipeline %s + %s refused by bgmtl.refuse (AIR in /private/var/tmp/bgmtl-refused)",
               [[vertex name] UTF8String], [[fragment name] UTF8String]);
      bgmtl_note_line(line);
      return [self bgFail:error with:@"refused by /private/var/tmp/bgmtl.refuse"];
   }
   char why[512] = { 0 };
   struct chl_pipeline_info info;
   uint64_t pt0 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
   struct chl_render_pipeline *r = chl_render_pipeline_create(_dev, &pd, &info, why, sizeof(why));
   uint64_t pt1 = clock_gettime_nsec_np(CLOCK_MONOTONIC);
   if (bg_perf_on()) {
      static uint64_t n, buildns;
      n++; buildns += pt1 - pt0;
      bg_perf_slow("pipeline build", fragment ? [[fragment name] UTF8String] : [[vertex name] UTF8String], pt1 - pt0);
      if (n % 16 == 0) {
         char line[160];
         snprintf(line, sizeof(line), "pipelines: %llu built, %.0f ms building", (unsigned long long)n, buildns / 1e6);
         bgmtl_note_line(line);
      }
   }
   if (!r) {
      bg_fail_dump(why, vair, [vertex name], fair, fragment ? [fragment name] : nil, "f");
      for (unsigned i = 0; i < nvl + nfl && access("/private/var/tmp/bgmtl.faildump", F_OK) == 0; i++) {
         const struct chl_linked_function *l = i < nvl ? &vlk[i] : &flk[i - nvl];
         [[NSData dataWithBytesNoCopy:(void *)l->bitcode length:l->size freeWhenDone:NO]
            writeToFile:[NSString stringWithFormat:@"/private/var/tmp/bgmtl-fail/%d-%s.%s.bc", getpid(), l->name ? l->name : "linked",
                                                  i < nvl ? "lv" : "lf"] atomically:NO];
      }
      return [self bgFail:error with:[NSString stringWithUTF8String:why]];
   }
   BGMTLRenderPipeline *p = [[BGMTLRenderPipeline alloc] initWithDevice:self pipelineStateDescriptor:descriptor];
   if (!p) { chl_render_pipeline_release(r); return [self bgFail:error with:@"_MTLRenderPipelineState refused to initialise"]; }
   p->_owner = self; p->_p = r; p->_info = info;
   snprintf(p->_names, sizeof(p->_names), "%s+%s", [[vertex name] UTF8String], fragment ? [[fragment name] UTF8String] : "-");
   BG_LOG("render pipeline %s: vertex buffers %#x, fragment buffers %#x textures %#x samplers %#x (%.1f ms)\n", p->_names,
          info.buffers_used[0], info.buffers_used[1], info.textures_used[1], info.samplers_used[1], (pt1 - pt0) / 1e6);
   return p;
}

// -- compute pipelines: both of _MTLDevice's entry points go through its (nil) -compiler, so they are ours --
- (id)bgPipelineForFunction:(id)function descriptor:(id)descriptor error:(NSError **)error
{
   static SEL bitcode;
   if (!bitcode) bitcode = sel_getUid("bitcodeData");
   if (!function) return [self bgFail:error with:@"no compute function"];
   if (!_dev) return [self bgFail:error with:[NSString stringWithFormat:@"no GPU: %s", _why]];
   if (![function respondsToSelector:bitcode]) return [self bgFail:error with:@"function has no bitcodeData selector"];
   NSData *air = ((NSData * (*)(id, SEL))objc_msgSend)(function, bitcode);
   if (!air.length) return [self bgFail:error with:@"function has no bitcode"];
   char why[512] = { 0 };
   struct chl_pipeline_info info;
   struct chl_function fn = { air.bytes, air.length, [[function name] UTF8String], NULL, 0, NULL, 0 };
   struct chl_compute_pipeline *c = chl_compute_pipeline_create(_dev, &fn, &info, why, sizeof(why));
   if (!c) {
      bg_fail_dump(why, nil, nil, air, [function name], "c");
      return [self bgFail:error with:[NSString stringWithUTF8String:why]];
   }
   BGMTLComputePipeline *p = [[BGMTLComputePipeline alloc] initWithDevice:self pipelineStateDescriptor:descriptor];
   if (!p) { chl_compute_pipeline_release(c); return [self bgFail:error with:@"_MTLComputePipelineState refused to initialise"]; }
   p->_owner = self;
   p->_p = c;
   p->_info = info;
   BG_LOG("pipeline %s: buffers %#x textures %#x samplers %#x\n", [[function name] UTF8String], info.buffers_used[0], info.textures_used[0],
          info.samplers_used[0]);
   return p;
}

- (id)newComputePipelineStateWithDescriptor:(id)descriptor options:(NSUInteger)options reflection:(id *)reflection error:(NSError **)error
{
   (void)options;
   if (reflection) *reflection = nil;
   return [self bgPipelineForFunction:[descriptor computeFunction] descriptor:descriptor error:error];
}

- (id)newComputePipelineStateWithDescriptor:(id)descriptor error:(NSError **)error
{
   return [self bgPipelineForFunction:[descriptor computeFunction] descriptor:descriptor error:error];
}

- (id)newComputePipelineStateWithFunction:(id)function error:(NSError **)error
{
   MTLComputePipelineDescriptor *d = [[MTLComputePipelineDescriptor alloc] init];
   [d setComputeFunction:function];
   id s = [self bgPipelineForFunction:function descriptor:d error:error];
   [d release];
   return s;
}

// The other ways to ask for a pipeline: with reflection, and asynchronously. They reached _MTLDevice's own implementations,
// which hand the work to MTLCompilerService for an Apple GPU and come back with nothing usable here (iOS 26's RenderBox, which
// draws the icons in iconservicesagent: MTLCompilerService at 100 % for minutes, then "precondition failure" and an abort).
// Built by the methods above; the reflection object is not made (nil). An asynchronous request is built on a queue of its own
// and its handler called from there, as Metal does: calling it on the caller's thread deadlocked backboardd's first frame
// (CoreAnimation asks while holding a lock its handler takes).
// the first use of each of these creation paths, to the injection's log (which one a client takes)
static void
bg_note_path(const char *what)
{
   static NSMutableSet *seen;
   static os_unfair_lock lk = OS_UNFAIR_LOCK_INIT;
   os_unfair_lock_lock(&lk);
   if (!seen) seen = [[NSMutableSet alloc] init];
   BOOL fresh = ![seen containsObject:@(what)];
   if (fresh) [seen addObject:@(what)];
   os_unfair_lock_unlock(&lk);
   if (fresh) {
      char line[160];
      snprintf(line, sizeof(line), "pipeline path: %s", what);
      bgmtl_note_line(line);
   }
}

static dispatch_queue_t
bg_pipeline_queue(void)
{
   static dispatch_queue_t q;
   static dispatch_once_t once;
   dispatch_once(&once, ^{ q = dispatch_queue_create("chlorophyll.pipelines", DISPATCH_QUEUE_SERIAL); });
   return q;
}

- (id)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor options:(NSUInteger)options
                                reflection:(id *)reflection error:(NSError **)error
{
   BG_TRACE("newRenderPipelineState...options:%lu reflection:%s\n", (unsigned long)options, reflection ? "asked" : "-");
   // A caller that wants the reflection (backboardd's CoreAnimation) keeps _MTLDevice's own path, which it has always taken
   // and which builds the pipeline through the method above; answering it nil hung backboardd's first frame for good.
   // Without it (RenderBox in iconservicesagent) the pipeline is ours alone.
   if (reflection) {
      bg_note_path("newRenderPipelineState...options reflection (asked: _MTLDevice's)");
      return ((id (*)(struct objc_super *, SEL, id, NSUInteger, id *, NSError **))objc_msgSendSuper)(
         &(struct objc_super){ self, [_MTLDevice class] }, _cmd, descriptor, options, reflection, error);
   }
   bg_note_path("newRenderPipelineState...options reflection (none)");
   return [self newRenderPipelineStateWithDescriptor:descriptor error:error];
}
- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor completionHandler:(void (^)(id, NSError *))handler
{
   bg_note_path("newRenderPipelineState...completionHandler");
   BG_TRACE("newRenderPipelineState...completionHandler\n");
   MTLRenderPipelineDescriptor *d = [descriptor copy];
   void (^h)(id, NSError *) = [handler copy];
   [self retain];
   dispatch_async(bg_pipeline_queue(), ^{
      NSError *err = nil;
      id p = [self newRenderPipelineStateWithDescriptor:d error:&err];
      if (h) h(p, err);
      [p release]; [d release]; [h release]; [self release];
   });
}
- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor options:(NSUInteger)options
                           completionHandler:(void (^)(id, id, NSError *))handler
{
   bg_note_path("newRenderPipelineState...options completionHandler");
   BG_TRACE("newRenderPipelineState...options:%lu completionHandler\n", (unsigned long)options);
   MTLRenderPipelineDescriptor *d = [descriptor copy];
   void (^h)(id, id, NSError *) = [handler copy];
   [self retain];
   dispatch_async(bg_pipeline_queue(), ^{
      NSError *err = nil;
      id p = [self newRenderPipelineStateWithDescriptor:d error:&err];
      if (h) h(p, nil, err);
      [p release]; [d release]; [h release]; [self release];
   });
}
- (void)newComputePipelineStateWithFunction:(id)function completionHandler:(void (^)(id, NSError *))handler
{
   bg_note_path("newComputePipelineState...completionHandler");
   BG_TRACE("newComputePipelineState...completionHandler\n");
   id fn = [function retain];
   void (^h)(id, NSError *) = [handler copy];
   [self retain];
   dispatch_async(bg_pipeline_queue(), ^{
      NSError *err = nil;
      id p = [self newComputePipelineStateWithFunction:fn error:&err];
      if (h) h(p, err);
      [p release]; [fn release]; [h release]; [self release];
   });
}
- (void)newComputePipelineStateWithFunction:(id)function options:(NSUInteger)options completionHandler:(void (^)(id, id, NSError *))handler
{
   (void)options;
   bg_note_path("newComputePipelineState...options completionHandler");
   BG_TRACE("newComputePipelineState...options completionHandler\n");
   id fn = [function retain];
   void (^h)(id, id, NSError *) = [handler copy];
   [self retain];
   dispatch_async(bg_pipeline_queue(), ^{
      NSError *err = nil;
      id p = [self newComputePipelineStateWithFunction:fn error:&err];
      if (h) h(p, nil, err);
      [p release]; [fn release]; [h release]; [self release];
   });
}
- (id)newComputePipelineStateWithFunction:(id)function options:(NSUInteger)options reflection:(id *)reflection error:(NSError **)error
{
   (void)options;
   if (reflection) *reflection = nil;
   return [self newComputePipelineStateWithFunction:function error:error];
}
- (void)newComputePipelineStateWithDescriptor:(id)descriptor options:(NSUInteger)options completionHandler:(void (^)(id, id, NSError *))handler
{
   (void)options;
   bg_note_path("newComputePipelineState(descriptor)...completionHandler");
   BG_TRACE("newComputePipelineState(descriptor)...completionHandler\n");
   id d = [descriptor copy];
   void (^h)(id, id, NSError *) = [handler copy];
   [self retain];
   dispatch_async(bg_pipeline_queue(), ^{
      NSError *err = nil;
      id p = [self newComputePipelineStateWithDescriptor:d error:&err];
      if (h) h(p, nil, err);
      [p release]; [d release]; [h release]; [self release];
   });
}

@end

// ---- buffer ----------------------------------------------------------------------------------------------------------
@implementation BGMTLHeap
- (void)dealloc { [_label release]; [super dealloc]; }
- (id)device { return _owner; }
- (NSString *)label { return _label; }
- (void)setLabel:(NSString *)l { [_label release]; _label = [l copy]; }
- (NSUInteger)storageMode { return _storage; }
- (NSUInteger)cpuCacheMode { return _cache; }
- (NSUInteger)hazardTrackingMode { return _hazard; }
- (NSUInteger)resourceOptions { return (_storage << MTLResourceStorageModeShift) | (_cache << MTLResourceCPUCacheModeShift) | (_hazard << MTLResourceHazardTrackingModeShift); }
- (NSUInteger)type { return _type; }
- (NSUInteger)size { return _size; }
- (NSUInteger)usedSize { return 0; }
- (NSUInteger)currentAllocatedSize { return _size; }
- (NSUInteger)maxAvailableSizeWithAlignment:(NSUInteger)alignment { (void)alignment; return _size; }
- (NSUInteger)setPurgeableState:(NSUInteger)state { bg_count_purge(state); return MTLPurgeableStateNonVolatile; }
- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options
{ __atomic_add_fetch(&g_bg_heap_allocs, 1, __ATOMIC_RELAXED); return [_owner newBufferWithLength:length options:options]; }
- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options offset:(NSUInteger)offset
{ (void)offset; __atomic_add_fetch(&g_bg_heap_allocs, 1, __ATOMIC_RELAXED); return [_owner newBufferWithLength:length options:options]; }
- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)d
{ __atomic_add_fetch(&g_bg_heap_allocs, 1, __ATOMIC_RELAXED); return [_owner newTextureWithDescriptor:d]; }
- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)d offset:(NSUInteger)offset
{ (void)offset; __atomic_add_fetch(&g_bg_heap_allocs, 1, __ATOMIC_RELAXED); return [_owner newTextureWithDescriptor:d]; }
@end

@implementation BGMTLBuffer
+ (void)load { ensure_protocol(self, "MTLBuffer"); }

- (instancetype)initWithBGDevice:(BGMTLDevice *)device length:(NSUInteger)length
{
   self = [super init];
   if (!self) return nil;
   char why[256] = { 0 };
   if (![device chl] || !length) { BG_LOG("buffer of %lu bytes: no GPU device\n", (unsigned long)length); [self release]; return nil; }
   _buf = chl_buffer_create([device chl], length, why, sizeof(why));
   if (!_buf) { BG_LOG("chl_buffer_create(%lu): %s\n", (unsigned long)length, why); [self release]; return nil; }
   _owner = device;
   _length = length;
   _contents = chl_buffer_contents(_buf);
   __atomic_add_fetch(&g_bg_buf_n, 1, __ATOMIC_RELAXED);
   __atomic_add_fetch(&g_bg_buf_bytes, (long)length, __ATOMIC_RELAXED);
   return self;
}
// newBufferWithBytesNoCopy: contents is the caller's memory, as Metal promises; the GPU reads its own copy, refreshed from the
// caller's bytes when a command buffer that binds it is committed. A GPU write into such a buffer does not reach the caller's
// memory; CoreAnimation's are vertex and glyph data it writes on the CPU.
- (instancetype)initWithBGDevice:(BGMTLDevice *)device bytesNoCopy:(void *)p length:(NSUInteger)length deallocator:(void (^)(void *, NSUInteger))dealloc
{
   self = [self initWithBGDevice:device length:length];
   if (!self) return nil;
   _mapped = _contents;
   _contents = p;
   _deallocator = dealloc ? Block_copy(dealloc) : NULL;
   memcpy(_mapped, p, length);
   return self;
}
- (void)dealloc
{
   if (_buf) {
      chl_buffer_release(_buf);
      __atomic_sub_fetch(&g_bg_buf_n, 1, __ATOMIC_RELAXED);
      __atomic_sub_fetch(&g_bg_buf_bytes, (long)_length, __ATOMIC_RELAXED);
   }
   if (_deallocator) { _deallocator(_contents, _length); Block_release(_deallocator); }
   [super dealloc];
}
- (struct chl_buffer *)chlBuffer { return _buf; }
- (BOOL)bgIsNoCopy { return _mapped != NULL; }
- (void)bgRefresh { if (_mapped && _contents) memcpy(_mapped, _contents, _length); }
- (void *)contents { return _contents; }
- (NSUInteger)length { return _length; }
- (id)device { return _owner; }
- (NSUInteger)allocatedSize { return _length; }
- (uint64_t)gpuAddress { return chl_buffer_gpu_address(_buf); }
- (void)didModifyRange:(NSRange)range { (void)range; }
- (NSUInteger)storageMode { return 0; }
- (NSUInteger)cpuCacheMode { return 1; }
- (NSUInteger)resourceOptions { return 0; }
- (NSUInteger)hazardTrackingMode { return 0; }
- (NSUInteger)setPurgeableState:(NSUInteger)state { bg_count_purge(state); return 2; }
- (id)heap { return nil; }
- (NSUInteger)heapOffset { return 0; }
- (BOOL)isAliasable { return NO; }
- (void)makeAliasable { __atomic_add_fetch(&g_bg_aliasable, 1, __ATOMIC_RELAXED); }
- (BOOL)doesAliasResource:(id)r { (void)r; return NO; }
- (BOOL)isComplete { return YES; }
- (void)waitUntilComplete {}
- (BOOL)isPurgeable { return NO; }
- (NSUInteger)protectionOptions { return 0; }
- (id)newLinearTextureWithDescriptor:(MTLTextureDescriptor *)d offset:(NSUInteger)offset bytesPerRow:(NSUInteger)bpr bytesPerImage:(NSUInteger)bpi
{
   (void)bpi;
   BGMTLTexture *t = [[BGMTLTexture alloc] initWithBGDevice:_owner descriptor:d];
   if (!t) return nil;
   t->_linSrc = [self retain];
   t->_linOff = offset;
   t->_linBpr = bpr;
   return t;
}
// MTLBuffer's own texture over its memory (iOS 26's IconRendering: "Could not make MTLTexture referencing MTLBuffer" without it)
- (id)newTextureWithDescriptor:(MTLTextureDescriptor *)d offset:(NSUInteger)offset bytesPerRow:(NSUInteger)bpr
{
   return [self newLinearTextureWithDescriptor:d offset:offset bytesPerRow:bpr bytesPerImage:bpr * [d height]];
}
@end

// ---- pipeline --------------------------------------------------------------------------------------------------------
@implementation BGMTLComputePipeline
+ (void)load { ensure_protocol(self, "MTLComputePipelineState"); }
- (void)dealloc { chl_compute_pipeline_release(_p); [super dealloc]; }
- (id)device { return _owner; }
- (NSUInteger)maxTotalThreadsPerThreadgroup { return _info.max_threads_per_group; }
- (NSUInteger)threadExecutionWidth { return _info.thread_execution_width; }
- (NSUInteger)staticThreadgroupMemoryLength { return _info.static_threadgroup_bytes; }
@end

// ---- command queue / buffer / encoders -------------------------------------------------------------------------------
@implementation BGMTLCommandQueue
- (instancetype)initWithBGDevice:(BGMTLDevice *)device descriptor:(id)descriptor
{
   self = [super initWithDevice:device descriptor:descriptor];
   if (!self) return nil;
   _owner = device;
   return self;
}
- (id)device { return _owner; }
- (id)commandBuffer { return [[[BGMTLCommandBuffer alloc] initWithBGQueue:self] autorelease]; }
- (id)commandBufferWithUnretainedReferences { return [self commandBuffer]; }
// CoreAnimation asks with a descriptor; _MTLCommandQueue's default would hand back a generic command buffer that never reaches the GPU
- (id)commandBufferWithDescriptor:(id)d { (void)d; return [self commandBuffer]; }
- (void)insertDebugCaptureBoundary {}
@end

@implementation BGMTLCommandBuffer
/* iOS 26: _MTLCommandBuffer's initializer waits on its queue's _commandBufferSemaphore (maxCommandBufferCount, 64 by default),
 * which Apple signals from its own completion path (commandBufferDidComplete:...). Ours complete on their own, so the queue
 * never got its slots back and command buffer 65 blocked for ever (backboardd froze after a few frames; the CA probe stopped
 * at exactly 64 pipelines). Each buffer returns its slot once: at completion, or at dealloc if it never completed. Looked up
 * by name, so a Metal without the ivar is left alone. */
static void
bg_return_slot(BGMTLCommandBuffer *cb, BGMTLCommandQueue *queue)
{
   if (!queue || __atomic_exchange_n(&cb->_slotReturned, 1, __ATOMIC_ACQ_REL)) return;
   static Ivar sem_ivar;
   static dispatch_once_t once;
   dispatch_once(&once, ^{ sem_ivar = class_getInstanceVariable(objc_getClass("_MTLCommandQueue"), "_commandBufferSemaphore"); });
   if (!sem_ivar) return;
   dispatch_semaphore_t sem = (dispatch_semaphore_t)object_getIvar(queue, sem_ivar);
   if (sem) dispatch_semaphore_signal(sem);
}
- (instancetype)initWithBGQueue:(BGMTLCommandQueue *)queue
{
   self = [super initWithQueue:queue retainedReferences:YES synchronousDebugMode:NO];
   if (!self) return nil;
   _queue = [queue retain];
   _completed = [[NSMutableArray alloc] init];
   _scheduled = [[NSMutableArray alloc] init];
   _presents = [[NSMutableArray alloc] init];
   _done = dispatch_group_create();
   _status = 1;   // MTLCommandBufferStatusEnqueued
   _seq = __atomic_add_fetch(&g_bg_cb_seq, 1, __ATOMIC_RELAXED);
   char why[256] = { 0 };
   struct chl_device *dev = [(BGMTLDevice *)[queue device] chl];
   _ccb = dev ? chl_cmdbuf_create(dev, why, sizeof(why)) : NULL;
   if (!_ccb) BG_LOG("command buffer: %s\n", dev ? why : "no GPU device");
   return self;
}
- (void)dealloc
{
   bg_return_slot(self, _queue);
   if (_ccb) chl_cmdbuf_release(_ccb);   // never committed: discarded; committed: complete by now (the completion holds a reference)
   [_queue release]; [_error release]; [_completed release]; [_scheduled release]; [_presents release];
   [_linWritten release]; [_written release]; [_noCopy release];
   if (_done) dispatch_release(_done);
   [super dealloc];
}
- (id)device { return [_queue device]; }
- (struct chl_cmdbuf *)bgChl { if (!_t_first) _t_first = clock_gettime_nsec_np(CLOCK_MONOTONIC); return _ccb; }
- (NSInteger)status { return _status; }
- (NSError *)error { return _error; }
- (void)enqueue {}
- (void)addScheduledHandler:(void (^)(id))block
{
   if (!block) return;
   if (_status >= 3) block(self); else [_scheduled addObject:[[block copy] autorelease]];
}
- (void)presentDrawable:(id)drawable { if (drawable) [_presents addObject:@[ drawable, @0.0, @0.0 ]]; }
- (void)presentDrawable:(id)drawable atTime:(CFTimeInterval)t { if (drawable) [_presents addObject:@[ drawable, @(t), @0.0 ]]; }
- (void)presentDrawable:(id)drawable afterMinimumDuration:(CFTimeInterval)d { if (drawable) [_presents addObject:@[ drawable, @0.0, @(d) ]]; }
- (void)addCompletedHandler:(void (^)(id))block
{
   if (!block) return;
   if (_status >= 4) block(self); else [_completed addObject:[[block copy] autorelease]];
}
- (void)waitUntilScheduled { if (_status >= 2 && getenv("BGMTL_BB")) dispatch_group_wait(_done, DISPATCH_TIME_FOREVER); }

- (void)bgFailWith:(NSString *)why code:(NSInteger)code
{
   BG_LOG("command buffer failed: %s\n", [why UTF8String]);
   bgmtl_note_failure([why UTF8String]);   // the injection dylib counts these (its dead-man)
   if (!_error) _error = [[NSError errorWithDomain:@"BGMTL" code:code userInfo:@{ NSLocalizedDescriptionKey : why }] retain];
   _status = 5;   // MTLCommandBufferStatusError
}
// what an encoder could not record: logged and skipped, the rest of the command buffer still runs (failing the whole command
// buffer left the frame's later passes undrawn and counted towards the watchdog that switches Metal off)
- (void)bgSkip:(const char *)what why:(const char *)why { BG_LOG("%s skipped: %s\n", what, why); }
- (void)bgWrote:(BGMTLTexture *)t
{
   if (!t) return;
   if (!_written) _written = [[NSMutableSet alloc] init];
   [_written addObject:t];
   BGMTLTexture *root = [t bgRoot];
   if (root->_linSrc) {
      if (!_linWritten) _linWritten = [[NSMutableSet alloc] init];
      [_linWritten addObject:root];
   }
}
- (void)bgUses:(BGMTLBuffer *)b
{
   if (![b bgIsNoCopy]) return;
   if (!_noCopy) _noCopy = [[NSMutableSet alloc] init];
   [_noCopy addObject:b];
}

// Per-second timing of commits (the compositor): how a frame's time splits into CoreAnimation's own work, encoding (from the
// first encoder to the commit), submission and the wait for the GPU. One line a second in bgmtl.log; off unless
// /private/var/tmp/bgmtl.perf exists when the process starts.
static void
bg_perf_note(uint64_t t_first, uint64_t t_begin, uint64_t t_flushed, uint64_t t_done, unsigned ndraw)
{
   static int on = -1;
   static uint64_t win_start, last_end, n, draws, enc, sub, wait, gap, enc_max, wait_max;
   static uint64_t last_begin, burst_ns, burst_frames, bin[4];
   if (on < 0) on = access("/private/var/tmp/bgmtl.perf", F_OK) == 0;
   if (!on) return;
   if (last_begin && t_begin > last_begin && t_begin - last_begin < 100000000ull) {
      uint64_t d = t_begin - last_begin;
      burst_ns += d; burst_frames++;
      bin[d <= 17000000ull ? 0 : d <= 25000000ull ? 1 : d <= 34000000ull ? 2 : 3]++;
   }
   last_begin = t_begin;
   if (win_start == 0) win_start = t_begin;
   uint64_t e = t_first && t_first < t_begin ? t_begin - t_first : 0, s = t_flushed - t_begin, w = t_done > t_flushed ? t_done - t_flushed : 0;
   if (last_end && t_first > last_end) gap += t_first - last_end;
   last_end = t_done;
   n++; draws += ndraw; enc += e; sub += s; wait += w;
   if (e > enc_max) enc_max = e;
   if (w > wait_max) wait_max = w;
   if (t_done - win_start >= 1000000000ull) {
      char line[300];
      double sec = (t_done - win_start) / 1e9;
      snprintf(line, sizeof(line), "perf: %.1f cbuf/s, %.1f draws/cbuf; per second: encode %.0f ms, submit %.0f ms, gpu wait %.0f ms, "
               "between %.0f ms; max encode %.1f ms, max wait %.1f ms", n / sec, n ? (double)draws / n : 0, enc / 1e6 / sec,
               sub / 1e6 / sec, wait / 1e6 / sec, gap / 1e6 / sec, enc_max / 1e6, wait_max / 1e6);
      bgmtl_note_line(line);
      if (burst_frames >= 3) {
         snprintf(line, sizeof(line), "frames: %.1f fps while animating (%llu intervals, %.0f ms); intervals <=17 ms %llu, <=25 %llu, <=34 %llu, "
                  "<=100 %llu", burst_frames * 1e9 / burst_ns, (unsigned long long)burst_frames, burst_ns / 1e6,
                  (unsigned long long)bin[0], (unsigned long long)bin[1], (unsigned long long)bin[2], (unsigned long long)bin[3]);
         bgmtl_note_line(line);
      }
      if (bg_texpool_cap()) {
         snprintf(line, sizeof(line), "texpool: %lu hits, %lu misses, %.1f MB held", g_texpool_hits, g_texpool_misses,
                  g_texpool_bytes / 1048576.0);
         bgmtl_note_line(line);
         g_texpool_hits = g_texpool_misses = 0;
      }
      burst_ns = burst_frames = bin[0] = bin[1] = bin[2] = bin[3] = 0;
      win_start = t_done; n = draws = enc = sub = wait = gap = enc_max = wait_max = 0;
   }
}

struct bg_done_ctx { BGMTLCommandBuffer *cb; uint64_t t_begin, t_flushed; };

static void
bg_cmdbuf_done(void *ctx, int status, const char *why)
{
   struct bg_done_ctx *d = ctx;
   BGMTLCommandBuffer *cb = d->cb;
   @autoreleasepool {
      if (status != 0) [cb bgFailWith:[NSString stringWithUTF8String:why ? why : "the GPU did not finish"] code:3];
      else cb->_status = 4;   // Completed
      bg_perf_note(cb->_t_first, d->t_begin, d->t_flushed, clock_gettime_nsec_np(CLOCK_MONOTONIC), cb->_draws);
      [cb bgCompleteFrom:d->t_begin];
      dispatch_group_leave(cb->_done);
   }
   free(d);
   [cb release];
}

- (void)commit
{
   __atomic_add_fetch(&g_bg_commit_seq, 1, __ATOMIC_RELAXED);
   bgmtl_note_committed();
   uint64_t t_begin = clock_gettime_nsec_np(CLOCK_MONOTONIC);
   _status = 2;   // Committed
   BG_TRACE("commit: %u draws\n", _draws);
   for (BGMTLBuffer *b in _noCopy) [b bgRefresh];
   for (BGMTLTexture *t in _linWritten) [t bgLinearPending:1];
   dispatch_group_enter(_done);
   // "scheduled" before the GPU has finished, unless BGMTL_DEFER_SCHEDULED=1 (the kernel holds a swap until the GPU work before
   // it completed, so the render server may hand the framebuffer on now and encode the next frame while the GPU draws this one)
   static int defer_scheduled = -1;
   if (defer_scheduled < 0) defer_scheduled = getenv("BGMTL_DEFER_SCHEDULED") && getenv("BGMTL_DEFER_SCHEDULED")[0] == '1';
   static int sync_commit = -1;     // BGMTL_SYNC=1 or /private/var/tmp/bgmtl.sync: wait inline
   if (sync_commit < 0)
      sync_commit = (getenv("BGMTL_SYNC") && getenv("BGMTL_SYNC")[0] == '1') || access("/private/var/tmp/bgmtl.sync", F_OK) == 0;
   char why[512] = { 0 };
   if (!_ccb) {
      [self bgFailWith:@"no GPU command buffer" code:2];
   } else if (bg_trace() || sync_commit) {
      if (chl_cmdbuf_commit(_ccb, NULL, NULL, why, sizeof(why)) != 0) [self bgFailWith:[NSString stringWithUTF8String:why] code:2];
      if (_status != 5 && !defer_scheduled) _status = 3;
      uint64_t t_flushed = clock_gettime_nsec_np(CLOCK_MONOTONIC);
      if (!defer_scheduled) { for (void (^h)(id) in _scheduled) h(self); [_scheduled removeAllObjects]; }
      if (_status != 5 && chl_cmdbuf_wait(_ccb, why, sizeof(why)) != 0) [self bgFailWith:[NSString stringWithUTF8String:why] code:3];
      else if (_status != 5) _status = 4;
      bg_perf_note(_t_first, t_begin, t_flushed, clock_gettime_nsec_np(CLOCK_MONOTONIC), _draws);
      [self bgCompleteFrom:t_begin];
      dispatch_group_leave(_done);
      return;
   } else {
      struct bg_done_ctx *ctx = malloc(sizeof(*ctx));
      ctx->cb = [self retain];
      ctx->t_begin = t_begin;
      ctx->t_flushed = 0;
      if (!defer_scheduled) _status = 3;
      if (chl_cmdbuf_commit(_ccb, bg_cmdbuf_done, ctx, why, sizeof(why)) != 0)
         return;   // bg_cmdbuf_done ran with the failure
      ctx->t_flushed = clock_gettime_nsec_np(CLOCK_MONOTONIC);
      if (!defer_scheduled) { for (void (^h)(id) in _scheduled) h(self); [_scheduled removeAllObjects]; }
      return;
   }
   [self bgCompleteFrom:t_begin];
   dispatch_group_leave(_done);
}

// Chlorophyll: give memory back once the process goes quiet (iOS kills an idle iconservicesagent above 17 MB). chl keeps no
// caches of its own beyond recycled command buffers and upload slabs, so only the texture pool is emptied here.
static unsigned long g_bg_trimmed_seq = (unsigned long)-1;
static void
bg_idle_trim_schedule(void)
{
   static int skip = -1;
   if (skip < 0) skip = !strcmp(getprogname(), "backboardd") || access("/private/var/tmp/bgmtl.notrim", F_OK) == 0;
   if (skip) return;
   static uint64_t delay_ms;
   if (!delay_ms) delay_ms = !strcmp(getprogname(), "SpringBoard") ? 30000 : 1500;
   unsigned long seq = __atomic_load_n(&g_bg_commit_seq, __ATOMIC_RELAXED);
   dispatch_after(dispatch_time(DISPATCH_TIME_NOW, delay_ms * NSEC_PER_MSEC), dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
      if (__atomic_load_n(&g_bg_commit_seq, __ATOMIC_RELAXED) != seq) return;
      if (__atomic_exchange_n(&g_bg_trimmed_seq, seq, __ATOMIC_RELAXED) == seq) return;
      os_unfair_lock_lock(&g_texpool_lock);
      for (unsigned k = 0; k < BG_TEXPOOL_SLOTS; k++) if (g_texpool[k].tex) bg_texpool_drop(k);
      os_unfair_lock_unlock(&g_texpool_lock);
      if (bg_perf_on()) {
         char line[300];
         snprintf(line, sizeof(line), "idle trim: Metal: %ld textures ~%.1f MB, %ld buffers %.1f MB, %lu heap allocations, %lu makeAliasable",
                  g_bg_tex_n, g_bg_tex_bytes / 1048576.0, g_bg_buf_n, g_bg_buf_bytes / 1048576.0, g_bg_heap_allocs, g_bg_aliasable);
         bgmtl_note_line(line);
      }
   });
}

// What follows a completed (or failed) command buffer; run on chl's completion thread, or inline when tracing.
- (void)bgCompleteFrom:(uint64_t)t_begin
{
   bg_return_slot(self, _queue);
   for (void (^h)(id) in _scheduled) h(self);    // left here by the compositor's deferral (see commit)
   [_scheduled removeAllObjects];
   // buffer-backed textures: the GPU's result goes back into their buffers before any completed handler reads them
   for (BGMTLTexture *t in _linWritten) {
      if (_status == 4) [t bgLinearWriteBack];
      [t bgLinearPending:-1];
   }
   [_linWritten removeAllObjects];
   // What the GPU wrote into an IOSurface becomes visible to the processor (and to a compositor reading it) before the drawable
   // is presented. In the compositor (BGMTL_BB) the surfaces drawn into are its own framebuffers, which the CPU never reads: no
   // cache clean there. A GPU write does not move an IOSurface's seed but consumers cache by it, so a write lock/unlock bumps it.
   static int in_compositor = -1;
   if (in_compositor < 0) in_compositor = getenv("BGMTL_BB") != NULL;
   if (_status == 4 && !in_compositor) {
      NSMutableSet *surfaces = nil, *xr = nil;
      for (BGMTLTexture *t in _written) {
         BGMTLTexture *root = [t bgRoot];
         if (root->_xr2p) { if (!xr) xr = [NSMutableSet set]; [xr addObject:root]; continue; }
         chl_texture_flush_surface([t chlTexture]);
         if ([t iosurface]) { if (!surfaces) surfaces = [NSMutableSet set]; [surfaces addObject:(id)[t iosurface]]; }
      }
      for (id surf in surfaces) {
         IOSurfaceLock((IOSurfaceRef)surf, 0, NULL);
         IOSurfaceUnlock((IOSurfaceRef)surf, 0, NULL);
      }
      for (BGMTLTexture *t in xr) [t bgXR10WriteBack];   // the write unlock moves these surfaces' seeds
   }
   if (_status == 4 && [_presents count]) {
      uint64_t t_end = clock_gettime_nsec_np(CLOCK_MONOTONIC);
      char line[200];
      snprintf(line, sizeof(line), "present: %u presents, %u draws, commit %.1f ms, at %.3f s", (unsigned)[_presents count], _draws,
               (t_end - t_begin) / 1e6, t_end / 1e9);
      bgmtl_note_present(line);
   }
   if (_status == 4) bgmtl_note_completed();
   if (_status == 4) {
      for (NSArray *p in _presents) {
         id dr = p[0];
         double at = [p[1] doubleValue], after = [p[2] doubleValue];
         if (at > 0 && [dr respondsToSelector:@selector(presentAtTime:)]) [dr presentAtTime:at];
         else if (after > 0 && [dr respondsToSelector:@selector(presentAfterMinimumDuration:)]) [dr presentAfterMinimumDuration:after];
         else [dr present];
      }
   }
   [_presents removeAllObjects];
   [_written removeAllObjects];
   for (void (^h)(id) in _completed) h(self);
   [_completed removeAllObjects];
   bg_idle_trim_schedule();
}

- (void)waitUntilCompleted { if (_status >= 2) dispatch_group_wait(_done, DISPATCH_TIME_FOREVER); }

- (id)computeCommandEncoder { return [[[BGMTLComputeEncoder alloc] initWithBGCommandBuffer:self] autorelease]; }
- (id)computeCommandEncoderWithDescriptor:(id)d { (void)d; return [self computeCommandEncoder]; }
- (id)computeCommandEncoderWithDispatchType:(NSUInteger)t { (void)t; return [self computeCommandEncoder]; }
// SPI hints CoreAnimation sets on every command buffer: which tasks the work is accounted to, and the protection class of the
// content. Neither changes what this GPU does.
- (void)setResponsibleTaskIDs:(const int *)ids count:(NSUInteger)count { (void)ids; (void)count; }
- (void)setProtectionOptions:(uint64_t)options { (void)options; }
- (uint64_t)protectionOptions { return 0; }

- (id)renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)d
{
   if (bg_trace()) {
      id<MTLTexture> t = d.colorAttachments[0].texture;
      unsigned surf0 = (t && [t respondsToSelector:@selector(iosurface)] && [(BGMTLTexture *)t iosurface])
                          ? (unsigned)IOSurfaceGetID((IOSurfaceRef)[(BGMTLTexture *)t iosurface]) : 0;
      BG_LOG("render pass: color0 %lux%lu fmt %lu level %lu load %lu store %lu surface %u\n", (unsigned long)t.width, (unsigned long)t.height,
             (unsigned long)t.pixelFormat, (unsigned long)d.colorAttachments[0].level, (unsigned long)d.colorAttachments[0].loadAction,
             (unsigned long)d.colorAttachments[0].storeAction, surf0);
   }
   return [[[BGMTLRenderEncoder alloc] initWithBGCommandBuffer:self descriptor:d] autorelease];
}
- (id)blitCommandEncoder { return [[[BGMTLBlitEncoder alloc] initWithBGCommandBuffer:self] autorelease]; }
- (id)blitCommandEncoderWithDescriptor:(id)d { (void)d; return [self blitCommandEncoder]; }
@end

// binding helpers shared by the encoders: a Metal buffer, texture or sampler onto chl's per-stage tables
static void
bg_bind_buffer(BGMTLCommandBuffer *cb, unsigned stage, NSUInteger index, id buffer, NSUInteger offset)
{
   if (index >= BG_MAX_BINDINGS || !cb->_ccb) return;
   if (buffer) [cb bgUses:(BGMTLBuffer *)buffer];
   chl_set_buffer(cb->_ccb, stage, (unsigned)index, buffer ? [(BGMTLBuffer *)buffer chlBuffer] : NULL, offset);
}
static void
bg_bind_texture(BGMTLCommandBuffer *cb, unsigned stage, NSUInteger index, id texture)
{
   if (index >= BG_MAX_TEX || !cb->_ccb) return;
   chl_set_texture(cb->_ccb, stage, (unsigned)index, texture ? [(BGMTLTexture *)texture chlTextureFor:cb] : NULL);
}
static void
bg_bind_sampler(BGMTLCommandBuffer *cb, unsigned stage, NSUInteger index, id sampler)
{
   if (index >= BG_MAX_TEX || !cb->_ccb) return;
   chl_set_sampler(cb->_ccb, stage, (unsigned)index, sampler ? [(BGMTLSampler *)sampler chlSampler] : NULL);
}

@implementation BGMTLComputeEncoder
- (instancetype)initWithBGCommandBuffer:(BGMTLCommandBuffer *)cb
{
   self = [super initWithCommandBuffer:cb];
   if (!self) return nil;
   _cb = cb;
   if ([cb bgChl]) chl_compute_begin(cb->_ccb);
   return self;
}
- (id)device { return [_cb device]; }
- (void)setComputePipelineState:(id)state
{
   _pipeline = state;
   if (state && _cb->_ccb) chl_set_compute_pipeline(_cb->_ccb, ((BGMTLComputePipeline *)state)->_p);
}
- (void)setBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index { bg_bind_buffer(_cb, CHL_STAGE_COMPUTE, index, buffer, offset); }
- (void)setBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
   // chl keeps the buffer of the slot: re-binding it at the new offset needs it, which Metal does not pass here
   (void)offset; (void)index;
   BG_LOG("compute setBufferOffset: not served\n");
}
- (void)setBuffers:(const id *)b offsets:(const NSUInteger *)o withRange:(NSRange)r
{ for (NSUInteger k = 0; k < r.length; k++) [self setBuffer:b[k] offset:o[k] atIndex:r.location + k]; }
- (void)setTexture:(id)t atIndex:(NSUInteger)i { bg_bind_texture(_cb, CHL_STAGE_COMPUTE, i, t); if (t) [_cb bgWrote:(BGMTLTexture *)t]; }
- (void)setTextures:(const id *)t withRange:(NSRange)r { for (NSUInteger k = 0; k < r.length; k++) [self setTexture:t[k] atIndex:r.location + k]; }
- (void)setSamplerState:(id)s atIndex:(NSUInteger)i { bg_bind_sampler(_cb, CHL_STAGE_COMPUTE, i, s); }
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
   if (_cb->_ccb && index < BG_MAX_BINDINGS) chl_set_bytes(_cb->_ccb, CHL_STAGE_COMPUTE, (unsigned)index, bytes, length);
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index
{
   if (_cb->_ccb) chl_set_threadgroup_memory(_cb->_ccb, (unsigned)index, length);
}
- (void)dispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)size
{
   if (!_pipeline || !_cb->_ccb) { BG_LOG("dispatch with no pipeline\n"); return; }
   unsigned g[3] = { (unsigned)groups.width, (unsigned)groups.height, (unsigned)groups.depth };
   unsigned t[3] = { (unsigned)size.width, (unsigned)size.height, (unsigned)size.depth };
   char why[512] = { 0 };
   if (chl_dispatch(_cb->_ccb, g, t, why, sizeof(why)) != 0) [_cb bgSkip:"dispatch" why:why];
}
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)size
{
   MTLSize g = MTLSizeMake((threads.width + size.width - 1) / size.width, (threads.height + size.height - 1) / size.height,
                           (threads.depth + size.depth - 1) / size.depth);
   [self dispatchThreadgroups:g threadsPerThreadgroup:size];   // non-uniform tail threads run too; kernels guard with gid < n
}
- (void)useResource:(id)r usage:(NSUInteger)u { (void)r; (void)u; }
- (void)useResources:(const id *)r count:(NSUInteger)n usage:(NSUInteger)u { (void)r; (void)n; (void)u; }
- (void)useHeap:(id)h { (void)h; }
- (void)memoryBarrierWithScope:(NSUInteger)s { (void)s; }   // chl orders the dispatches of a compute encoder already
- (void)memoryBarrierWithResources:(const id *)r count:(NSUInteger)n { (void)r; (void)n; }
- (void)endEncoding { if (_cb->_ccb) chl_compute_end(_cb->_ccb); [super endEncoding]; }
@end

@implementation BGMTLBlitEncoder
- (instancetype)initWithBGCommandBuffer:(BGMTLCommandBuffer *)cb
{
   self = [super initWithCommandBuffer:cb];
   if (!self) return nil;
   _cb = cb;
   [cb bgChl];
   return self;
}
- (id)device { return [_cb device]; }
- (void)copyFromBuffer:(id)src sourceOffset:(NSUInteger)so toBuffer:(id)dst destinationOffset:(NSUInteger)doff size:(NSUInteger)size
{
   if (!_cb->_ccb || !src || !dst || !size) return;
   [_cb bgUses:src];
   chl_copy_buffer(_cb->_ccb, [(BGMTLBuffer *)src chlBuffer], so, [(BGMTLBuffer *)dst chlBuffer], doff, size);
}
- (void)fillBuffer:(id)buffer range:(NSRange)range value:(uint8_t)value
{
   if (!_cb->_ccb || !buffer || !range.length) return;
   chl_fill_buffer(_cb->_ccb, [(BGMTLBuffer *)buffer chlBuffer], range.location, range.length, value);
}
- (void)copyFromTexture:(id)src sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)size
             toTexture:(id)dst destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dl destinationOrigin:(MTLOrigin)dorg
{
   if (!_cb->_ccb || !src || !dst) return;
   const unsigned o[3] = { (unsigned)so.x, (unsigned)so.y, (unsigned)so.z }, sz[3] = { (unsigned)size.width, (unsigned)size.height, (unsigned)size.depth };
   const unsigned d[3] = { (unsigned)dorg.x, (unsigned)dorg.y, (unsigned)dorg.z };
   if (chl_copy_texture(_cb->_ccb, [(BGMTLTexture *)src chlTextureFor:_cb], (unsigned)ss, (unsigned)sl, o, sz,
                        [(BGMTLTexture *)dst chlTextureFor:_cb], (unsigned)ds, (unsigned)dl, d) != 0)
      [_cb bgSkip:"texture copy" why:"the two formats are not the same size per texel"];
   [_cb bgWrote:dst];
}
- (void)copyFromTexture:(id)src toTexture:(id)dst
{
   NSUInteger levels = [(id<MTLTexture>)src mipmapLevelCount], slices = [(id<MTLTexture>)src arrayLength];
   if (slices < 1) slices = 1;
   for (NSUInteger l = 0; l < levels; l++) {
      MTLSize sz = MTLSizeMake(MAX(1, [(id<MTLTexture>)src width] >> l), MAX(1, [(id<MTLTexture>)src height] >> l), MAX(1, [(id<MTLTexture>)src depth] >> l));
      for (NSUInteger s = 0; s < slices; s++)
         [self copyFromTexture:src sourceSlice:s sourceLevel:l sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:sz
                     toTexture:dst destinationSlice:s destinationLevel:l destinationOrigin:MTLOriginMake(0, 0, 0)];
   }
}
// whole levels of some slices (Mercury on iOS 26); each level is the source level's size
- (void)copyFromTexture:(id)src sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl toTexture:(id)dst destinationSlice:(NSUInteger)ds
       destinationLevel:(NSUInteger)dl sliceCount:(NSUInteger)sc levelCount:(NSUInteger)lc
{
   for (NSUInteger l = 0; l < lc; l++) {
      NSUInteger lvl = sl + l;
      MTLSize sz = MTLSizeMake(MAX(1, [(id<MTLTexture>)src width] >> lvl), MAX(1, [(id<MTLTexture>)src height] >> lvl),
                               MAX(1, [(id<MTLTexture>)src depth] >> lvl));
      for (NSUInteger s = 0; s < sc; s++)
         [self copyFromTexture:src sourceSlice:ss + s sourceLevel:lvl sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:sz
                     toTexture:dst destinationSlice:ds + s destinationLevel:dl + l destinationOrigin:MTLOriginMake(0, 0, 0)];
   }
}
- (void)copyFromBuffer:(id)buf sourceOffset:(NSUInteger)so sourceBytesPerRow:(NSUInteger)bpr sourceBytesPerImage:(NSUInteger)bpi sourceSize:(MTLSize)size
             toTexture:(id)dst destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dl destinationOrigin:(MTLOrigin)dorg
{
   if (!_cb->_ccb || !buf || !dst) return;
   [_cb bgUses:buf];
   const unsigned sz[3] = { (unsigned)size.width, (unsigned)size.height, (unsigned)size.depth }, d[3] = { (unsigned)dorg.x, (unsigned)dorg.y, (unsigned)dorg.z };
   chl_copy_buffer_to_texture(_cb->_ccb, [(BGMTLBuffer *)buf chlBuffer], so, bpr, bpi, sz, [(BGMTLTexture *)dst chlTextureFor:_cb], (unsigned)ds,
                              (unsigned)dl, d);
   [_cb bgWrote:dst];
}
- (void)copyFromBuffer:(id)buf sourceOffset:(NSUInteger)so sourceBytesPerRow:(NSUInteger)bpr sourceBytesPerImage:(NSUInteger)bpi sourceSize:(MTLSize)size
             toTexture:(id)dst destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dl destinationOrigin:(MTLOrigin)dorg options:(NSUInteger)opt
{
   (void)opt;
   [self copyFromBuffer:buf sourceOffset:so sourceBytesPerRow:bpr sourceBytesPerImage:bpi sourceSize:size toTexture:dst destinationSlice:ds destinationLevel:dl destinationOrigin:dorg];
}
- (void)copyFromTexture:(id)src sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)size
               toBuffer:(id)buf destinationOffset:(NSUInteger)doff destinationBytesPerRow:(NSUInteger)bpr destinationBytesPerImage:(NSUInteger)bpi
{
   if (!_cb->_ccb || !src || !buf) return;
   const unsigned o[3] = { (unsigned)so.x, (unsigned)so.y, (unsigned)so.z }, sz[3] = { (unsigned)size.width, (unsigned)size.height, (unsigned)size.depth };
   chl_copy_texture_to_buffer(_cb->_ccb, [(BGMTLTexture *)src chlTextureFor:_cb], (unsigned)ss, (unsigned)sl, o, sz, [(BGMTLBuffer *)buf chlBuffer], doff,
                              bpr, bpi);
}
- (void)copyFromTexture:(id)src sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)size
               toBuffer:(id)buf destinationOffset:(NSUInteger)doff destinationBytesPerRow:(NSUInteger)bpr destinationBytesPerImage:(NSUInteger)bpi options:(NSUInteger)opt
{
   (void)opt;
   [self copyFromTexture:src sourceSlice:ss sourceLevel:sl sourceOrigin:so sourceSize:size toBuffer:buf destinationOffset:doff destinationBytesPerRow:bpr destinationBytesPerImage:bpi];
}
// shared memory: nothing to flush
- (void)synchronizeResource:(id)r { (void)r; }
- (void)synchronizeTexture:(id)t slice:(NSUInteger)s level:(NSUInteger)l { (void)t; (void)s; (void)l; }
- (void)generateMipmapsForTexture:(id)t
{
   if (!_cb->_ccb || !t) return;
   chl_generate_mipmaps(_cb->_ccb, [(BGMTLTexture *)t chlTextureFor:_cb]);
   [_cb bgWrote:t];
}
- (void)optimizeContentsForGPUAccess:(id)t { (void)t; }
- (void)optimizeContentsForCPUAccess:(id)t { (void)t; }
- (void)endEncoding { [super endEncoding]; }
@end


// ---- texture / sampler / render pipeline ----------------------------------------------------------------------------
@implementation BGMTLTexture
+ (void)load { ensure_protocol(self, "MTLTexture"); ensure_protocol(self, "MTLTextureImplementation"); }
- (instancetype)initWithBGDevice:(BGMTLDevice *)device descriptor:(MTLTextureDescriptor *)d
{
   return [self initWithBGDevice:device descriptor:d iosurface:NULL plane:0];
}
- (instancetype)initWithBGDevice:(BGMTLDevice *)device descriptor:(MTLTextureDescriptor *)d iosurface:(void *)surface plane:(NSUInteger)plane
{
   BG_TRACE("texture %lux%lu fmt %lu type %lu usage %lu storage %lu %s\n", (unsigned long)d.width, (unsigned long)d.height,
            (unsigned long)d.pixelFormat, (unsigned long)d.textureType, (unsigned long)d.usage, (unsigned long)d.storageMode,
            surface ? "iosurface" : "own memory");
   self = [super init];
   if (!self) return nil;
   if (![device chl] || !d) { [self release]; return nil; }
   _desc.pixel_format = (unsigned)[d pixelFormat];
   _desc.type = (unsigned)[d textureType];
   _desc.width = (unsigned)[d width];
   _desc.height = (unsigned)[d height];
   _desc.depth = (unsigned)[d depth];
   _desc.array_length = (unsigned)[d arrayLength];
   _desc.mip_levels = (unsigned)[d mipmapLevelCount];
   _desc.sample_count = (unsigned)[d sampleCount];
   _desc.usage = (unsigned)[d usage];
   // MTLTextureDescriptor.swizzle (QuartzCore 26 forces alpha to one for opaque content this way)
   _swizzle = [d swizzle];
   _desc.swizzle[0] = (unsigned char)_swizzle.red; _desc.swizzle[1] = (unsigned char)_swizzle.green;
   _desc.swizzle[2] = (unsigned char)_swizzle.blue; _desc.swizzle[3] = (unsigned char)_swizzle.alpha;
   char why[512] = { 0 };
   if (surface) {
      IOSurfaceRef sf = (IOSurfaceRef)surface;
      _desc.usage |= 5;   /* a drawable is rendered into and may be sampled whatever CA asked for (RenderTarget | ShaderRead) */
      /* a CAMetalLayer drawable: CoreAnimation made the IOSurface and sends the device this one message; the texture IS that memory */
      if (plane == 0 && IOSurfaceGetPlaneCount(sf) <= 1) {
         _tex = chl_texture_from_iosurface([device chl], &_desc, (int)IOSurfaceGetID(sf), why, sizeof(why));
         static unsigned refusals;   // the first few refused imports, in bgmtl.log (which surfaces are copied instead)
         if (!_tex && __atomic_add_fetch(&refusals, 1, __ATOMIC_RELAXED) <= 8) {
            char line[640];
            snprintf(line, sizeof(line), "IOSurface %u (%ux%u fmt %lu) copied, not imported: %s", IOSurfaceGetID(sf), _desc.width,
                     _desc.height, (unsigned long)[d pixelFormat], why);
            bgmtl_note_line(line);
         }
      }
      /* what cannot be imported (a pitch this GPU does not lay a linear image over, a second plane) gets its own memory and the
       * surface's pixels are copied in at each use; good for sampling, which is all CA does with those */
      if (!_tex && [d pixelFormat] == 550 && IOSurfaceGetPlaneCount(sf) >= 2) {
         /* Chlorophyll: CoreAnimation's backing stores of transparent layers are two-plane MTLPixelFormatRGB10A8_2P_XR10
          * surfaces: XR10 RGB in plane 0, 8-bit alpha in plane 1. The texture gets its own BGRA8 memory (550's stand-in) and the
          * surface is decoded into it when it changes. */
         char why2[256];
         _tex = chl_texture_create([device chl], &_desc, why2, sizeof(why2));
         if (_tex) { _surfCopy = YES; _xr2p = YES; }
      } else if (!_tex) {
         char why2[256];
         _tex = chl_texture_create([device chl], &_desc, why2, sizeof(why2));
         if (_tex) _surfCopy = YES;
      }
      if (bg_xr10log() && [d pixelFormat] == 550)
         bg_xr10_note("xr10 wrap surface %u tex %p %ux%u planes %zu usage 0x%lx: %s", IOSurfaceGetID(sf), (void *)self, _desc.width,
                      _desc.height, IOSurfaceGetPlaneCount(sf), (unsigned long)[d usage],
                      !_tex ? "no texture" : _xr2p ? "own memory, XR10 decode/write-back" : _surfCopy ? "own memory, plain copy" : "imported");
      _surface = CFRetain(surface);
      _plane = plane;
   } else {
      _tex = bg_texpool_take([device chl], &_desc);
      if (!_tex) _tex = chl_texture_create([device chl], &_desc, why, sizeof(why));
   }
   if (!_tex) {
      BG_LOG("texture refused: %s\n", why);
      [self release];
      return nil;
   }
   _owner = device;
   if (!surface || _surfCopy) {
      __atomic_add_fetch(&g_bg_tex_n, 1, __ATOMIC_RELAXED);
      __atomic_add_fetch(&g_bg_tex_bytes, bg_tex_bytes(&_desc), __ATOMIC_RELAXED);
   }
   return self;
}

// ---- texture views: the parent's memory as another format, level range, slice range or swizzle ----
- (id)bgViewWithPixelFormat:(NSUInteger)pf textureType:(NSUInteger)tt levels:(NSRange)lv slices:(NSRange)sl swizzle:(MTLTextureSwizzleChannels)sw
{
   BG_TRACE("texture view of %p: fmt %lu type %lu levels %lu+%lu slices %lu+%lu\n", self, (unsigned long)pf, (unsigned long)tt,
            (unsigned long)lv.location, (unsigned long)lv.length, (unsigned long)sl.location, (unsigned long)sl.length);
   const unsigned char s4[4] = { (unsigned char)sw.red, (unsigned char)sw.green, (unsigned char)sw.blue, (unsigned char)sw.alpha };
   char why[256] = { 0 };
   struct chl_texture *v = chl_texture_view(_tex, (unsigned)pf, (unsigned)tt, (unsigned)lv.location, (unsigned)lv.length,
                                            (unsigned)sl.location, (unsigned)sl.length, s4, why, sizeof(why));
   if (!v) { BG_LOG("texture view refused: %s\n", why); return nil; }
   BGMTLTexture *t = [[BGMTLTexture alloc] init];
   if (!t) { chl_texture_release(v); return nil; }
   t->_tex = v;
   t->_owner = _owner;
   t->_parent = [self retain];
   t->_parentLevel = lv.location;
   t->_parentSlice = sl.location;
   t->_swizzle = sw;
   t->_desc = _desc;
   t->_desc.pixel_format = (unsigned)pf;
   t->_desc.type = (unsigned)tt;
   t->_desc.width = MAX(1u, _desc.width >> lv.location);
   t->_desc.height = MAX(1u, _desc.height >> lv.location);
   t->_desc.depth = MAX(1u, _desc.depth >> lv.location);
   t->_desc.mip_levels = (unsigned)lv.length;
   t->_desc.array_length = (unsigned)MAX(sl.length, 1u);
   return t;
}
- (id)newTextureViewWithPixelFormat:(NSUInteger)pf
{
   return [self bgViewWithPixelFormat:pf textureType:_desc.type levels:NSMakeRange(0, MAX(_desc.mip_levels, 1u))
                               slices:NSMakeRange(0, MAX(_desc.array_length, 1u)) swizzle:MTLTextureSwizzleChannelsDefault];
}
- (id)newTextureViewWithPixelFormat:(NSUInteger)pf textureType:(NSUInteger)tt levels:(NSRange)lv slices:(NSRange)sl
{
   return [self bgViewWithPixelFormat:pf textureType:tt levels:lv slices:sl swizzle:MTLTextureSwizzleChannelsDefault];
}
- (id)newTextureViewWithPixelFormat:(NSUInteger)pf textureType:(NSUInteger)tt levels:(NSRange)lv slices:(NSRange)sl
                            swizzle:(MTLTextureSwizzleChannels)sw
{
   return [self bgViewWithPixelFormat:pf textureType:tt levels:lv slices:sl swizzle:sw];
}
- (id)newTextureViewWithDescriptor:(MTLTextureViewDescriptor *)d
{
   if (!d) return nil;
   return [self bgViewWithPixelFormat:[d pixelFormat] textureType:[d textureType] levels:[d levelRange] slices:[d sliceRange]
                              swizzle:[d swizzle]];
}
- (MTLTextureSwizzleChannels)swizzle { return _swizzle; }
- (NSUInteger)parentRelativeLevel { return _parentLevel; }
- (NSUInteger)parentRelativeSlice { return _parentSlice; }
- (void)dealloc
{
   if (_tex && (_parent || _surface || _linSrc || _ncBytes || !bg_texpool_put([_owner chl], _tex, &_desc)))
      chl_texture_release(_tex);
   if (_tex && !_parent && (!_surface || _surfCopy)) {
      __atomic_sub_fetch(&g_bg_tex_n, 1, __ATOMIC_RELAXED);
      __atomic_sub_fetch(&g_bg_tex_bytes, bg_tex_bytes(&_desc), __ATOMIC_RELAXED);
   }
   if (_surface) CFRelease(_surface);
   [_parent release];
   [_linSrc release];
   if (_ncDealloc) { _ncDealloc((void *)_ncBytes, _ncLength); Block_release(_ncDealloc); }
   [super dealloc];
}
- (BGMTLTexture *)bgRoot { BGMTLTexture *t = self; while (t->_parent) t = t->_parent; return t; }
- (void *)iosurface { return _parent ? [_parent iosurface] : (void *)_surface; }
- (NSUInteger)iosurfacePlane { return _plane; }
- (BOOL)isDrawable { return _surface != NULL; }
- (struct chl_texture *)chlTexture { return _tex; }

// The texture for a command buffer that is about to use it, with its pixels brought up to date from wherever they come from:
// another process's CPU writes into an IOSurface (a cache clean when the surface's seed moved), a surface that could not be
// imported (copied in), a two-plane XR10 surface (decoded), caller memory (no-copy) or a buffer (linear texture, once per
// command buffer and not while a command buffer that writes it is in flight). The copies go through chl_texture_replace, which
// the GPU runs before this command buffer.
- (struct chl_texture *)chlTextureFor:(BGMTLCommandBuffer *)cb
{
   if (_parent) { [_parent chlTextureFor:cb]; return _tex; }
   char why[256] = { 0 };
   if (_surface && !_surfCopy) {
      static NSMutableDictionary *seeds;
      static os_unfair_lock lk = OS_UNFAIR_LOCK_INIT;
      NSNumber *key = @(IOSurfaceGetID((IOSurfaceRef)_surface));
      uint32_t seed = IOSurfaceGetSeed((IOSurfaceRef)_surface);
      os_unfair_lock_lock(&lk);
      if (!seeds) seeds = [[NSMutableDictionary alloc] init];
      NSNumber *was = seeds[key];
      BOOL changed = !was || [was unsignedIntValue] != seed;
      if (changed) seeds[key] = @(seed);
      if ([seeds count] > 4096) [seeds removeAllObjects];
      os_unfair_lock_unlock(&lk);
      if (changed) chl_texture_flush_surface(_tex);
   } else if (_xr2p && _surface) {
      IOSurfaceRef sf = (IOSurfaceRef)_surface;
      uint32_t seed = IOSurfaceGetSeed(sf);
      _xrUses++;
      if ((!_surfFresh || seed != _surfSeed) && IOSurfaceLock(sf, kIOSurfaceLockReadOnly, NULL) == 0) {
         // plane 0: one little-endian word per texel, B bits 0-9, G 10-19, R 20-29 in XR10 (384 = 0.0, 894 = 1.0, the extended
         // range clamped here); plane 1: one alpha byte per texel. Out: BGRA8, the stand-in format of 550.
         const uint8_t *p0 = IOSurfaceGetBaseAddressOfPlane(sf, 0), *p1 = IOSurfaceGetBaseAddressOfPlane(sf, 1);
         size_t r0 = IOSurfaceGetBytesPerRowOfPlane(sf, 0), r1 = IOSurfaceGetBytesPerRowOfPlane(sf, 1);
         unsigned w = MIN(_desc.width, (unsigned)IOSurfaceGetWidthOfPlane(sf, 0)), h = MIN(_desc.height, (unsigned)IOSurfaceGetHeightOfPlane(sf, 0));
         uint8_t *out = (p0 && p1 && w && h) ? malloc((size_t)w * h * 4) : NULL;
         if (out) {
            static uint8_t lut[1024];
            static int lut_ready;
            if (!lut_ready) {
               for (int v = 0; v < 1024; v++) { int c = (v - 384) * 255 / 510; lut[v] = (uint8_t)(c < 0 ? 0 : c > 255 ? 255 : c); }
               lut_ready = 1;
            }
            for (unsigned y = 0; y < h; y++) {
               const uint8_t *s0 = p0 + y * r0, *s1 = p1 + y * r1;
               uint8_t *o = out + (size_t)y * w * 4;
               for (unsigned x = 0; x < w; x++) {
                  uint32_t v = (uint32_t)s0[4 * x] | (uint32_t)s0[4 * x + 1] << 8 | (uint32_t)s0[4 * x + 2] << 16 | (uint32_t)s0[4 * x + 3] << 24;
                  o[4 * x] = lut[v & 0x3ff];
                  o[4 * x + 1] = lut[(v >> 10) & 0x3ff];
                  o[4 * x + 2] = lut[(v >> 20) & 0x3ff];
                  o[4 * x + 3] = s1[x];
               }
            }
         }
         IOSurfaceUnlock(sf, kIOSurfaceLockReadOnly, NULL);
         if (out) {
            const unsigned o[3] = { 0, 0, 0 }, sz[3] = { w, h, 1 };
            if (chl_texture_replace(_tex, 0, 0, o, sz, out, w * 4, 0, why, sizeof(why)) != 0) BG_LOG("XR10 surface refresh: %s\n", why);
            if (bg_xr10log())
               bg_xr10_note("xr10 decode surface %u tex %p %ux%u seed %u (had %u, used %u times since) alpha %u", IOSurfaceGetID(sf),
                            (void *)self, w, h, seed, _surfSeed, _xrUses - 1, bg_alpha_count(out, (size_t)w * h));
            _xrUses = 0;
            free(out);
            _surfSeed = seed;
            _surfFresh = YES;
         }
      }
   } else if (_surfCopy && _surface) {
      IOSurfaceRef sf = (IOSurfaceRef)_surface;
      uint32_t seed = IOSurfaceGetSeed(sf);
      if ((!_surfFresh || seed != _surfSeed) && IOSurfaceLock(sf, kIOSurfaceLockReadOnly, NULL) == 0) {
         const unsigned o[3] = { 0, 0, 0 }, sz[3] = { _desc.width, _desc.height, 1 };
         if (chl_texture_replace(_tex, 0, 0, o, sz, IOSurfaceGetBaseAddress(sf), IOSurfaceGetBytesPerRow(sf), 0, why, sizeof(why)) != 0)
            BG_LOG("surface copy refresh: %s\n", why);
         IOSurfaceUnlock(sf, kIOSurfaceLockReadOnly, NULL);
         _surfSeed = seed;
         _surfFresh = YES;
      }
   }
   if (_ncBytes) {
      const unsigned o[3] = { 0, 0, 0 }, sz[3] = { _desc.width, _desc.height, 1 };
      if (chl_texture_replace(_tex, 0, 0, o, sz, _ncBytes, _ncLength / (_desc.height ? _desc.height : 1), 0, why, sizeof(why)) != 0)
         BG_LOG("no-copy texture refresh: %s\n", why);
   }
   if (_linSrc && cb && __atomic_load_n(&_linPending, __ATOMIC_ACQUIRE) == 0 && _linSeq != cb->_seq) {
      _linSeq = cb->_seq;
      const unsigned o[3] = { 0, 0, 0 }, sz[3] = { _desc.width, _desc.height, 1 };
      if (chl_texture_replace(_tex, 0, 0, o, sz, (const uint8_t *)[(BGMTLBuffer *)_linSrc contents] + _linOff, _linBpr, 0, why,
                              sizeof(why)) != 0)
         BG_LOG("linear texture refresh: %s\n", why);
   }
   return _tex;
}

// Chlorophyll: the texture's pixels back into its two-plane RGB10A8_2P_XR10 surface (the encoding chlTextureFor: decodes: XR10
// 384 + 2c per 8-bit channel, exact both ways; alpha into plane 1), after a command buffer drew into it.
- (BOOL)bgXR10WriteBack
{
   if (_parent) return [(BGMTLTexture *)_parent bgXR10WriteBack];
   if (!_xr2p || !_tex || !_surface) return NO;
   IOSurfaceRef sf = (IOSurfaceRef)_surface;
   unsigned w = MIN(_desc.width, (unsigned)IOSurfaceGetWidthOfPlane(sf, 0)), h = MIN(_desc.height, (unsigned)IOSurfaceGetHeightOfPlane(sf, 0));
   uint8_t *px = (w && h) ? malloc((size_t)w * h * 4) : NULL;
   if (!px) return YES;
   char why[256] = { 0 };
   const unsigned o[3] = { 0, 0, 0 }, sz[3] = { w, h, 1 };
   if (chl_texture_get(_tex, 0, 0, o, sz, px, w * 4, 0, why, sizeof(why)) != 0) { BG_LOG("XR10 write-back: %s\n", why); free(px); return YES; }
   if (IOSurfaceLock(sf, 0, NULL) == 0) {
      uint8_t *p0 = IOSurfaceGetBaseAddressOfPlane(sf, 0), *p1 = IOSurfaceGetBaseAddressOfPlane(sf, 1);
      size_t r0 = IOSurfaceGetBytesPerRowOfPlane(sf, 0), r1 = IOSurfaceGetBytesPerRowOfPlane(sf, 1);
      for (unsigned y = 0; p0 && p1 && y < h; y++) {
         const uint8_t *s = px + (size_t)y * w * 4;
         uint8_t *d0 = p0 + y * r0, *d1 = p1 + y * r1;
         for (unsigned x = 0; x < w; x++) {
            uint32_t v = (384u + 2u * s[4 * x]) | (384u + 2u * s[4 * x + 1]) << 10 | (384u + 2u * s[4 * x + 2]) << 20;
            d0[4 * x] = (uint8_t)v; d0[4 * x + 1] = (uint8_t)(v >> 8); d0[4 * x + 2] = (uint8_t)(v >> 16); d0[4 * x + 3] = (uint8_t)(v >> 24);
            d1[x] = s[4 * x + 3];
         }
      }
      IOSurfaceUnlock(sf, 0, NULL);
      _surfSeed = IOSurfaceGetSeed(sf);   // what the surface holds now IS the texture: no decode back at the next use
      _surfFresh = YES;
      if (bg_xr10log())
         bg_xr10_note("xr10 write-back surface %u tex %p %ux%u seed %u alpha %u", IOSurfaceGetID(sf), (void *)self, w, h, _surfSeed,
                      bg_alpha_count(px, (size_t)w * h));
   }
   free(px);
   return YES;
}

- (void)bgLinearPending:(int)delta { __atomic_add_fetch(&_linPending, delta, __ATOMIC_ACQ_REL); }
// What the GPU wrote, back into the buffer the texture was made over (the caller reads the buffer: iOS 26's icon renderer
// draws an icon into such a texture and serializes the buffer's bytes). Called once the command buffer has completed.
- (void)bgLinearWriteBack
{
   if (!_linSrc || !_tex) return;
   char why[256] = { 0 };
   const unsigned o[3] = { 0, 0, 0 }, sz[3] = { _desc.width, _desc.height, 1 };
   if (chl_texture_get(_tex, 0, 0, o, sz, (uint8_t *)[(BGMTLBuffer *)_linSrc contents] + _linOff, _linBpr, 0, why, sizeof(why)) != 0)
      BG_LOG("linear texture write-back: %s\n", why);
}
- (id)device { return _owner; }
- (NSUInteger)width { return _desc.width; }
- (NSUInteger)height { return _desc.height; }
- (NSUInteger)depth { return _desc.depth; }
- (NSUInteger)mipmapLevelCount { return _desc.mip_levels; }
- (NSUInteger)arrayLength { return _desc.array_length; }
- (NSUInteger)sampleCount { return _desc.sample_count; }
- (NSUInteger)pixelFormat { return _desc.pixel_format; }
- (NSUInteger)textureType { return _desc.type; }
- (NSUInteger)usage { return _desc.usage; }
- (BOOL)isFramebufferOnly { return NO; }
- (id)parentTexture { return _parent; }
// a texture over a buffer (newTextureWithDescriptor:offset:bytesPerRow: / newLinearTexture...): which buffer, where, how laid out
- (id)buffer { return _linSrc; }
- (NSUInteger)bufferOffset { return _linOff; }
- (NSUInteger)bufferBytesPerRow { return _linBpr; }
- (NSUInteger)storageMode { return 0; }
- (NSUInteger)cpuCacheMode { return 0; }
- (NSUInteger)resourceOptions { return 0; }
- (NSUInteger)hazardTrackingMode { return 0; }
- (NSUInteger)allocatedSize { return 0; }
- (NSUInteger)setPurgeableState:(NSUInteger)state { bg_count_purge(state); return 2; }
- (void)makeAliasable { __atomic_add_fetch(&g_bg_aliasable, 1, __ATOMIC_RELAXED); }
- (id)heap { return nil; }
- (BOOL)isAliasable { return NO; }
- (BOOL)isComplete { return YES; }
- (void)waitUntilComplete {}
- (NSUInteger)protectionOptions { return 0; }
- (void)replaceRegion:(MTLRegion)r mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice withBytes:(const void *)bytes
          bytesPerRow:(NSUInteger)bpr bytesPerImage:(NSUInteger)bpi
{
   BG_TRACE("replaceRegion %lu,%lu %lux%lu level %lu bpr %lu\n", (unsigned long)r.origin.x, (unsigned long)r.origin.y,
            (unsigned long)r.size.width, (unsigned long)r.size.height, (unsigned long)level, (unsigned long)bpr);
   const unsigned o[3] = { (unsigned)r.origin.x, (unsigned)r.origin.y, (unsigned)r.origin.z };
   const unsigned sz[3] = { (unsigned)r.size.width, (unsigned)r.size.height, (unsigned)r.size.depth };
   char why[256] = { 0 };
   if (chl_texture_replace(_tex, (unsigned)level, (unsigned)slice, o, sz, bytes, bpr, bpi, why, sizeof(why)) != 0) BG_LOG("replaceRegion: %s\n", why);
}
- (void)replaceRegion:(MTLRegion)r mipmapLevel:(NSUInteger)level withBytes:(const void *)bytes bytesPerRow:(NSUInteger)bpr
{
   [self replaceRegion:r mipmapLevel:level slice:0 withBytes:bytes bytesPerRow:bpr bytesPerImage:0];
}
- (void)getBytes:(void *)bytes bytesPerRow:(NSUInteger)bpr bytesPerImage:(NSUInteger)bpi fromRegion:(MTLRegion)r
     mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice
{
   const unsigned o[3] = { (unsigned)r.origin.x, (unsigned)r.origin.y, (unsigned)r.origin.z };
   const unsigned sz[3] = { (unsigned)r.size.width, (unsigned)r.size.height, (unsigned)r.size.depth };
   char why[256] = { 0 };
   if (chl_texture_get(_tex, (unsigned)level, (unsigned)slice, o, sz, bytes, bpr, bpi, why, sizeof(why)) != 0) BG_LOG("getBytes: %s\n", why);
}
- (void)getBytes:(void *)bytes bytesPerRow:(NSUInteger)bpr fromRegion:(MTLRegion)r mipmapLevel:(NSUInteger)level
{
   [self getBytes:bytes bytesPerRow:bpr bytesPerImage:0 fromRegion:r mipmapLevel:level slice:0];
}
@end

@implementation BGMTLSampler
+ (void)load { ensure_protocol(self, "MTLSamplerState"); }
- (instancetype)initWithBGDevice:(BGMTLDevice *)device descriptor:(MTLSamplerDescriptor *)d
{
   self = [super initWithDevice:device samplerDescriptor:d];
   if (!self) return nil;
   struct chl_sampler_desc st;
   memset(&st, 0, sizeof(st));
   st.address[0] = (unsigned)[d sAddressMode]; st.address[1] = (unsigned)[d tAddressMode]; st.address[2] = (unsigned)[d rAddressMode];
   st.min_filter = (unsigned)[d minFilter]; st.mag_filter = (unsigned)[d magFilter]; st.mip_filter = (unsigned)[d mipFilter];
   st.compare_function = (unsigned)[d compareFunction];
   st.max_anisotropy = (unsigned)[d maxAnisotropy];
   st.normalized_coordinates = [d normalizedCoordinates];
   st.lod_min = [d lodMinClamp]; st.lod_max = [d lodMaxClamp];
   if ([d respondsToSelector:@selector(borderColor)]) st.border_color = (unsigned)[d borderColor];
   char why[256] = { 0 };
   _smp = chl_sampler_create([device chl], &st, why, sizeof(why));
   if (!_smp) { BG_LOG("sampler refused: %s\n", why); [self release]; return nil; }
   _owner = device;
   return self;
}
- (void)dealloc { chl_sampler_release(_smp); [super dealloc]; }
- (struct chl_sampler *)chlSampler { return _smp; }
- (id)device { return _owner; }
@end

@implementation BGMTLDepthStencil
+ (void)load { ensure_protocol(self, "MTLDepthStencilState"); }
- (instancetype)initWithBGDevice:(BGMTLDevice *)device descriptor:(MTLDepthStencilDescriptor *)d
{
   self = [super initWithDevice:device depthStencilDescriptor:d];
   if (!self) return nil;
   struct chl_depth_stencil_desc st;
   memset(&st, 0, sizeof(st));
   st.depth_compare = (unsigned)[d depthCompareFunction];
   st.depth_write = [d isDepthWriteEnabled];
   MTLStencilDescriptor *f = [d frontFaceStencil], *b = [d backFaceStencil];
   st.stencil_enabled = f != nil || b != nil;
   st.front = (struct chl_stencil_face){ f ? (unsigned)[f stencilCompareFunction] : 7, (unsigned)[f stencilFailureOperation],
                                         (unsigned)[f depthFailureOperation], (unsigned)[f depthStencilPassOperation],
                                         f ? (unsigned)[f readMask] : 0xff, f ? (unsigned)[f writeMask] : 0xff };
   st.back = (struct chl_stencil_face){ b ? (unsigned)[b stencilCompareFunction] : 7, (unsigned)[b stencilFailureOperation],
                                        (unsigned)[b depthFailureOperation], (unsigned)[b depthStencilPassOperation],
                                        b ? (unsigned)[b readMask] : 0xff, b ? (unsigned)[b writeMask] : 0xff };
   char why[256] = { 0 };
   _ds = chl_depth_stencil_create([device chl], &st, why, sizeof(why));
   if (!_ds) { BG_LOG("depth/stencil state refused: %s\n", why); [self release]; return nil; }
   _owner = device;
   _readsDepth = st.depth_compare != MTLCompareFunctionAlways;
   _writesDepth = st.depth_write;
   _readsStencil = st.front.compare != MTLCompareFunctionAlways || st.back.compare != MTLCompareFunctionAlways;
   _writesStencil = (f && f.writeMask) || (b && b.writeMask);
   return self;
}
- (void)dealloc { chl_depth_stencil_release(_ds); [super dealloc]; }
- (struct chl_depth_stencil *)chlState { return _ds; }
- (id)device { return _owner; }
- (BOOL)readsDepth { return _readsDepth; }
- (BOOL)writesDepth { return _writesDepth; }
- (BOOL)readsStencil { return _readsStencil; }
- (BOOL)writesStencil { return _writesStencil; }
@end

@implementation BGMTLRenderPipeline
+ (void)load { ensure_protocol(self, "MTLRenderPipelineState"); }
- (void)dealloc { chl_render_pipeline_release(_p); [super dealloc]; }
- (id)device { return _owner; }
@end

@implementation BGMTLRenderEncoder
static void
bg_attachment(BGMTLCommandBuffer *cb, MTLRenderPassAttachmentDescriptor *a, struct chl_attachment *out)
{
   out->texture = [(BGMTLTexture *)[a texture] chlTextureFor:cb];
   out->level = (unsigned)[a level];
   out->slice = (unsigned)([a slice] + [a depthPlane]);
   out->load_action = (unsigned)[a loadAction];
   out->store_action = (unsigned)[a storeAction];
   if ((out->store_action == 2 || out->store_action == 3) && [a resolveTexture]) {
      out->resolve = [(BGMTLTexture *)[a resolveTexture] chlTextureFor:cb];
      out->resolve_level = (unsigned)[a resolveLevel];
      out->resolve_slice = (unsigned)([a resolveSlice] + [a resolveDepthPlane]);
      [cb bgWrote:(BGMTLTexture *)[a resolveTexture]];
   }
   [cb bgWrote:(BGMTLTexture *)[a texture]];
}
- (instancetype)initWithBGCommandBuffer:(BGMTLCommandBuffer *)cb descriptor:(MTLRenderPassDescriptor *)d
{
   self = [super initWithCommandBuffer:cb];
   if (!self) return nil;
   _cb = cb;
   if (![cb bgChl]) return self;
   struct chl_render_pass rp;
   memset(&rp, 0, sizeof(rp));
   for (unsigned i = 0; i < 8; i++) {
      MTLRenderPassColorAttachmentDescriptor *a = [[d colorAttachments] objectAtIndexedSubscript:i];
      if (!a || ![a texture]) continue;
      bg_attachment(cb, a, &rp.color[i]);
      MTLClearColor c = [a clearColor];
      rp.color[i].clear[0] = (float)c.red; rp.color[i].clear[1] = (float)c.green; rp.color[i].clear[2] = (float)c.blue;
      rp.color[i].clear[3] = (float)c.alpha;
   }
   MTLRenderPassDepthAttachmentDescriptor *da = [d depthAttachment];
   if (da && [da texture]) {
      bg_attachment(cb, da, &rp.depth);
      rp.depth.clear[0] = (float)[da clearDepth];
   }
   MTLRenderPassStencilAttachmentDescriptor *sa = [d stencilAttachment];
   if (sa && [sa texture]) {
      bg_attachment(cb, sa, &rp.stencil);
      rp.stencil.clear_stencil = (unsigned)[sa clearStencil];
   }
   if ([d respondsToSelector:@selector(renderTargetWidth)]) { rp.width = (unsigned)[d renderTargetWidth]; rp.height = (unsigned)[d renderTargetHeight]; }
   char why[512] = { 0 };
   _open = chl_render_begin(cb->_ccb, &rp, why, sizeof(why)) == 0;
   if (!_open) [cb bgSkip:"render pass" why:why];
   return self;
}
- (id)device { return [_cb device]; }
- (void)setRenderPipelineState:(id)s
{
   _pipeline = s;
   if (_open && s) chl_set_render_pipeline(_cb->_ccb, ((BGMTLRenderPipeline *)s)->_p);
   BG_TRACE("setPipeline %s\n", s ? ((BGMTLRenderPipeline *)s)->_names : "-");
}
- (void)setVertexBuffer:(id)b offset:(NSUInteger)o atIndex:(NSUInteger)i { if (i < BG_MAX_BINDINGS) _vbuf[i] = b; bg_bind_buffer(_cb, CHL_STAGE_VERTEX, i, b, o); }
- (void)setFragmentBuffer:(id)b offset:(NSUInteger)o atIndex:(NSUInteger)i { if (i < BG_MAX_BINDINGS) _fbuf[i] = b; bg_bind_buffer(_cb, CHL_STAGE_FRAGMENT, i, b, o); }
// CoreAnimation binds a buffer once per pass and then only moves the offset between draws
- (void)setVertexBufferOffset:(NSUInteger)o atIndex:(NSUInteger)i { if (i < BG_MAX_BINDINGS) bg_bind_buffer(_cb, CHL_STAGE_VERTEX, i, _vbuf[i], o); }
- (void)setFragmentBufferOffset:(NSUInteger)o atIndex:(NSUInteger)i { if (i < BG_MAX_BINDINGS) bg_bind_buffer(_cb, CHL_STAGE_FRAGMENT, i, _fbuf[i], o); }
- (void)setVertexBuffers:(const id *)b offsets:(const NSUInteger *)o withRange:(NSRange)r
{ for (NSUInteger k = 0; k < r.length; k++) [self setVertexBuffer:b[k] offset:o[k] atIndex:r.location + k]; }
- (void)setFragmentBuffers:(const id *)b offsets:(const NSUInteger *)o withRange:(NSRange)r
{ for (NSUInteger k = 0; k < r.length; k++) [self setFragmentBuffer:b[k] offset:o[k] atIndex:r.location + k]; }
- (void)setVertexBytes:(const void *)bytes length:(NSUInteger)l atIndex:(NSUInteger)i
{
   if (_cb->_ccb && i < BG_MAX_BINDINGS) { _vbuf[i] = nil; chl_set_bytes(_cb->_ccb, CHL_STAGE_VERTEX, (unsigned)i, bytes, l); }
}
- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)l atIndex:(NSUInteger)i
{
   if (_cb->_ccb && i < BG_MAX_BINDINGS) { _fbuf[i] = nil; chl_set_bytes(_cb->_ccb, CHL_STAGE_FRAGMENT, (unsigned)i, bytes, l); }
}
- (void)setVertexTexture:(id)t atIndex:(NSUInteger)i { bg_bind_texture(_cb, CHL_STAGE_VERTEX, i, t); }
- (void)setFragmentTexture:(id)t atIndex:(NSUInteger)i
{
   if (bg_trace() && t)
      BG_LOG("setFragmentTexture %p index %lu: %lux%lu fmt %lu surface %u%s\n", t, (unsigned long)i, (unsigned long)[(id<MTLTexture>)t width],
             (unsigned long)[(id<MTLTexture>)t height], (unsigned long)[(id<MTLTexture>)t pixelFormat],
             [(BGMTLTexture *)t iosurface] ? (unsigned)IOSurfaceGetID((IOSurfaceRef)[(BGMTLTexture *)t iosurface]) : 0u,
             [(id<MTLTexture>)t parentTexture] ? " (view)" : "");
   bg_bind_texture(_cb, CHL_STAGE_FRAGMENT, i, t);
}
- (void)setVertexTextures:(const id *)t withRange:(NSRange)r { for (NSUInteger k = 0; k < r.length; k++) [self setVertexTexture:t[k] atIndex:r.location + k]; }
- (void)setFragmentTextures:(const id *)t withRange:(NSRange)r { for (NSUInteger k = 0; k < r.length; k++) [self setFragmentTexture:t[k] atIndex:r.location + k]; }
- (void)setVertexSamplerState:(id)s atIndex:(NSUInteger)i { bg_bind_sampler(_cb, CHL_STAGE_VERTEX, i, s); }
- (void)setFragmentSamplerState:(id)s atIndex:(NSUInteger)i { bg_bind_sampler(_cb, CHL_STAGE_FRAGMENT, i, s); }
- (void)setFragmentSamplerStates:(const id *)s withRange:(NSRange)r { for (NSUInteger k = 0; k < r.length; k++) [self setFragmentSamplerState:s[k] atIndex:r.location + k]; }
- (void)setViewport:(MTLViewport)v
{
   BG_TRACE("setViewport %g %g %g %g z %g..%g\n", v.originX, v.originY, v.width, v.height, v.znear, v.zfar);
   if (_open) chl_set_viewport(_cb->_ccb, v.originX, v.originY, v.width, v.height, v.znear, v.zfar);
}
- (void)setViewports:(const MTLViewport *)v count:(NSUInteger)n { if (n) [self setViewport:v[0]]; }
- (void)setDepthStencilState:(id)state { if (_open) chl_set_depth_stencil(_cb->_ccb, state ? [(BGMTLDepthStencil *)state chlState] : NULL); }
- (void)setStencilReferenceValue:(uint32_t)v { if (_open) chl_set_stencil_reference(_cb->_ccb, v, v); }
- (void)setStencilFrontReferenceValue:(uint32_t)f backReferenceValue:(uint32_t)b { if (_open) chl_set_stencil_reference(_cb->_ccb, f, b); }
- (void)setDepthBias:(float)bias slopeScale:(float)slope clamp:(float)clamp { if (_open) chl_set_depth_bias(_cb->_ccb, bias, slope, clamp); }
- (void)setDepthClipMode:(NSUInteger)m { (void)m; }
- (void)setScissorRect:(MTLScissorRect)r { if (_open) chl_set_scissor(_cb->_ccb, (unsigned)r.x, (unsigned)r.y, (unsigned)r.width, (unsigned)r.height); }
- (void)setScissorRects:(const MTLScissorRect *)r count:(NSUInteger)n { if (n) [self setScissorRect:r[0]]; }
- (void)setCullMode:(NSUInteger)m { _cull = m; if (_open) chl_set_cull(_cb->_ccb, (unsigned)_cull, (unsigned)_winding); }
- (void)setFrontFacingWinding:(NSUInteger)w { _winding = w; if (_open) chl_set_cull(_cb->_ccb, (unsigned)_cull, (unsigned)_winding); }
- (void)setTriangleFillMode:(NSUInteger)m { (void)m; }
- (void)setBlendColorRed:(float)r green:(float)g blue:(float)b alpha:(float)a
{
   const float c[4] = { r, g, b, a };
   if (_open) chl_set_blend_color(_cb->_ccb, c);
}
- (void)useResource:(id)r usage:(NSUInteger)u { (void)r; (void)u; }
- (void)useResource:(id)r usage:(NSUInteger)u stages:(NSUInteger)s { (void)r; (void)u; (void)s; }
- (void)useHeap:(id)h { (void)h; }
- (void)useHeap:(id)h stages:(NSUInteger)s { (void)h; (void)s; }

- (void)bgDraw:(NSUInteger)prim start:(NSUInteger)start count:(NSUInteger)count instances:(NSUInteger)inst
   indexBuffer:(id)ib indexType:(NSUInteger)it indexOffset:(NSUInteger)io baseVertex:(NSInteger)bv baseInstance:(NSUInteger)bi
{
   if (!_open) return;
   if (!_pipeline) { BG_LOG("draw with no pipeline\n"); return; }
   char why[512] = { 0 };
   int rc;
   if (ib) {
      [_cb bgUses:ib];
      rc = chl_draw_indexed(_cb->_ccb, (unsigned)prim, (unsigned)count, (unsigned)it, [(BGMTLBuffer *)ib chlBuffer], io, (unsigned)inst,
                            (int)bv, (unsigned)bi, why, sizeof(why));
   } else {
      rc = chl_draw(_cb->_ccb, (unsigned)prim, (unsigned)start, (unsigned)count, (unsigned)inst, (unsigned)bi, why, sizeof(why));
   }
   if (rc != 0) [_cb bgSkip:_pipeline->_names why:why];
   else _cb->_draws++;
}
- (void)drawPrimitives:(NSUInteger)p vertexStart:(NSUInteger)s vertexCount:(NSUInteger)c
{ [self bgDraw:p start:s count:c instances:1 indexBuffer:nil indexType:0 indexOffset:0 baseVertex:0 baseInstance:0]; }
- (void)drawPrimitives:(NSUInteger)p vertexStart:(NSUInteger)s vertexCount:(NSUInteger)c instanceCount:(NSUInteger)n
{ [self bgDraw:p start:s count:c instances:n indexBuffer:nil indexType:0 indexOffset:0 baseVertex:0 baseInstance:0]; }
- (void)drawPrimitives:(NSUInteger)p vertexStart:(NSUInteger)s vertexCount:(NSUInteger)c instanceCount:(NSUInteger)n baseInstance:(NSUInteger)b
{ [self bgDraw:p start:s count:c instances:n indexBuffer:nil indexType:0 indexOffset:0 baseVertex:0 baseInstance:b]; }
- (void)drawIndexedPrimitives:(NSUInteger)p indexCount:(NSUInteger)c indexType:(NSUInteger)t indexBuffer:(id)b indexBufferOffset:(NSUInteger)o
{ [self bgDraw:p start:0 count:c instances:1 indexBuffer:b indexType:t indexOffset:o baseVertex:0 baseInstance:0]; }
- (void)drawIndexedPrimitives:(NSUInteger)p indexCount:(NSUInteger)c indexType:(NSUInteger)t indexBuffer:(id)b indexBufferOffset:(NSUInteger)o
                instanceCount:(NSUInteger)n
{ [self bgDraw:p start:0 count:c instances:n indexBuffer:b indexType:t indexOffset:o baseVertex:0 baseInstance:0]; }
- (void)drawIndexedPrimitives:(NSUInteger)p indexCount:(NSUInteger)c indexType:(NSUInteger)t indexBuffer:(id)b indexBufferOffset:(NSUInteger)o
                instanceCount:(NSUInteger)n baseVertex:(NSInteger)bv baseInstance:(NSUInteger)bi
{ [self bgDraw:p start:0 count:c instances:n indexBuffer:b indexType:t indexOffset:o baseVertex:bv baseInstance:bi]; }
- (void)endEncoding
{
   if (_open) chl_render_end(_cb->_ccb);   // the pass's resolves happen here (store actions)
   _open = NO;
   [super endEncoding];
}
@end

// ---- GPU state, as the kernel reports it (SWEET_GPU_SEL_INFO): -1 no answer, 0 not ready, 1 ready, 2 LOST ----
struct bg_gpu_info { uint32_t abi_version, flags, gpu_id, chip_id; uint64_t va_start, va_size; uint32_t last_fence, abi_minor; };
extern int sweet_drm_info(int fd, struct bg_gpu_info *out);

int
bgmtl_gpu_state(unsigned *abi_minor)
{
   struct bg_gpu_info in;
   memset(&in, 0, sizeof(in));
   if (sweet_drm_info(-1, &in) != 0)
      return -1;
   if (abi_minor) *abi_minor = in.abi_minor;
   uint32_t f = in.flags & 3;
   return (f & 2) ? 2 : (f & 1) ? 1 : 0;
}

// ---- entry point for test processes ----------------------------------------------------------------------------------
id
bgmtl_create_device(void)
{
   unsigned minor = 0;
   int st = bgmtl_gpu_state(&minor);
   if (st == 2) { BG_LOG("the kernel reports the GPU LOST: no device is made\n"); return nil; }
   if (st == 1) BG_LOG("GPU ready (kernel ABI minor %u)\n", minor);
   BGMTLDevice *d = [[BGMTLDevice alloc] initWithAcceleratorPort:0];
   if (d && ![d chl]) {   // a device that cannot reach the GPU (sandbox, no service) is not a device
      BG_LOG("no GPU behind the device: none is made\n");
      [d release];
      return nil;
   }
   return d;
}

struct chl_device *
bgmtl_chl(id device)
{
   return [(BGMTLDevice *)device chl];
}
