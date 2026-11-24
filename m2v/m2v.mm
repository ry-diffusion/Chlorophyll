/*
 * libm2v.dylib: a Metal device from metal2vulkan (steelbrain/metal2vulkan, GPL-2.0-or-later OR LGPL-3.0-or-later) through
 * ios-guacamole's NativeAdrenoMetal frontend and bridge (MIT, UniXNU via ios-flame), on Chlorophyll's own Turnip
 * (libchlorophyll.dylib's ICD entry points, our kernel through the libdrm stand-in). A trial beside Chlorophyll's own
 * runtime: the inject (bgmtl_inject.m) asks m2v_create_device() instead of bgmtl_create_device() for a process whose
 * /private/var/tmp/bgmtl.m2v.<bundle id> exists, and registers the answer with MTLAddDevice the same way.
 * Log: /private/var/tmp/m2v.log (when it exists), lines "M2V[pid name]: ...".
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <IOSurface/IOSurfaceRef.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "NativeBridgeMetalFrontend.h"

#define M2V_TURNIP "/usr/local/lib/bgmtl/libchlorophyll.dylib"
#define M2V_LOG "/private/var/tmp/m2v.log"

extern "C" void sm2_logs(const char *s)
{
	static int fd = -2;
	if (fd == -2)
		fd = open(M2V_LOG, O_WRONLY | O_APPEND);
	if (fd < 0)
		return;
	char buf[700];
	int n = snprintf(buf, sizeof(buf), "M2V[%d %s]: %s\n", getpid(), getprogname(), s);
	write(fd, buf, n < (int)sizeof(buf) ? (size_t)n : sizeof(buf) - 1);
}

static void m2v_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void m2v_log(const char *fmt, ...)
{
	char buf[600];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	sm2_logs(buf);
}

/* The A630 does not snoop the CPU caches (as guacamole's A640): libSystem's sys_dcache_flush is a no-op on Apple GPUs.
 * User space may clean by VA (XNU sets SCTLR_EL1.UCI). This definition wins at link time for the frontend and bridge. */
extern "C" void sys_dcache_flush(void *start, size_t len)
{
	uintptr_t a = (uintptr_t)start & ~(uintptr_t)63, end = (uintptr_t)start + len;
	for (; a < end; a += 64)
		__asm__ volatile("dc civac, %0" ::"r"(a) : "memory");
	__asm__ volatile("dsb sy" ::: "memory");
}

/* ash's Entry::load_from(path) wants vkGetInstanceProcAddr; our Turnip exports only the ICD entry points. The bridge is
 * given this dylib's own path as its loader, and this forwards to Turnip's vk_icdGetInstanceProcAddr (Mesa answers the
 * global commands for a NULL instance). */
typedef void (*m2v_vkfn)(void);
extern "C" __attribute__((visibility("default"))) m2v_vkfn vkGetInstanceProcAddr(void *instance, const char *name)
{
	static m2v_vkfn (*icd)(void *, const char *);
	if (!icd) {
		void *h = dlopen(M2V_TURNIP, RTLD_NOW | RTLD_GLOBAL);
		icd = h ? (m2v_vkfn(*)(void *, const char *))dlsym(h, "vk_icdGetInstanceProcAddr") : NULL;
		if (!icd)
			m2v_log("no vk_icdGetInstanceProcAddr in %s: %s", M2V_TURNIP, dlerror());
	}
	return icd ? icd(instance, name) : NULL;
}

static void missing(const char *cls, const char *sel)
{
	static NSMutableSet *seen;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		seen = [NSMutableSet new];
	});
	NSString *k = [NSString stringWithFormat:@"%s %s", cls, sel];
	@synchronized(seen) {
		if ([seen containsObject:k])
			return;
		[seen addObject:k];
	}
	m2v_log("MISSING %s -%s", cls, sel);
}

/* Private probes CoreImage and QuartzCore send (guacamole SweetMetal2.mm): unanswered, CoreImage's dispatch_once never
 * finished. */
@interface NBDeviceMethods : NSObject
@end
@implementation NBDeviceMethods (M2V)
- (NSUInteger)sharedMemorySize
{
	return 1ull << 30;
}
- (NSUInteger)dedicatedMemorySize
{
	return 0;
}
- (uint64_t)recommendedMaxWorkingSetSize
{
	return 768ull << 20;
}
- (NSUInteger)currentAllocatedSize
{
	return 0;
}
- (id)newPipelineLibraryWithFilePath:(NSString *)path error:(NSError **)error
{
	if (error)
		*error = [NSError errorWithDomain:@"M2V" code:1 userInfo:@{NSLocalizedDescriptionKey : @"no pipeline libraries"}];
	return nil;
}
@end

extern "C" __attribute__((visibility("default"))) id m2v_create_device(void)
{
	static id device;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		Dl_info self = {0};
		dladdr((void *)m2v_create_device, &self);
		char cache[256];
		snprintf(cache, sizeof(cache), "/private/var/tmp/m2v-cache/%s", getprogname());
		mkdir("/private/var/tmp/m2v-cache", 0777);
		chmod("/private/var/tmp/m2v-cache", 0777);
		mkdir(cache, 0777);
		if (access(cache, W_OK) != 0) {   /* sandboxed: its own caches directory */
			char dir[200];
			if (confstr(_CS_DARWIN_USER_CACHE_DIR, dir, sizeof(dir)))
				snprintf(cache, sizeof(cache), "%sm2v", dir), mkdir(cache, 0700);
		}
		/* plain rows everywhere: no Apple framebuffer compression, no 10-bit deep framebuffer (guacamole) */
		setenv("CA_DISABLE_FRAMEBUFFER_COMPRESSION", "1", 1);
		setenv("METAL2VULKAN_LOG_PIPELINES", "1", 0);   /* the runtime counts its VkPipeline makings (fps meter, telemetry) */
		CFPreferencesSetAppValue(CFSTR("CAEnableFramebufferCompression"), kCFBooleanFalse, kCFPreferencesCurrentApplication);
		CFPreferencesSetAppValue(CFSTR("CAEnableIOSurfaceCompression"), kCFBooleanFalse, kCFPreferencesCurrentApplication);
		CFPreferencesSetAppValue(CFSTR("CAEnableDeepFramebuffer"), kCFBooleanFalse, kCFPreferencesCurrentApplication);
		/* debugging: KEY=VALUE lines into this process's environment (TU_DEBUG, MESA_LOG_FILE, METAL2VULKAN_*) */
		if (FILE *ef = fopen("/private/var/tmp/m2v-env", "r")) {
			char line[256];
			while (fgets(line, sizeof(line), ef)) {
				line[strcspn(line, "\r\n")] = 0;
				char *eq = strchr(line, '=');
				if (line[0] != '#' && eq) {
					*eq = 0;
					setenv(line, eq + 1, 1);
					m2v_log("env %s=%s", line, eq + 1);
				}
			}
			fclose(ef);
		}
		CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
		@try {
			device = native_bridge_metal_open(self.dli_fname, cache, 0, missing);
		} @catch (NSException *e) {
			m2v_log("device: exception %s", e.reason.UTF8String);
			device = nil;
		}
		m2v_log("device %s in %.0f ms (loader %s, cache %s)", device ? [[device name] UTF8String] : "refused",
			(CFAbsoluteTimeGetCurrent() - t0) * 1000, self.dli_fname, cache);
		if (device && strcmp(getprogname(), "backboardd") != 0)
			class_replaceMethod(object_getClass(device), sel_registerName("supportsExtendedXR10Formats"),
					    imp_implementationWithBlock(^BOOL(id me) {
						    (void)me;
						    return YES;
					    }),
					    "B16@0:8");
	});
	return device;
}
