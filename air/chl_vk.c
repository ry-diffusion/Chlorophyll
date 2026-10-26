/*
 * SPDX-License-Identifier: MIT
 *
 * Chlorophyll's GPU runtime on Turnip (chl.h). Linked into the same dylib as Turnip: the Vulkan entry points are the
 * driver's own, reached through vk_icdGetInstanceProcAddr with no loader in between.
 */

#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan_core.h"

#include "chl.h"
#include "air_nir_vk.h"
#include "nir_air.h"

#include "nir.h"
#include "nir_builder.h"
#include "compiler/glsl_types.h"
#include "ir3/ir3_compiler.h"
#include "common/freedreno_dev_info.h"
#include "util/half_float.h"
#include "util/ralloc.h"

/* Mesa-internal: a pipeline stage given as NIR instead of SPIR-V (vk_pipeline.c, VkPipelineShaderStageNirCreateInfoMESA) */
#ifndef VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA
#define VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA ((VkStructureType)1000290001)
#endif
typedef struct VkPipelineShaderStageNirCreateInfoMESA {
   VkStructureType sType;
   const void *pNext;
   struct nir_shader *nir;
} VkPipelineShaderStageNirCreateInfoMESA;

extern PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name);

#define CHL_FENCE_TIMEOUT_NS 5000000000ull
#define CHL_RING_SLAB (256u << 10)

static int chl_trace = -1;
#define TRACE(...) do { if (chl_trace < 0) chl_trace = getenv("CHL_TRACE") != NULL; if (chl_trace) { fprintf(stderr, "chl: " __VA_ARGS__); fputc('\n', stderr); } } while (0)

static void
say(char *why, size_t why_size, const char *fmt, ...)
{
   if (!why || !why_size)
      return;
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(why, why_size, fmt, ap);
   va_end(ap);
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Vulkan entry points
 */
#define CHL_INSTANCE_FNS(X) \
   X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceMemoryProperties) \
   X(vkGetPhysicalDeviceFormatProperties) X(vkGetPhysicalDeviceImageFormatProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr) \
   X(vkDestroyInstance) X(vkGetPhysicalDeviceFeatures2)
#define CHL_DEVICE_FNS(X) \
   X(vkGetDeviceQueue) X(vkDestroyDevice) X(vkDeviceWaitIdle) X(vkQueueSubmit) X(vkCreateFence) X(vkDestroyFence) \
   X(vkWaitForFences) X(vkResetFences) X(vkGetFenceStatus) \
   X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
   X(vkBindBufferMemory) X(vkGetBufferDeviceAddress) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) \
   X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) X(vkDestroySampler) \
   X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
   X(vkCreateComputePipelines) X(vkCreateGraphicsPipelines) X(vkDestroyPipeline) \
   X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkResetCommandPool) X(vkAllocateCommandBuffers) \
   X(vkBeginCommandBuffer) X(vkEndCommandBuffer) \
   X(vkCmdPipelineBarrier) X(vkCmdBindPipeline) X(vkCmdPushDescriptorSetKHR) X(vkCmdPushConstants) X(vkCmdDispatch) \
   X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkCmdCopyImage) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) \
   X(vkCmdBlitImage) X(vkCmdResolveImage) X(vkCmdClearColorImage) \
   X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) X(vkCmdDraw) \
   X(vkCmdDrawIndexed) X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetStencilReference) X(vkCmdSetBlendConstants) \
   X(vkCmdSetDepthBias) X(vkCmdSetCullMode) X(vkCmdSetFrontFace) X(vkCmdSetPrimitiveTopology) \
   X(vkCmdSetDepthTestEnable) X(vkCmdSetDepthWriteEnable) X(vkCmdSetDepthCompareOp) X(vkCmdSetStencilTestEnable) \
   X(vkCmdSetStencilOp) X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask) X(vkCmdSetDepthBiasEnable) \
   X(vkCmdSetPrimitiveRestartEnable)

struct chl_vk {
#define X(n) PFN_##n n;
   PFN_vkCreateInstance vkCreateInstance;
   CHL_INSTANCE_FNS(X)
   CHL_DEVICE_FNS(X)
#undef X
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Pixel formats: MTLPixelFormat -> VkFormat, with what a copy needs to know (block size and dimensions) and the aspects.
 * The packed formats are matched by memory layout: Metal names a packed format's channels from the least significant bits
 * (RGB10A2Unorm has red in bits 0-9), Vulkan's _PACK formats from the most significant, so the letters come out reversed.
 */
enum { ASPECT_COLOR = 1, ASPECT_DEPTH = 2, ASPECT_STENCIL = 4 };
struct chl_format {
   unsigned mtl;
   const char *name;
   VkFormat vk;
   uint8_t bytes, bw, bh, aspects;
   VkFormat srgb_pair;   /* the sRGB/linear twin a view may take without PixelFormatView usage, or 0 */
   bool alpha_only;      /* A8Unorm: R8 with the red channel read as alpha */
};
#define C(m, n, v, b) { m, n, v, b, 1, 1, ASPECT_COLOR, 0, false }
#define CP(m, n, v, b, p) { m, n, v, b, 1, 1, ASPECT_COLOR, p, false }
#define ASTC(i, w, h) \
   { 204 + i, "MTLPixelFormatASTC_" #w "x" #h "_LDR", (VkFormat)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * i), 16, w, h, ASPECT_COLOR, \
     (VkFormat)(VK_FORMAT_ASTC_4x4_SRGB_BLOCK + 2 * i), false }, \
   { 186 + i, "MTLPixelFormatASTC_" #w "x" #h "_sRGB", (VkFormat)(VK_FORMAT_ASTC_4x4_SRGB_BLOCK + 2 * i), 16, w, h, ASPECT_COLOR, \
     (VkFormat)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * i), false }
static const struct chl_format chl_formats[] = {
   { 1, "MTLPixelFormatA8Unorm", VK_FORMAT_R8_UNORM, 1, 1, 1, ASPECT_COLOR, 0, true },
   CP(10, "MTLPixelFormatR8Unorm", VK_FORMAT_R8_UNORM, 1, VK_FORMAT_R8_SRGB),
   CP(11, "MTLPixelFormatR8Unorm_sRGB", VK_FORMAT_R8_SRGB, 1, VK_FORMAT_R8_UNORM),
   C(12, "MTLPixelFormatR8Snorm", VK_FORMAT_R8_SNORM, 1),
   C(13, "MTLPixelFormatR8Uint", VK_FORMAT_R8_UINT, 1),
   C(14, "MTLPixelFormatR8Sint", VK_FORMAT_R8_SINT, 1),
   C(20, "MTLPixelFormatR16Unorm", VK_FORMAT_R16_UNORM, 2),
   C(22, "MTLPixelFormatR16Snorm", VK_FORMAT_R16_SNORM, 2),
   C(23, "MTLPixelFormatR16Uint", VK_FORMAT_R16_UINT, 2),
   C(24, "MTLPixelFormatR16Sint", VK_FORMAT_R16_SINT, 2),
   C(25, "MTLPixelFormatR16Float", VK_FORMAT_R16_SFLOAT, 2),
   CP(30, "MTLPixelFormatRG8Unorm", VK_FORMAT_R8G8_UNORM, 2, VK_FORMAT_R8G8_SRGB),
   CP(31, "MTLPixelFormatRG8Unorm_sRGB", VK_FORMAT_R8G8_SRGB, 2, VK_FORMAT_R8G8_UNORM),
   C(32, "MTLPixelFormatRG8Snorm", VK_FORMAT_R8G8_SNORM, 2),
   C(33, "MTLPixelFormatRG8Uint", VK_FORMAT_R8G8_UINT, 2),
   C(34, "MTLPixelFormatRG8Sint", VK_FORMAT_R8G8_SINT, 2),
   C(40, "MTLPixelFormatB5G6R5Unorm", VK_FORMAT_B5G6R5_UNORM_PACK16, 2),
   C(41, "MTLPixelFormatA1BGR5Unorm", VK_FORMAT_B5G5R5A1_UNORM_PACK16, 2),
   C(42, "MTLPixelFormatABGR4Unorm", VK_FORMAT_B4G4R4A4_UNORM_PACK16, 2),
   C(43, "MTLPixelFormatBGR5A1Unorm", VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR, 2),
   C(53, "MTLPixelFormatR32Uint", VK_FORMAT_R32_UINT, 4),
   C(54, "MTLPixelFormatR32Sint", VK_FORMAT_R32_SINT, 4),
   C(55, "MTLPixelFormatR32Float", VK_FORMAT_R32_SFLOAT, 4),
   C(60, "MTLPixelFormatRG16Unorm", VK_FORMAT_R16G16_UNORM, 4),
   C(62, "MTLPixelFormatRG16Snorm", VK_FORMAT_R16G16_SNORM, 4),
   C(63, "MTLPixelFormatRG16Uint", VK_FORMAT_R16G16_UINT, 4),
   C(64, "MTLPixelFormatRG16Sint", VK_FORMAT_R16G16_SINT, 4),
   C(65, "MTLPixelFormatRG16Float", VK_FORMAT_R16G16_SFLOAT, 4),
   CP(70, "MTLPixelFormatRGBA8Unorm", VK_FORMAT_R8G8B8A8_UNORM, 4, VK_FORMAT_R8G8B8A8_SRGB),
   CP(71, "MTLPixelFormatRGBA8Unorm_sRGB", VK_FORMAT_R8G8B8A8_SRGB, 4, VK_FORMAT_R8G8B8A8_UNORM),
   C(72, "MTLPixelFormatRGBA8Snorm", VK_FORMAT_R8G8B8A8_SNORM, 4),
   C(73, "MTLPixelFormatRGBA8Uint", VK_FORMAT_R8G8B8A8_UINT, 4),
   C(74, "MTLPixelFormatRGBA8Sint", VK_FORMAT_R8G8B8A8_SINT, 4),
   CP(80, "MTLPixelFormatBGRA8Unorm", VK_FORMAT_B8G8R8A8_UNORM, 4, VK_FORMAT_B8G8R8A8_SRGB),
   CP(81, "MTLPixelFormatBGRA8Unorm_sRGB", VK_FORMAT_B8G8R8A8_SRGB, 4, VK_FORMAT_B8G8R8A8_UNORM),
   C(90, "MTLPixelFormatRGB10A2Unorm", VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4),
   C(91, "MTLPixelFormatRGB10A2Uint", VK_FORMAT_A2B10G10R10_UINT_PACK32, 4),
   C(92, "MTLPixelFormatRG11B10Float", VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4),
   C(93, "MTLPixelFormatRGB9E5Float", VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 4),
   C(94, "MTLPixelFormatBGR10A2Unorm", VK_FORMAT_A2R10G10B10_UNORM_PACK32, 4),
   C(103, "MTLPixelFormatRG32Uint", VK_FORMAT_R32G32_UINT, 8),
   C(104, "MTLPixelFormatRG32Sint", VK_FORMAT_R32G32_SINT, 8),
   C(105, "MTLPixelFormatRG32Float", VK_FORMAT_R32G32_SFLOAT, 8),
   C(110, "MTLPixelFormatRGBA16Unorm", VK_FORMAT_R16G16B16A16_UNORM, 8),
   C(112, "MTLPixelFormatRGBA16Snorm", VK_FORMAT_R16G16B16A16_SNORM, 8),
   C(113, "MTLPixelFormatRGBA16Uint", VK_FORMAT_R16G16B16A16_UINT, 8),
   C(114, "MTLPixelFormatRGBA16Sint", VK_FORMAT_R16G16B16A16_SINT, 8),
   C(115, "MTLPixelFormatRGBA16Float", VK_FORMAT_R16G16B16A16_SFLOAT, 8),
   C(123, "MTLPixelFormatRGBA32Uint", VK_FORMAT_R32G32B32A32_UINT, 16),
   C(124, "MTLPixelFormatRGBA32Sint", VK_FORMAT_R32G32B32A32_SINT, 16),
   C(125, "MTLPixelFormatRGBA32Float", VK_FORMAT_R32G32B32A32_SFLOAT, 16),
   ASTC(0, 4, 4), ASTC(1, 5, 4), ASTC(2, 5, 5), ASTC(3, 6, 5), ASTC(4, 6, 6), ASTC(5, 8, 5), ASTC(6, 8, 6),
   ASTC(7, 8, 8), ASTC(8, 10, 5), ASTC(9, 10, 6), ASTC(10, 10, 8), ASTC(11, 10, 10), ASTC(12, 12, 10), ASTC(13, 12, 12),
   { 250, "MTLPixelFormatDepth16Unorm", VK_FORMAT_D16_UNORM, 2, 1, 1, ASPECT_DEPTH, 0, false },
   { 252, "MTLPixelFormatDepth32Float", VK_FORMAT_D32_SFLOAT, 4, 1, 1, ASPECT_DEPTH, 0, false },
   { 253, "MTLPixelFormatStencil8", VK_FORMAT_S8_UINT, 1, 1, 1, ASPECT_STENCIL, 0, false },
   { 260, "MTLPixelFormatDepth32Float_Stencil8", VK_FORMAT_D32_SFLOAT_S8_UINT, 8, 1, 1, ASPECT_DEPTH | ASPECT_STENCIL, 0, false },
   /* Apple's extended-range formats, by a native format of the same size (the XR encoding itself is not reproduced):
    * QuartzCore builds its deep-colour pipelines in these whatever the device answers */
   C(550, "MTLPixelFormatRGB10A8_2P_XR10 (as BGRA8)", VK_FORMAT_B8G8R8A8_UNORM, 4),
   C(552, "MTLPixelFormatBGRA10_XR (as RGBA16Float)", VK_FORMAT_R16G16B16A16_SFLOAT, 8),
   C(553, "MTLPixelFormatBGRA10_XR_sRGB (as RGBA16Float)", VK_FORMAT_R16G16B16A16_SFLOAT, 8),
   C(554, "MTLPixelFormatBGR10_XR (as BGRA8)", VK_FORMAT_B8G8R8A8_UNORM, 4),
   C(555, "MTLPixelFormatBGR10_XR_sRGB (as BGRA8 sRGB)", VK_FORMAT_B8G8R8A8_SRGB, 4),
};
#undef C
#undef CP
#undef ASTC

static const struct chl_format *
chl_format(unsigned mtl)
{
   for (unsigned i = 0; i < sizeof(chl_formats) / sizeof(chl_formats[0]); i++)
      if (chl_formats[i].mtl == mtl)
         return &chl_formats[i];
   return NULL;
}

static VkImageAspectFlags
vk_aspects(const struct chl_format *f)
{
   return (f->aspects & ASPECT_COLOR ? VK_IMAGE_ASPECT_COLOR_BIT : 0) | (f->aspects & ASPECT_DEPTH ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
          (f->aspects & ASPECT_STENCIL ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Objects
 */
enum chl_kind { K_BUFFER, K_TEXTURE, K_SAMPLER, K_COMPUTE, K_RENDER, K_DEPTH_STENCIL };

struct chl_obj {
   _Atomic int refs;
   enum chl_kind kind;
   struct chl_device *dev;
};

struct chl_buffer {
   struct chl_obj o;
   VkBuffer vk;
   VkDeviceMemory mem;
   void *map;
   size_t size;
   uint64_t addr;
   bool ring;            /* an upload-ring slab, owned by the device's slab list */
};

struct chl_view_cache { unsigned level, slice; VkImageView view; };

struct chl_texture {
   struct chl_obj o;
   struct chl_texture *parent;    /* a view: the image and memory are the parent's */
   VkImage image;
   VkDeviceMemory mem;
   const struct chl_format *fmt;
   VkFormat vkfmt;                /* the view format */
   unsigned type, width, height, depth, levels, layers, samples;  /* layers: Vulkan array layers (6 per cube) */
   unsigned base_level, base_layer;   /* into the image, for a view */
   unsigned usage;
   VkImageUsageFlags vkusage;
   VkImageView view;              /* the sampled view: all of this texture's levels and layers */
   VkImageView storage[16];       /* single-level views for image stores, by level */
   struct chl_view_cache att[8];  /* single-level single-layer attachment views */
   unsigned n_att;
   unsigned char swizzle[4];
   uint32_t surface_handle;       /* an IOSurface's BO (chl_texture_from_iosurface), for the cache clean */
};

struct chl_sampler {
   struct chl_obj o;
   VkSampler vk, vk_compare;
};

/* One stage's AIR translation and what its resources are. */
struct chl_stage_info {
   nir_shader *nir;               /* translated and bridged (air_nir_vk.c); cloned for each pipeline variant */
   struct air_shader_info info;
   uint32_t buffers, textures, samplers, inline_samplers;
};

struct chl_layout {
   VkDescriptorSetLayout set;
   VkPipelineLayout layout;
   struct air_vk_layout bindings;
   VkSampler immutable[AIR_VK_MAX_BINDINGS];
   /* a level slot (air_resource.level_slots): the texture slot + 1 it is a level of, and which level */
   uint8_t level_src[AIR_VK_MAX_BINDINGS], level[AIR_VK_MAX_BINDINGS];
   unsigned push_bytes;
};

#define CHL_COMPUTE_VARIANTS 8
struct chl_compute_pipeline {
   struct chl_obj o;
   void *mem_ctx;
   struct chl_stage_info cs;
   struct chl_layout layout;
   bool dynamic_threadgroup;
   struct { unsigned threads[3]; unsigned shared; VkPipeline vk; } variant[CHL_COMPUTE_VARIANTS];
   unsigned n_variants;
   pthread_mutex_t lock;
};

struct chl_render_pipeline {
   struct chl_obj o;
   void *mem_ctx;
   struct chl_stage_info vs, fs;
   struct chl_layout layout;
   VkPipeline vk;
   unsigned color_count;
   VkFormat color_format[8];
   VkFormat depth_format, stencil_format;
   unsigned samples;
   VkVertexInputBindingDescription vbind[CHL_MAX_BUFFERS];
   unsigned n_vbind;
   uint32_t vertex_stream_mask;   /* [[buffer(n)]] that are vertex streams */
};

struct chl_depth_stencil {
   struct chl_obj o;
   struct chl_depth_stencil_desc d;
};

/* Upload ring slab, recycled by the device once the command buffer that used it has completed. */
struct chl_slab {
   struct chl_buffer *buf;
   struct chl_slab *next;
};

struct chl_stage_bind {
   struct chl_buffer *buf[AIR_VK_BUFFER_SLOTS];
   size_t off[AIR_VK_BUFFER_SLOTS];
   struct chl_texture *tex[AIR_VK_TEXTURE_SLOTS];
   struct chl_sampler *smp[AIR_VK_SAMPLER_SLOTS];
};

enum chl_encoder { ENC_NONE, ENC_RENDER, ENC_COMPUTE };

struct chl_cmdbuf {
   struct chl_device *dev;
   VkCommandPool pool;
   VkCommandBuffer cb;
   VkFence fence;
   bool committed, done, failed;
   chl_done_fn done_fn;
   void *done_ctx;
   /* retained objects, released at completion */
   struct chl_obj **held;
   unsigned n_held, cap_held;
   /* upload ring */
   struct chl_slab *slabs;
   size_t slab_used;
   /* encoder state */
   enum chl_encoder enc;
   struct chl_stage_bind bind[2];
   uint32_t tg_len[AIR_VK_BUFFER_SLOTS];
   struct chl_compute_pipeline *cp;
   struct chl_render_pipeline *rp;
   struct chl_depth_stencil *ds;
   VkPrimitiveTopology topology;
   bool topology_set;
   unsigned pass_width, pass_height;
   VkImageView pass_color[8];   /* the render pass's colour attachments, for the framebuffer-fetch input attachments */
   /* what flush_bindings last pushed, per bind point (0 graphics, 1 compute), valid while pushed_layout is the bound layout:
    * a draw pushes only the descriptors and push constants that changed */
   VkPipelineLayout pushed_layout[2];
   struct { VkDescriptorBufferInfo b; VkDescriptorImageInfo i; } pushed[2][AIR_VK_MAX_BINDINGS];
   uint32_t pushed_pc[2][64];
   bool vp_set, sc_set;
   struct chl_cmdbuf *next_free, *next_inflight;
};

struct chl_shared_layout {
   struct chl_shared_layout *next;
   size_t key_size;
   void *key;
   VkDescriptorSetLayout set;
   VkPipelineLayout layout;
};

struct chl_shared_sampler {
   struct chl_shared_sampler *next;
   VkSamplerCreateInfo ci;
   VkSampler sampler;
};

struct chl_device {
   struct chl_vk vk;
   VkInstance instance;
   VkPhysicalDevice pd;
   VkDevice dev;
   VkQueue queue;
   pthread_mutex_t queue_lock;
   VkPhysicalDeviceProperties props;
   VkPhysicalDeviceMemoryProperties memprops;
   uint32_t mt_host;                 /* host-visible, coherent */
   char name[256];
   struct ir3_compiler *ir3;
   const nir_shader_compiler_options *nir_options;
   unsigned align;                   /* descriptor offset alignment (64 on Turnip) */
   VkSampler default_sampler;
   /* bound in every texture slot nothing is bound to: Turnip's null descriptor would send the fetch the shader still makes
    * (the null mask only discards its result) to GPU address 0 + offset, an SMMU fault on this kernel */
   struct chl_texture *null_tex;

   /* work that has to reach the GPU before anything committed after it: image layout transitions, uploads */
   pthread_mutex_t pro_lock;
   VkCommandPool pro_pool;
   VkCommandBuffer pro_cb;
   bool pro_open;
   struct chl_obj **pro_held;
   unsigned n_pro_held, cap_pro_held;
   struct { VkFence fence; VkCommandBuffer cb; struct chl_obj **held; unsigned n_held; } pro_inflight[8];

   /* layouts and constexpr samplers shared by every pipeline with the same signature, kept until the device closes:
    * pipelines that share a layout keep each other's push descriptors, so a switch between them pushes only what changed */
   pthread_mutex_t share_lock;
   struct chl_shared_layout *layouts;
   struct chl_shared_sampler *samplers;

   /* free lists */
   pthread_mutex_t free_lock;
   struct chl_cmdbuf *free_cbs;
   struct chl_slab *free_slabs;

   /* completion thread */
   pthread_t thread;
   pthread_mutex_t done_lock;
   pthread_cond_t done_cv;
   struct chl_cmdbuf *inflight_head, *inflight_tail;
   bool quit;
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Reference counting
 */
static void destroy_obj(struct chl_obj *o);

static void
obj_init(struct chl_obj *o, struct chl_device *dev, enum chl_kind kind)
{
   atomic_init(&o->refs, 1);
   o->kind = kind;
   o->dev = dev;
}

static void
obj_retain(struct chl_obj *o)
{
   if (o)
      atomic_fetch_add_explicit(&o->refs, 1, memory_order_relaxed);
}

static void
obj_release(struct chl_obj *o)
{
   if (o && atomic_fetch_sub_explicit(&o->refs, 1, memory_order_acq_rel) == 1)
      destroy_obj(o);
}

static void
hold(struct chl_obj ***list, unsigned *n, unsigned *cap, struct chl_obj *o)
{
   if (!o)
      return;
   if (*n == *cap) {
      unsigned c = *cap ? *cap * 2 : 64;
      struct chl_obj **l = realloc(*list, c * sizeof(*l));
      if (!l)
         return;   /* out of memory: the object is simply not held (and not retained) */
      *list = l;
      *cap = c;
   }
   obj_retain(o);
   (*list)[(*n)++] = o;
}

static void
cb_hold(struct chl_cmdbuf *cb, void *o)
{
   if (!o)
      return;
   /* the most recent one again (a draw re-binding what the last one bound) is held once */
   if (cb->n_held && cb->held[cb->n_held - 1] == o)
      return;
   hold(&cb->held, &cb->n_held, &cb->cap_held, o);
}

void chl_buffer_retain(struct chl_buffer *b) { obj_retain(b ? &b->o : NULL); }
void chl_buffer_release(struct chl_buffer *b) { obj_release(b ? &b->o : NULL); }
void chl_texture_retain(struct chl_texture *t) { obj_retain(t ? &t->o : NULL); }
void chl_texture_release(struct chl_texture *t) { obj_release(t ? &t->o : NULL); }
void chl_sampler_retain(struct chl_sampler *s) { obj_retain(s ? &s->o : NULL); }
void chl_sampler_release(struct chl_sampler *s) { obj_release(s ? &s->o : NULL); }
void chl_compute_pipeline_release(struct chl_compute_pipeline *p) { obj_release(p ? &p->o : NULL); }
void chl_render_pipeline_release(struct chl_render_pipeline *p) { obj_release(p ? &p->o : NULL); }
void chl_depth_stencil_release(struct chl_depth_stencil *d) { obj_release(d ? &d->o : NULL); }

/* ---------------------------------------------------------------------------------------------------------------------
 * The prologue: layout transitions and uploads that every later submission must find done
 */
static VkCommandBuffer
pro_begin(struct chl_device *d)
{
   /* called with pro_lock held */
   if (!d->pro_open) {
      VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, d->pro_pool,
                                         VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
      if (d->vk.vkAllocateCommandBuffers(d->dev, &ai, &d->pro_cb) != VK_SUCCESS)
         return VK_NULL_HANDLE;
      VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
      d->vk.vkBeginCommandBuffer(d->pro_cb, &bi);
      d->pro_open = true;
   }
   return d->pro_cb;
}

static void
full_barrier(struct chl_device *d, VkCommandBuffer cb)
{
   VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT,
                          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
   d->vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void
pro_reap(struct chl_device *d, bool wait)
{
   /* pro_lock held */
   for (unsigned i = 0; i < 8; i++) {
      if (!d->pro_inflight[i].fence)
         continue;
      VkResult r = wait ? d->vk.vkWaitForFences(d->dev, 1, &d->pro_inflight[i].fence, VK_TRUE, CHL_FENCE_TIMEOUT_NS)
                        : d->vk.vkGetFenceStatus(d->dev, d->pro_inflight[i].fence);
      if (r != VK_SUCCESS)
         continue;
      d->vk.vkDestroyFence(d->dev, d->pro_inflight[i].fence, NULL);
      d->pro_inflight[i].fence = VK_NULL_HANDLE;
      for (unsigned k = 0; k < d->pro_inflight[i].n_held; k++)
         obj_release(d->pro_inflight[i].held[k]);
      free(d->pro_inflight[i].held);
      d->pro_inflight[i].held = NULL;
      d->pro_inflight[i].n_held = 0;
   }
}

/* Submit the prologue if it holds anything. queue_lock and pro_lock are held. */
static VkResult
pro_submit_locked(struct chl_device *d)
{
   if (!d->pro_open)
      return VK_SUCCESS;
   full_barrier(d, d->pro_cb);
   d->vk.vkEndCommandBuffer(d->pro_cb);
   d->pro_open = false;
   pro_reap(d, false);
   unsigned slot = 8;
   for (unsigned i = 0; i < 8 && slot == 8; i++)
      if (!d->pro_inflight[i].fence)
         slot = i;
   if (slot == 8) {
      pro_reap(d, true);
      for (unsigned i = 0; i < 8 && slot == 8; i++)
         if (!d->pro_inflight[i].fence)
            slot = i;
      if (slot == 8)
         return VK_ERROR_DEVICE_LOST;
   }
   VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
   VkFence fence;
   if (d->vk.vkCreateFence(d->dev, &fi, NULL, &fence) != VK_SUCCESS)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &d->pro_cb, 0, NULL };
   VkResult r = d->vk.vkQueueSubmit(d->queue, 1, &si, fence);
   d->pro_inflight[slot].fence = fence;
   d->pro_inflight[slot].cb = d->pro_cb;
   d->pro_inflight[slot].held = d->pro_held;
   d->pro_inflight[slot].n_held = d->n_pro_held;
   d->pro_held = NULL;
   d->n_pro_held = d->cap_pro_held = 0;
   d->pro_cb = VK_NULL_HANDLE;
   return r;
}

/* Run `record` on a one-time command buffer after everything submitted so far, and wait for it. */
typedef void (*record_fn)(struct chl_device *d, VkCommandBuffer cb, void *ctx);
static int
run_now(struct chl_device *d, record_fn record, void *ctx, char *why, size_t why_size)
{
   pthread_mutex_lock(&d->queue_lock);
   pthread_mutex_lock(&d->pro_lock);
   VkCommandBuffer cb = pro_begin(d);
   if (!cb) {
      pthread_mutex_unlock(&d->pro_lock);
      pthread_mutex_unlock(&d->queue_lock);
      say(why, why_size, "no command buffer for an immediate copy");
      return -1;
   }
   full_barrier(d, cb);
   record(d, cb, ctx);
   VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_HOST_READ_BIT };
   d->vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
   VkResult r = pro_submit_locked(d);
   if (r == VK_SUCCESS)
      pro_reap(d, true);
   pthread_mutex_unlock(&d->pro_lock);
   pthread_mutex_unlock(&d->queue_lock);
   if (r != VK_SUCCESS) {
      say(why, why_size, "the GPU did not take or finish an immediate copy (VkResult %d)", r);
      return -1;
   }
   return 0;
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Completion thread
 */
static void cb_retire(struct chl_cmdbuf *cb);

static void *
completion_thread(void *arg)
{
   struct chl_device *d = arg;
   pthread_setname_np("chl.completion");
   pthread_mutex_lock(&d->done_lock);
   for (;;) {
      while (!d->inflight_head && !d->quit)
         pthread_cond_wait(&d->done_cv, &d->done_lock);
      if (!d->inflight_head && d->quit)
         break;
      struct chl_cmdbuf *cb = d->inflight_head;
      pthread_mutex_unlock(&d->done_lock);
      VkResult r = d->vk.vkWaitForFences(d->dev, 1, &cb->fence, VK_TRUE, CHL_FENCE_TIMEOUT_NS);
      pthread_mutex_lock(&d->done_lock);
      if (r == VK_TIMEOUT && !d->quit) {
         /* a long frame: keep waiting, but say nothing yet; chl_cmdbuf_wait reports timeouts to its caller */
         continue;
      }
      d->inflight_head = cb->next_inflight;
      if (!d->inflight_head)
         d->inflight_tail = NULL;
      pthread_mutex_unlock(&d->done_lock);
      /* what it held goes first: once it is reported done, its owner may recycle it (chl_cmdbuf_release) */
      cb_retire(cb);
      chl_done_fn fn = cb->done_fn;
      void *ctx = cb->done_ctx;
      bool failed = r != VK_SUCCESS;
      pthread_mutex_lock(&d->done_lock);
      cb->failed = failed;
      cb->done = true;
      pthread_cond_broadcast(&d->done_cv);
      pthread_mutex_unlock(&d->done_lock);
      if (fn)
         fn(ctx, failed ? -1 : 0, failed ? "the GPU did not finish the command buffer" : NULL);
      pthread_mutex_lock(&d->done_lock);
   }
   pthread_mutex_unlock(&d->done_lock);
   return NULL;
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Device
 */
struct chl_device *
chl_device_open(char *why, size_t why_size)
{
   struct chl_device *d = calloc(1, sizeof(*d));
   if (!d) {
      say(why, why_size, "out of memory making a device");
      return NULL;
   }
   pthread_mutex_init(&d->queue_lock, NULL);
   pthread_mutex_init(&d->pro_lock, NULL);
   pthread_mutex_init(&d->free_lock, NULL);
   pthread_mutex_init(&d->share_lock, NULL);
   pthread_mutex_init(&d->done_lock, NULL);
   pthread_cond_init(&d->done_cv, NULL);
   glsl_type_singleton_init_or_ref();

   /* /private/var/tmp/chl.tudebug: Turnip's TU_DEBUG for every chl process (system daemons get no environment of ours),
    * e.g. "sysmem" to render without GMEM tiles. Read once, before Turnip reads its debug flags. */
   if (!getenv("TU_DEBUG")) {
      FILE *f = fopen("/private/var/tmp/chl.tudebug", "r");
      if (f) {
         char flags[128] = "";
         if (fgets(flags, sizeof(flags), f)) {
            flags[strcspn(flags, "\r\n")] = 0;
            if (flags[0])
               setenv("TU_DEBUG", flags, 1);
         }
         fclose(f);
      }
   }

   d->vk.vkCreateInstance = (PFN_vkCreateInstance)vk_icdGetInstanceProcAddr(NULL, "vkCreateInstance");
   VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "Chlorophyll", 1, "chl", 1, VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL };
   if (!d->vk.vkCreateInstance || d->vk.vkCreateInstance(&ici, NULL, &d->instance) != VK_SUCCESS) {
      say(why, why_size, "Turnip made no Vulkan instance");
      goto fail;
   }
#define X(n) d->vk.n = (PFN_##n)vk_icdGetInstanceProcAddr(d->instance, #n);
   CHL_INSTANCE_FNS(X)
#undef X
   uint32_t n = 1;
   if (d->vk.vkEnumeratePhysicalDevices(d->instance, &n, &d->pd) < 0 || n == 0) {
      say(why, why_size, "Turnip found no GPU: the kernel's SweetAdreno630 user client is missing or would not open");
      goto fail;
   }
   d->vk.vkGetPhysicalDeviceProperties(d->pd, &d->props);
   d->vk.vkGetPhysicalDeviceMemoryProperties(d->pd, &d->memprops);
   snprintf(d->name, sizeof(d->name), "%s", d->props.deviceName);
   d->mt_host = ~0u;
   for (uint32_t i = 0; i < d->memprops.memoryTypeCount && d->mt_host == ~0u; i++) {
      VkMemoryPropertyFlags f = d->memprops.memoryTypes[i].propertyFlags;
      if ((f & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
          (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
         d->mt_host = i;
   }
   if (d->mt_host == ~0u) {
      say(why, why_size, "%s has no host-visible coherent memory", d->name);
      goto fail;
   }
   d->align = (unsigned)d->props.limits.minStorageBufferOffsetAlignment;
   if (d->props.limits.minUniformBufferOffsetAlignment > d->align)
      d->align = (unsigned)d->props.limits.minUniformBufferOffsetAlignment;
   if (d->align < 4)
      d->align = 4;

   /* Metal's framebuffer fetch ([[color(n)]] read in a fragment function): input attachments of the dynamic render pass,
    * read in rasterization order with no barrier between draws (air_nir_vk.c, bridge_fb_fetch) */
   VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT rao = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_EXT };
   rao.rasterizationOrderColorAttachmentAccess = VK_TRUE;
   VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR lread = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR, &rao };
   lread.dynamicRenderingLocalRead = VK_TRUE;
   VkPhysicalDeviceRobustness2FeaturesEXT rob2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, &lread };
   rob2.nullDescriptor = VK_TRUE;
   VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &rob2 };
   f13.dynamicRendering = VK_TRUE;
   f13.synchronization2 = VK_TRUE;
   VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13 };
   f12.bufferDeviceAddress = VK_TRUE;
   f12.shaderFloat16 = VK_TRUE;
   f12.shaderInt8 = VK_TRUE;
   f12.uniformAndStorageBuffer8BitAccess = VK_FALSE;
   VkPhysicalDeviceFeatures2 feats = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12 };
   feats.features.independentBlend = VK_TRUE;
   feats.features.samplerAnisotropy = VK_TRUE;
   feats.features.textureCompressionASTC_LDR = VK_TRUE;
   feats.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
   feats.features.shaderStorageImageReadWithoutFormat = VK_TRUE;
   feats.features.fragmentStoresAndAtomics = VK_TRUE;
   feats.features.vertexPipelineStoresAndAtomics = VK_TRUE;
   feats.features.sampleRateShading = VK_TRUE;
   feats.features.depthBiasClamp = VK_TRUE;
   feats.features.imageCubeArray = VK_TRUE;
   static const char *const exts[] = { VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
                                       VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
                                       VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME, VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME,
                                       VK_EXT_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME };
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio };
   VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &feats, 0, 1, &qci, 0, NULL, sizeof(exts) / sizeof(exts[0]), exts,
                             NULL };
   VkResult r = d->vk.vkCreateDevice(d->pd, &dci, NULL, &d->dev);
   if (r != VK_SUCCESS) {
      say(why, why_size, "%s would not make a Vulkan device with push descriptors, null descriptors, dynamic rendering and local reads "
                         "(VkResult %d)", d->name, r);
      goto fail;
   }
#define X(n) d->vk.n = (PFN_##n)d->vk.vkGetDeviceProcAddr(d->dev, #n);
   CHL_DEVICE_FNS(X)
#undef X
   d->vk.vkGetDeviceQueue(d->dev, 0, 0, &d->queue);

   VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
                                   VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0 };
   if (d->vk.vkCreateCommandPool(d->dev, &pci, NULL, &d->pro_pool) != VK_SUCCESS) {
      say(why, why_size, "no command pool");
      goto fail;
   }

   VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
   sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
   sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   sci.maxLod = VK_LOD_CLAMP_NONE;
   d->vk.vkCreateSampler(d->dev, &sci, NULL, &d->default_sampler);

   /* The NIR options AIR is translated for: ir3's for this GPU (Turnip compiles with the same ir3) */
   struct fd_dev_id id = { .gpu_id = 630, .chip_id = 0x06030001 };
   const struct fd_dev_info *info = fd_dev_info_raw(&id);
   struct ir3_compiler_options opts = { 0 };
   d->ir3 = info ? ir3_compiler_create(NULL, &id, info, &opts) : NULL;
   if (!d->ir3) {
      say(why, why_size, "no ir3 compiler for the Adreno 630");
      goto fail;
   }
   d->nir_options = ir3_get_compiler_options(d->ir3);

   if (pthread_create(&d->thread, NULL, completion_thread, d) != 0) {
      say(why, why_size, "no completion thread");
      goto fail;
   }
   {
      static const struct chl_texture_desc nd = { .pixel_format = 80 /* BGRA8Unorm */, .type = CHL_TEX_2D, .width = 1, .height = 1,
                                                  .depth = 1, .array_length = 1, .mip_levels = 1, .sample_count = 1,
                                                  .usage = CHL_USAGE_SHADER_READ | CHL_USAGE_SHADER_WRITE };
      static const uint8_t zero[4] = { 0, 0, 0, 0 };
      char nwhy[256];
      d->null_tex = chl_texture_create(d, &nd, nwhy, sizeof(nwhy));
      if (d->null_tex)
         chl_texture_replace(d->null_tex, 0, 0, NULL, NULL, zero, 4, 0, nwhy, sizeof(nwhy));
   }
   return d;

fail:
   if (d->dev)
      d->vk.vkDestroyDevice(d->dev, NULL);
   if (d->instance)
      d->vk.vkDestroyInstance(d->instance, NULL);
   glsl_type_singleton_decref();
   free(d);
   return NULL;
}

void
chl_device_close(struct chl_device *d)
{
   if (!d)
      return;
   pthread_mutex_lock(&d->done_lock);
   d->quit = true;
   pthread_cond_broadcast(&d->done_cv);
   pthread_mutex_unlock(&d->done_lock);
   pthread_join(d->thread, NULL);
   d->vk.vkDeviceWaitIdle(d->dev);
   if (d->null_tex)
      chl_texture_release(d->null_tex);
   pthread_mutex_lock(&d->pro_lock);
   pro_reap(d, true);
   pthread_mutex_unlock(&d->pro_lock);
   while (d->layouts) {
      struct chl_shared_layout *sl = d->layouts;
      d->layouts = sl->next;
      d->vk.vkDestroyPipelineLayout(d->dev, sl->layout, NULL);
      d->vk.vkDestroyDescriptorSetLayout(d->dev, sl->set, NULL);
      free(sl->key);
      free(sl);
   }
   while (d->samplers) {
      struct chl_shared_sampler *ss = d->samplers;
      d->samplers = ss->next;
      d->vk.vkDestroySampler(d->dev, ss->sampler, NULL);
      free(ss);
   }
   d->vk.vkDestroyDevice(d->dev, NULL);
   d->vk.vkDestroyInstance(d->instance, NULL);
   glsl_type_singleton_decref();
   free(d);
}

const char *
chl_device_name(const struct chl_device *d)
{
   return d->name;
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Buffers
 */
static struct chl_buffer *
buffer_new(struct chl_device *d, size_t size, char *why, size_t why_size)
{
   struct chl_buffer *b = calloc(1, sizeof(*b));
   if (!b) {
      say(why, why_size, "out of memory making a buffer");
      return NULL;
   }
   obj_init(&b->o, d, K_BUFFER);
   b->size = size;
   VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
   if (d->vk.vkCreateBuffer(d->dev, &bci, NULL, &b->vk) != VK_SUCCESS) {
      say(why, why_size, "%s would not make a buffer of %zu bytes", d->name, size);
      free(b);
      return NULL;
   }
   VkMemoryRequirements req;
   d->vk.vkGetBufferMemoryRequirements(d->dev, b->vk, &req);
   VkMemoryAllocateFlagsInfo fl = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0 };
   VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fl, req.size, d->mt_host };
   if (d->vk.vkAllocateMemory(d->dev, &ai, NULL, &b->mem) != VK_SUCCESS ||
       d->vk.vkBindBufferMemory(d->dev, b->vk, b->mem, 0) != VK_SUCCESS ||
       d->vk.vkMapMemory(d->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map) != VK_SUCCESS) {
      say(why, why_size, "%s would not allocate or map %zu bytes for a buffer", d->name, size);
      if (b->mem)
         d->vk.vkFreeMemory(d->dev, b->mem, NULL);
      d->vk.vkDestroyBuffer(d->dev, b->vk, NULL);
      free(b);
      return NULL;
   }
   VkBufferDeviceAddressInfo dai = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, b->vk };
   b->addr = d->vk.vkGetBufferDeviceAddress(d->dev, &dai);
   return b;
}

struct chl_buffer *
chl_buffer_create(struct chl_device *d, size_t size, char *why, size_t why_size)
{
   if (size == 0 || size > (1ull << 31)) {
      say(why, why_size, "a buffer of %zu bytes", size);
      return NULL;
   }
   return buffer_new(d, size, why, why_size);
}

void *chl_buffer_contents(struct chl_buffer *b) { return b->map; }
size_t chl_buffer_length(const struct chl_buffer *b) { return b->size; }
uint64_t chl_buffer_gpu_address(const struct chl_buffer *b) { return b->addr; }

/* ---------------------------------------------------------------------------------------------------------------------
 * Textures
 */
static VkImageViewType
view_type(unsigned type)
{
   switch (type) {
   case CHL_TEX_1D: return VK_IMAGE_VIEW_TYPE_1D;
   case CHL_TEX_1D_ARRAY: return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
   case CHL_TEX_2D_ARRAY: case CHL_TEX_2D_MS_ARRAY: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
   case CHL_TEX_CUBE: return VK_IMAGE_VIEW_TYPE_CUBE;
   case CHL_TEX_CUBE_ARRAY: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
   case CHL_TEX_3D: return VK_IMAGE_VIEW_TYPE_3D;
   default: return VK_IMAGE_VIEW_TYPE_2D;
   }
}

static VkComponentSwizzle
swz(unsigned char m, VkComponentSwizzle identity)
{
   static const VkComponentSwizzle map[6] = { VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE, VK_COMPONENT_SWIZZLE_R,
                                              VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
   return m < 6 ? map[m] : identity;
}

static bool
swizzle_identity(const unsigned char s[4])
{
   return !s || (!s[0] && !s[1] && !s[2] && !s[3]) || (s[0] == 2 && s[1] == 3 && s[2] == 4 && s[3] == 5);
}

static VkImageView
make_view(struct chl_texture *t, VkImageViewType vt, unsigned base_level, unsigned levels, unsigned base_layer, unsigned layers,
          bool swizzled)
{
   struct chl_device *d = t->o.dev;
   VkImageViewCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
   ci.image = t->image;
   ci.viewType = vt;
   ci.format = t->vkfmt;
   if (swizzled) {
      if (t->fmt->alpha_only)
         ci.components = (VkComponentMapping){ VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO,
                                               VK_COMPONENT_SWIZZLE_R };
      if (!swizzle_identity(t->swizzle))
         ci.components = (VkComponentMapping){ swz(t->swizzle[0], VK_COMPONENT_SWIZZLE_R), swz(t->swizzle[1], VK_COMPONENT_SWIZZLE_G),
                                               swz(t->swizzle[2], VK_COMPONENT_SWIZZLE_B), swz(t->swizzle[3], VK_COMPONENT_SWIZZLE_A) };
   }
   VkImageAspectFlags asp = vk_aspects(t->fmt);
   if (asp == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) && swizzled)
      asp = VK_IMAGE_ASPECT_DEPTH_BIT;   /* a sampled view of depth+stencil reads depth */
   ci.subresourceRange = (VkImageSubresourceRange){ asp, base_level, levels, base_layer, layers };
   VkImageView v = VK_NULL_HANDLE;
   d->vk.vkCreateImageView(d->dev, &ci, NULL, &v);
   return v;
}

static VkImageView
attachment_view(struct chl_texture *t, unsigned level, unsigned slice)
{
   for (unsigned i = 0; i < t->n_att; i++)
      if (t->att[i].level == level && t->att[i].slice == slice)
         return t->att[i].view;
   VkImageViewType vt = t->samples > 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D;
   VkImageView v = make_view(t, vt, t->base_level + level, 1, t->base_layer + slice, 1, false);
   if (v && t->n_att < 8)
      t->att[t->n_att++] = (struct chl_view_cache){ level, slice, v };
   return v;
}

static VkImageView
storage_view(struct chl_texture *t, unsigned level)
{
   if (level >= 16)
      return VK_NULL_HANDLE;
   if (!t->storage[level]) {
      VkImageViewType vt = view_type(t->type);
      if (vt == VK_IMAGE_VIEW_TYPE_CUBE)
         vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      else if (vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)
         vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      t->storage[level] = make_view(t, vt, t->base_level + level, 1, t->base_layer, t->type == CHL_TEX_3D ? 1 : t->layers, false);
   }
   return t->storage[level];
}

static void
transition_new_image(struct chl_device *d, struct chl_texture *t)
{
   pthread_mutex_lock(&d->pro_lock);
   VkCommandBuffer cb = pro_begin(d);
   if (cb) {
      VkImageMemoryBarrier ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0,
                                  VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, t->image,
                                  { vk_aspects(t->fmt), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS } };
      d->vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
      hold(&d->pro_held, &d->n_pro_held, &d->cap_pro_held, &t->o);
   }
   pthread_mutex_unlock(&d->pro_lock);
}

struct chl_texture *
chl_texture_create(struct chl_device *d, const struct chl_texture_desc *desc, char *why, size_t why_size)
{
   const struct chl_format *f = desc ? chl_format(desc->pixel_format) : NULL;
   if (!f) {
      say(why, why_size, "pixel format %u is not one Chlorophyll has a Vulkan format for", desc ? desc->pixel_format : 0);
      return NULL;
   }
   unsigned w = desc->width, h = desc->height ? desc->height : 1, dep = desc->depth ? desc->depth : 1;
   unsigned slices = desc->array_length ? desc->array_length : 1, mips = desc->mip_levels ? desc->mip_levels : 1;
   unsigned samples = desc->sample_count ? desc->sample_count : 1;
   if (!w) {
      say(why, why_size, "a texture that is no texels wide");
      return NULL;
   }
   VkImageType it = VK_IMAGE_TYPE_2D;
   VkImageCreateFlags flags = 0;
   unsigned layers = 1;
   switch (desc->type) {
   case CHL_TEX_1D: it = VK_IMAGE_TYPE_1D; h = dep = 1; break;
   case CHL_TEX_1D_ARRAY: it = VK_IMAGE_TYPE_1D; h = dep = 1; layers = slices; break;
   case CHL_TEX_2D: case CHL_TEX_2D_MS: dep = 1; break;
   case CHL_TEX_2D_ARRAY: case CHL_TEX_2D_MS_ARRAY: dep = 1; layers = slices; break;
   case CHL_TEX_CUBE: dep = 1; layers = 6; flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; break;
   case CHL_TEX_CUBE_ARRAY: dep = 1; layers = 6 * slices; flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; break;
   case CHL_TEX_3D: it = VK_IMAGE_TYPE_3D; break;
   default:
      say(why, why_size, "texture type %u (a texture buffer or not a MTLTextureType) is not served", desc->type);
      return NULL;
   }
   bool ms = desc->type == CHL_TEX_2D_MS || desc->type == CHL_TEX_2D_MS_ARRAY;
   if (ms != (samples > 1)) {
      say(why, why_size, "a texture of type %u with %u samples", desc->type, samples);
      return NULL;
   }
   unsigned usage = desc->usage ? desc->usage : (CHL_USAGE_SHADER_READ | CHL_USAGE_SHADER_WRITE | CHL_USAGE_RENDER_TARGET);
   bool ds = f->aspects & (ASPECT_DEPTH | ASPECT_STENCIL);
   VkImageUsageFlags vu = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
   if ((usage & CHL_USAGE_SHADER_WRITE) && !ds && !ms)
      vu |= VK_IMAGE_USAGE_STORAGE_BIT;
   if ((usage & CHL_USAGE_RENDER_TARGET) || ds)
      vu |= ds ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

   /* drop what the format cannot do here (a compressed format is never a render target); keep what was asked otherwise */
   VkFormatProperties fp;
   d->vk.vkGetPhysicalDeviceFormatProperties(d->pd, f->vk, &fp);
   VkFormatFeatureFlags ff = fp.optimalTilingFeatures;
   if (!(ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
      vu &= ~VK_IMAGE_USAGE_STORAGE_BIT;
   if (!(ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
      vu &= ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
   if (!(ff & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
      say(why, why_size, "%s cannot sample %s", d->name, f->name);
      return NULL;
   }
   if ((usage & CHL_USAGE_SHADER_WRITE) && !(vu & VK_IMAGE_USAGE_STORAGE_BIT) && !ds && !ms) {
      say(why, why_size, "%s cannot store to %s from a shader (MTLTextureUsageShaderWrite)", d->name, f->name);
      return NULL;
   }

   VkFormat list[2] = { f->vk, f->srgb_pair };
   VkImageFormatListCreateInfo fl = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, NULL, 2, list };
   const void *pnext = NULL;
   if (usage & CHL_USAGE_PIXEL_FORMAT_VIEW) {
      flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
   } else if (f->srgb_pair) {
      flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
      pnext = &fl;   /* sRGB <-> linear views, which Metal allows without the usage bit */
   }

   struct chl_texture *t = calloc(1, sizeof(*t));
   if (!t) {
      say(why, why_size, "out of memory making a texture");
      return NULL;
   }
   obj_init(&t->o, d, K_TEXTURE);
   t->fmt = f;
   t->vkfmt = f->vk;
   t->type = desc->type;
   t->width = w, t->height = h, t->depth = dep, t->levels = mips, t->layers = layers, t->samples = samples;
   t->usage = usage;
   t->vkusage = vu;
   memcpy(t->swizzle, desc->swizzle, 4);

   VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, pnext, flags, it, f->vk, { w, h, dep }, mips, layers,
                             (VkSampleCountFlagBits)samples, VK_IMAGE_TILING_OPTIMAL, vu, VK_SHARING_MODE_EXCLUSIVE, 0, NULL,
                             VK_IMAGE_LAYOUT_UNDEFINED };
   VkResult r = d->vk.vkCreateImage(d->dev, &ici, NULL, &t->image);
   if (r != VK_SUCCESS) {
      say(why, why_size, "%s would not make a %ux%ux%u %s with %u levels, %u layers and %u samples (VkResult %d)", d->name, w, h,
          dep, f->name, mips, layers, samples, r);
      free(t);
      return NULL;
   }
   VkMemoryRequirements req;
   d->vk.vkGetImageMemoryRequirements(d->dev, t->image, &req);
   VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, d->mt_host };
   if (d->vk.vkAllocateMemory(d->dev, &ai, NULL, &t->mem) != VK_SUCCESS ||
       d->vk.vkBindImageMemory(d->dev, t->image, t->mem, 0) != VK_SUCCESS) {
      say(why, why_size, "%s would not allocate %llu bytes for a %ux%u %s", d->name, (unsigned long long)req.size, w, h, f->name);
      if (t->mem)
         d->vk.vkFreeMemory(d->dev, t->mem, NULL);
      d->vk.vkDestroyImage(d->dev, t->image, NULL);
      free(t);
      return NULL;
   }
   t->view = make_view(t, view_type(t->type), 0, mips, 0, layers, true);
   transition_new_image(d, t);
   return t;
}

/* The libdrm stand-in's IOSurface import (gpu/freedreno-sweet): SURFACE_BO's answer, as the kernel ABI lays it out
 * (xnu/gpu-abi/sweet_adreno_abi.h, struct sweet_gpu_surface_bo; the first fields, the same in every minor). */
struct chl_sbo {
   uint32_t surface_id, handle;
   uint64_t size, iova;
   uint32_t offset, bytes_per_row, width, height, pixel_format, planes;
   uint32_t flags;               /* minor 1: SWEET_GPU_SBO_* */
   uint8_t rest[128 - 52];
};
#define CHL_SBO_WRAPPED  0x4     /* SWEET_GPU_SBO_WRAPPED: another process's memory, mapped read-only, no tail */
#define CHL_SBO_EXISTING 0x8     /* SWEET_GPU_SBO_EXISTING: this connection already had the BO (Turnip may hold it) */
extern int sweet_drm_surface_prime(int surface_id, struct chl_sbo *out);
extern int sweet_drm_gem_close(int fd, uint32_t handle);

/* A refused import gives back the BO SURFACE_BO just made (it holds a reference on the surface), unless this connection
 * already had it: then a texture of ours may be using it. */
static void
sbo_give_back(int pfd, const struct chl_sbo *sb)
{
   if (pfd >= 0)
      close(pfd);
   if (pfd >= 0 && sb->handle && !(sb->flags & CHL_SBO_EXISTING))
      sweet_drm_gem_close(-1, sb->handle);
}
extern int sweet_drm_bo_sync(int fd, uint32_t handle, uint64_t offset, uint64_t size);

struct chl_texture *
chl_texture_from_iosurface(struct chl_device *d, const struct chl_texture_desc *desc, int surface_id, char *why, size_t why_size)
{
   const struct chl_format *f = desc ? chl_format(desc->pixel_format) : NULL;
   if (!f || (f->aspects & ~ASPECT_COLOR) || f->bw != 1) {
      say(why, why_size, "an IOSurface as pixel format %u: only uncompressed colour formats can be laid over one",
          desc ? desc->pixel_format : 0);
      return NULL;
   }
   if (desc->type != CHL_TEX_2D || (desc->mip_levels > 1) || (desc->array_length > 1)) {
      say(why, why_size, "an IOSurface is one level of one 2D image (type %u, %u levels, %u slices asked)", desc->type,
          desc->mip_levels, desc->array_length);
      return NULL;
   }
   struct chl_sbo sb;
   int pfd = sweet_drm_surface_prime(surface_id, &sb);
   if (pfd < 0 || !sb.bytes_per_row || sb.planes > 1) {
      say(why, why_size, "the kernel would not describe IOSurface %d (planes %u, %u bytes a row)", surface_id, sb.planes,
          sb.bytes_per_row);
      sbo_give_back(pfd, &sb);
      return NULL;
   }
   /* Client memory (another process's pages, CoreUI's icon images among them): the kernel maps it read-only and with no
    * scratch tail after it, and the GPU's texture and tile fetches read past an image's last row, into the unmapped guard
    * page after every BO (SMMU faults while SpringBoard drew icons, 2026-10-26). Such a surface is copied instead
    * (bgmtl's own-memory fallback). */
   if (sb.flags & CHL_SBO_WRAPPED) {
      say(why, why_size, "IOSurface %d is another process's client memory (no tail page after it): copied, not mapped", surface_id);
      sbo_give_back(pfd, &sb);
      return NULL;
   }
   unsigned w = desc->width ? desc->width : sb.width, h = desc->height ? desc->height : sb.height;
   if (sb.width < w || sb.height < h || sb.bytes_per_row < w * f->bytes) {
      say(why, why_size, "IOSurface %d is %ux%u with %u bytes a row and the texture asks for %ux%u of %s", surface_id, sb.width,
          sb.height, sb.bytes_per_row, w, h, f->name);
      sbo_give_back(pfd, &sb);
      return NULL;
   }
   struct chl_texture *t = calloc(1, sizeof(*t));
   if (!t) {
      sbo_give_back(pfd, &sb);
      return NULL;
   }
   obj_init(&t->o, d, K_TEXTURE);
   t->fmt = f;
   t->vkfmt = f->vk;
   t->type = CHL_TEX_2D;
   t->width = w, t->height = h, t->depth = 1, t->levels = 1, t->layers = 1, t->samples = 1;
   t->usage = desc->usage ? desc->usage : (CHL_USAGE_SHADER_READ | CHL_USAGE_RENDER_TARGET);
   t->surface_handle = sb.handle;
   memcpy(t->swizzle, desc->swizzle, 4);
   VkFormatProperties fp;
   d->vk.vkGetPhysicalDeviceFormatProperties(d->pd, f->vk, &fp);
   t->vkusage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
   if (fp.linearTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
      t->vkusage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   if ((t->usage & CHL_USAGE_SHADER_WRITE) && (fp.linearTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
      t->vkusage |= VK_IMAGE_USAGE_STORAGE_BIT;

   VkSubresourceLayout plane = { sb.offset, 0, sb.bytes_per_row, 0, 0 };
   VkImageDrmFormatModifierExplicitCreateInfoEXT mod = { VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT, NULL,
                                                        0 /* DRM_FORMAT_MOD_LINEAR */, 1, &plane };
   VkExternalMemoryImageCreateInfo ext = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, &mod,
                                           VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ext, 0, VK_IMAGE_TYPE_2D, f->vk, { w, h, 1 }, 1, 1,
                             VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT, t->vkusage, VK_SHARING_MODE_EXCLUSIVE, 0,
                             NULL, VK_IMAGE_LAYOUT_UNDEFINED };
   VkResult r = d->vk.vkCreateImage(d->dev, &ici, NULL, &t->image);
   if (r != VK_SUCCESS) {
      say(why, why_size, "%s would not lay a %ux%u %s over IOSurface %d (%u bytes a row from offset %u): VkResult %d", d->name, w, h,
          f->name, surface_id, sb.bytes_per_row, sb.offset, r);
      sbo_give_back(pfd, &sb);
      free(t);
      return NULL;
   }
   VkMemoryRequirements req;
   d->vk.vkGetImageMemoryRequirements(d->dev, t->image, &req);
   VkMemoryDedicatedAllocateInfo ded = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, t->image, VK_NULL_HANDLE };
   VkImportMemoryFdInfoKHR imp = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, &ded, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                                   pfd };
   uint32_t mt = d->mt_host;
   if (!(req.memoryTypeBits & (1u << mt)))
      for (mt = 0; mt < d->memprops.memoryTypeCount && !(req.memoryTypeBits & (1u << mt)); mt++)
         ;
   VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, req.size, mt };
   r = d->vk.vkAllocateMemory(d->dev, &ai, NULL, &t->mem);   /* takes the fd */
   if (r != VK_SUCCESS || d->vk.vkBindImageMemory(d->dev, t->image, t->mem, 0) != VK_SUCCESS) {
      say(why, why_size, "%s would not import IOSurface %d's memory (%llu bytes): VkResult %d", d->name, surface_id,
          (unsigned long long)sb.size, r);
      if (r != VK_SUCCESS)
         sbo_give_back(pfd, &sb);   /* not taken by Turnip; a failed bind gives it back through vkFreeMemory */
      if (t->mem)
         d->vk.vkFreeMemory(d->dev, t->mem, NULL);
      d->vk.vkDestroyImage(d->dev, t->image, NULL);
      free(t);
      return NULL;
   }
   sweet_drm_bo_sync(-1, sb.handle, 0, 0);   /* the processor's dirty lines out before the GPU first reads or writes */
   t->view = make_view(t, VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1, true);
   transition_new_image(d, t);
   return t;
}

int
chl_texture_flush_surface(struct chl_texture *t)
{
   struct chl_texture *root = t && t->parent ? t->parent : t;
   if (!root || !root->surface_handle)
      return 0;
   return sweet_drm_bo_sync(-1, root->surface_handle, 0, 0);
}

struct chl_texture *
chl_texture_view(struct chl_texture *parent, unsigned pixel_format, unsigned type, unsigned base_level, unsigned level_count,
                 unsigned base_slice, unsigned slice_count, const unsigned char swizzle[4], char *why, size_t why_size)
{
   const struct chl_format *f = chl_format(pixel_format);
   if (!parent || !f) {
      say(why, why_size, "a view of %s in pixel format %u", parent ? "a texture" : "no texture", pixel_format);
      return NULL;
   }
   struct chl_texture *root = parent->parent ? parent->parent : parent;
   if (f->bytes != parent->fmt->bytes || f->bw != parent->fmt->bw) {
      say(why, why_size, "a %s view of a %s texture: not the same size per texel", f->name, parent->fmt->name);
      return NULL;
   }
   if (f->vk != root->fmt->vk && !(root->usage & CHL_USAGE_PIXEL_FORMAT_VIEW) && f->vk != root->fmt->srgb_pair) {
      say(why, why_size, "a %s view of a %s texture made without MTLTextureUsagePixelFormatView", f->name, root->fmt->name);
      return NULL;
   }
   level_count = level_count ? level_count : 1;
   unsigned layer_mul = (type == CHL_TEX_CUBE || type == CHL_TEX_CUBE_ARRAY) ? 6 : 1;
   unsigned layers = (slice_count ? slice_count : 1) * layer_mul;
   if (base_level + level_count > parent->levels || base_slice + layers > parent->layers) {
      say(why, why_size, "levels %u..%u and layers %u..%u of a texture with %u levels and %u layers", base_level,
          base_level + level_count - 1, base_slice, base_slice + layers - 1, parent->levels, parent->layers);
      return NULL;
   }
   struct chl_texture *v = calloc(1, sizeof(*v));
   if (!v) {
      say(why, why_size, "out of memory making a texture view");
      return NULL;
   }
   obj_init(&v->o, parent->o.dev, K_TEXTURE);
   obj_retain(&root->o);
   v->parent = root;
   v->image = root->image;
   v->fmt = f;
   v->vkfmt = f->vk;
   v->type = type;
   v->base_level = parent->base_level + base_level;
   v->base_layer = parent->base_layer + base_slice;
   v->width = root->width >> v->base_level ? root->width >> v->base_level : 1;
   v->height = root->height >> v->base_level ? root->height >> v->base_level : 1;
   v->depth = root->depth >> v->base_level ? root->depth >> v->base_level : 1;
   v->levels = level_count;
   v->layers = layers;
   v->samples = root->samples;
   v->usage = root->usage;
   v->vkusage = root->vkusage;
   if (swizzle)
      memcpy(v->swizzle, swizzle, 4);
   v->view = make_view(v, view_type(type), v->base_level, level_count, v->base_layer, layers, true);
   if (!v->view) {
      say(why, why_size, "%s would not make that view", parent->o.dev->name);
      obj_release(&root->o);
      free(v);
      return NULL;
   }
   return v;
}

void
chl_texture_size(const struct chl_texture *t, unsigned *w, unsigned *h, unsigned *d, unsigned *levels, unsigned *slices,
                 unsigned *samples)
{
   if (w) *w = t->width;
   if (h) *h = t->height;
   if (d) *d = t->depth;
   if (levels) *levels = t->levels;
   if (slices) *slices = t->layers;
   if (samples) *samples = t->samples;
}

struct region {
   struct chl_texture *t;
   unsigned level, slice;
   unsigned origin[3], size[3];
   size_t row_bytes, image_bytes;   /* tightly packed in the staging buffer */
   struct chl_buffer *staging;
   bool to_image;
};

static bool
region_of(struct chl_texture *t, unsigned level, unsigned slice, const unsigned origin[3], const unsigned size[3], struct region *r,
          char *why, size_t why_size)
{
   const struct chl_format *f = t->fmt;
   unsigned lw = t->width >> level ? t->width >> level : 1, lh = t->height >> level ? t->height >> level : 1;
   unsigned ld = t->depth >> level ? t->depth >> level : 1;
   r->t = t;
   r->level = level, r->slice = slice;
   for (int i = 0; i < 3; i++)
      r->origin[i] = origin ? origin[i] : 0;
   r->size[0] = size ? size[0] : lw;
   r->size[1] = size ? size[1] : lh;
   r->size[2] = size ? (size[2] ? size[2] : 1) : (t->type == CHL_TEX_3D ? ld : 1);
   if (level >= t->levels || slice >= t->layers || r->origin[0] + r->size[0] > lw || r->origin[1] + r->size[1] > lh ||
       r->origin[2] + r->size[2] > (t->type == CHL_TEX_3D ? ld : 1) || !r->size[0] || !r->size[1]) {
      say(why, why_size, "a region of %ux%ux%u at (%u, %u, %u) of level %u (%ux%ux%u) slice %u of a texture with %u levels and %u "
                         "layers", r->size[0], r->size[1], r->size[2], r->origin[0], r->origin[1], r->origin[2], level, lw, lh,
          ld, slice, t->levels, t->layers);
      return false;
   }
   if (t->samples > 1) {
      say(why, why_size, "texels of a multisample texture cannot be written or read by the processor");
      return false;
   }
   r->row_bytes = (size_t)((r->size[0] + f->bw - 1) / f->bw) * f->bytes;
   r->image_bytes = r->row_bytes * ((r->size[1] + f->bh - 1) / f->bh);
   return true;
}

static void
record_region_copy(struct chl_device *d, VkCommandBuffer cb, void *ctx)
{
   struct region *r = ctx;
   VkImageAspectFlags asp = vk_aspects(r->t->fmt);
   if (asp == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))
      asp = VK_IMAGE_ASPECT_DEPTH_BIT;
   VkBufferImageCopy c = { 0, (uint32_t)(r->row_bytes / r->t->fmt->bytes * r->t->fmt->bw), r->size[1],
                           { asp, r->t->base_level + r->level, r->t->base_layer + r->slice, 1 },
                           { (int32_t)r->origin[0], (int32_t)r->origin[1], (int32_t)r->origin[2] },
                           { r->size[0], r->size[1], r->size[2] } };
   if (r->to_image)
      d->vk.vkCmdCopyBufferToImage(cb, r->staging->vk, r->t->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
   else
      d->vk.vkCmdCopyImageToBuffer(cb, r->t->image, VK_IMAGE_LAYOUT_GENERAL, r->staging->vk, 1, &c);
}

int
chl_texture_replace(struct chl_texture *t, unsigned level, unsigned slice, const unsigned origin[3], const unsigned size[3],
                    const void *bytes, size_t bytes_per_row, size_t bytes_per_image, char *why, size_t why_size)
{
   struct chl_device *d = t->o.dev;
   struct region r;
   if (!bytes || !region_of(t, level, slice, origin, size, &r, why, why_size))
      return -1;
   size_t row = bytes_per_row ? bytes_per_row : r.row_bytes;
   size_t img = bytes_per_image ? bytes_per_image : row * ((r.size[1] + t->fmt->bh - 1) / t->fmt->bh);
   if (row < r.row_bytes) {
      say(why, why_size, "%zu bytes per row for a region whose rows need %zu", row, r.row_bytes);
      return -1;
   }
   r.staging = buffer_new(d, r.image_bytes * r.size[2], why, why_size);
   if (!r.staging)
      return -1;
   unsigned rows = (r.size[1] + t->fmt->bh - 1) / t->fmt->bh;
   for (unsigned z = 0; z < r.size[2]; z++)
      for (unsigned y = 0; y < rows; y++)
         memcpy((uint8_t *)r.staging->map + z * r.image_bytes + y * r.row_bytes, (const uint8_t *)bytes + z * img + y * row,
                r.row_bytes);
   r.to_image = true;
   /* into the prologue: done before anything committed from now on, without waiting here */
   pthread_mutex_lock(&d->pro_lock);
   VkCommandBuffer cb = pro_begin(d);
   if (cb) {
      full_barrier(d, cb);
      record_region_copy(d, cb, &r);
      hold(&d->pro_held, &d->n_pro_held, &d->cap_pro_held, &r.staging->o);
      hold(&d->pro_held, &d->n_pro_held, &d->cap_pro_held, &t->o);
   }
   pthread_mutex_unlock(&d->pro_lock);
   obj_release(&r.staging->o);
   if (!cb) {
      say(why, why_size, "no command buffer for the upload");
      return -1;
   }
   return 0;
}

int
chl_texture_get(struct chl_texture *t, unsigned level, unsigned slice, const unsigned origin[3], const unsigned size[3],
                void *bytes, size_t bytes_per_row, size_t bytes_per_image, char *why, size_t why_size)
{
   struct chl_device *d = t->o.dev;
   struct region r;
   if (!bytes || !region_of(t, level, slice, origin, size, &r, why, why_size))
      return -1;
   size_t row = bytes_per_row ? bytes_per_row : r.row_bytes;
   size_t img = bytes_per_image ? bytes_per_image : row * ((r.size[1] + t->fmt->bh - 1) / t->fmt->bh);
   r.staging = buffer_new(d, r.image_bytes * r.size[2], why, why_size);
   if (!r.staging)
      return -1;
   r.to_image = false;
   int rc = run_now(d, record_region_copy, &r, why, why_size);
   if (rc == 0) {
      unsigned rows = (r.size[1] + t->fmt->bh - 1) / t->fmt->bh;
      for (unsigned z = 0; z < r.size[2]; z++)
         for (unsigned y = 0; y < rows; y++)
            memcpy((uint8_t *)bytes + z * img + y * row, (const uint8_t *)r.staging->map + z * r.image_bytes + y * r.row_bytes,
                   r.row_bytes);
   }
   obj_release(&r.staging->o);
   return rc;
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Samplers
 */
static VkCompareOp
compare_op(unsigned mtl)
{
   static const VkCompareOp map[8] = { VK_COMPARE_OP_NEVER, VK_COMPARE_OP_LESS, VK_COMPARE_OP_EQUAL, VK_COMPARE_OP_LESS_OR_EQUAL,
                                       VK_COMPARE_OP_GREATER, VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                       VK_COMPARE_OP_ALWAYS };
   return mtl < 8 ? map[mtl] : VK_COMPARE_OP_ALWAYS;
}

static bool
sampler_info(const struct chl_sampler_desc *s, VkSamplerCreateInfo *ci, char *why, size_t why_size)
{
   static const VkSamplerAddressMode addr[6] = {
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,        /* ClampToEdge */
      VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE, /* MirrorClampToEdge */
      VK_SAMPLER_ADDRESS_MODE_REPEAT,               /* Repeat */
      VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,      /* MirrorRepeat */
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,      /* ClampToZero: a transparent black border */
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,      /* ClampToBorderColor */
   };
   for (int i = 0; i < 3; i++)
      if (s->address[i] >= 6) {
         say(why, why_size, "address mode %u is not one of MTLSamplerAddressMode's six", s->address[i]);
         return false;
      }
   memset(ci, 0, sizeof(*ci));
   ci->sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
   ci->magFilter = s->mag_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;   /* bicubic (2) is drawn as linear */
   ci->minFilter = s->min_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
   ci->mipmapMode = s->mip_filter == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
   ci->addressModeU = addr[s->address[0]];
   ci->addressModeV = addr[s->address[1]];
   ci->addressModeW = addr[s->address[2]];
   ci->anisotropyEnable = s->max_anisotropy > 1 && s->normalized_coordinates;
   ci->maxAnisotropy = s->max_anisotropy > 1 ? (float)s->max_anisotropy : 1.0f;
   ci->compareOp = compare_op(s->compare_function);
   ci->minLod = s->lod_min;
   /* MTLSamplerMipFilterNotMipmapped samples level 0 only */
   ci->maxLod = s->mip_filter == 0 ? 0.25f : s->lod_max;
   if (ci->maxLod < ci->minLod)
      ci->maxLod = ci->minLod;
   ci->borderColor = s->border_color == 1 ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK
                   : s->border_color == 2 ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
   ci->unnormalizedCoordinates = !s->normalized_coordinates;
   if (ci->unnormalizedCoordinates) {
      ci->mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      ci->minLod = ci->maxLod = 0;
      ci->anisotropyEnable = VK_FALSE;
      ci->minFilter = ci->magFilter;
      if (ci->addressModeU != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE && ci->addressModeU != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)
         ci->addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      if (ci->addressModeV != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE && ci->addressModeV != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)
         ci->addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   }
   return true;
}

struct chl_sampler *
chl_sampler_create(struct chl_device *d, const struct chl_sampler_desc *desc, char *why, size_t why_size)
{
   VkSamplerCreateInfo ci;
   if (!desc || !sampler_info(desc, &ci, why, why_size))
      return NULL;
   struct chl_sampler *s = calloc(1, sizeof(*s));
   if (!s) {
      say(why, why_size, "out of memory making a sampler");
      return NULL;
   }
   obj_init(&s->o, d, K_SAMPLER);
   ci.compareEnable = VK_FALSE;
   VkResult r1 = d->vk.vkCreateSampler(d->dev, &ci, NULL, &s->vk);
   ci.compareEnable = VK_TRUE;   /* Metal decides at the sample call whether a comparison happens: one of each */
   VkResult r2 = d->vk.vkCreateSampler(d->dev, &ci, NULL, &s->vk_compare);
   if (r1 != VK_SUCCESS || r2 != VK_SUCCESS) {
      say(why, why_size, "%s would not make that sampler", d->name);
      if (s->vk)
         d->vk.vkDestroySampler(d->dev, s->vk, NULL);
      if (s->vk_compare)
         d->vk.vkDestroySampler(d->dev, s->vk_compare, NULL);
      free(s);
      return NULL;
   }
   return s;
}

/* A shader's constexpr sampler (air_sampler_state) in Metal's words, then Vulkan's. */
static VkSampler
inline_sampler(struct chl_device *d, const struct air_resource *r, bool compare, char *why, size_t why_size)
{
   static const unsigned address[] = { 4 /* clamp_to_zero */, 0 /* clamp_to_edge */, 2 /* repeat */, 3 /* mirrored_repeat */,
                                       5 /* clamp_to_border */ };
   static const unsigned compare_map[9] = { 0, 1, 3, 4, 6, 2, 5, 7, 0 };
   const struct air_sampler_state *a = &r->sampler;
   struct chl_sampler_desc s = { 0 };
   for (int i = 0; i < 3; i++)
      s.address[i] = a->address[i] < 5 ? address[a->address[i]] : 0;
   s.min_filter = a->min_filter ? 1 : 0;
   s.mag_filter = a->mag_filter ? 1 : 0;
   s.mip_filter = a->mip_filter;
   s.compare_function = a->compare_func < 9 ? compare_map[a->compare_func] : 0;
   s.max_anisotropy = a->max_anisotropy;
   s.normalized_coordinates = !a->pixel_coord;
   s.lod_min = _mesa_half_to_float(a->lod_min);
   s.lod_max = _mesa_half_to_float(a->lod_max);
   s.border_color = a->border_color == 1 ? 1 : a->border_color == 2 ? 2 : 0;
   VkSamplerCreateInfo ci;
   if (!sampler_info(&s, &ci, why, why_size))
      return VK_NULL_HANDLE;
   ci.compareEnable = compare;
   /* one VkSampler per distinct state, shared by every pipeline (so that their layouts can be shared too) */
   pthread_mutex_lock(&d->share_lock);
   VkSampler out = VK_NULL_HANDLE;
   for (struct chl_shared_sampler *ss = d->samplers; ss && !out; ss = ss->next)
      if (!memcmp(&ss->ci, &ci, sizeof(ci)))
         out = ss->sampler;
   if (!out) {
      struct chl_shared_sampler *ss = calloc(1, sizeof(*ss));
      if (ss && d->vk.vkCreateSampler(d->dev, &ci, NULL, &ss->sampler) == VK_SUCCESS) {
         ss->ci = ci;
         ss->next = d->samplers;
         d->samplers = ss;
         out = ss->sampler;
      } else {
         free(ss);
      }
   }
   pthread_mutex_unlock(&d->share_lock);
   return out;
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Shaders and layouts
 */
static bool
translate(struct chl_device *d, void *mem_ctx, const struct chl_function *fn, mesa_shader_stage want, unsigned stage,
          struct chl_stage_info *out, struct air_vk_layout *layout, char *why, size_t why_size)
{
   struct air_constant_value *constants = (struct air_constant_value *)fn->constants;
   unsigned num_constants = fn->num_constants;
   if (!constants)
      num_constants = air_module_function_constants(mem_ctx, fn->bitcode, fn->size, &constants);
   _Static_assert(sizeof(struct chl_linked_function) == sizeof(struct air_linked_function), "linked function layouts");
   const struct air_to_nir_options opts = { .constants = constants, .num_constants = num_constants,
                                            .linked = (const struct air_linked_function *)fn->linked, .num_linked = fn->num_linked };
   const char *error = NULL;
   out->nir = air_to_nir(mem_ctx, fn->bitcode, fn->size, fn->entry, &opts, d->nir_options, &out->info, &error);
   if (!out->nir) {
      say(why, why_size, "%s", error ? error : "the function could not be translated");
      return false;
   }
   if (out->info.stage != want) {
      say(why, why_size, "%s is a %s function and this pipeline wants a %s one", fn->entry ? fn->entry : "the function",
          mesa_shader_stage_name(out->info.stage), mesa_shader_stage_name(want));
      return false;
   }
   for (unsigned i = 0; i < out->info.num_resources; i++) {
      const struct air_resource *r = &out->info.resources[i];
      switch (r->kind) {
      case AIR_RESOURCE_CONSTANT_BUFFER:
      case AIR_RESOURCE_DEVICE_BUFFER:
         if (r->index < 32)
            out->buffers |= 1u << r->index;
         break;
      case AIR_RESOURCE_TEXTURE:
      case AIR_RESOURCE_IMAGE:
         if (r->index < 32)
            out->textures |= 1u << r->index;
         break;
      case AIR_RESOURCE_SAMPLER:
         if (r->index < 32) {
            out->samplers |= 1u << r->index;
            if (r->inline_sampler)
               out->inline_samplers |= 1u << r->index;
         }
         break;
      default:
         break;
      }
   }
   if (!air_nir_to_vulkan(out->nir, &out->info, stage, layout, why, why_size))
      return false;
   out->nir->options = NULL;   /* Turnip sets its own when it takes the shader */
   return true;
}

/* Whether a sampler slot of a stage is read by a comparing (shadow) instruction. */
static bool
sampler_compares(nir_shader *nir, unsigned binding)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            int s = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
            if (!tex->is_shadow || s < 0)
               continue;
            nir_deref_instr *dr = nir_src_as_deref(tex->src[s].src);
            if (dr && dr->deref_type == nir_deref_type_var && dr->var->data.binding == binding)
               return true;
         }
      }
   }
   return false;
}

static bool
make_layout(struct chl_device *d, struct chl_layout *l, struct chl_stage_info *const *stages, unsigned n_stages,
            VkShaderStageFlags all, char *why, size_t why_size)
{
   VkDescriptorSetLayoutBinding b[AIR_VK_MAX_BINDINGS];
   unsigned nb = 0, ndesc = 0;
   for (unsigned i = 0; i < AIR_VK_MAX_BINDINGS; i++) {
      if (!l->bindings.type[i])
         continue;
      unsigned stage = i / AIR_VK_STAGE_BINDINGS;
      VkShaderStageFlags sf = n_stages == 1 ? all : (stage == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT);
      VkDescriptorType t = (VkDescriptorType)(l->bindings.type[i] - 1);
      b[nb] = (VkDescriptorSetLayoutBinding){ i, t, 1, sf, NULL };
      if (t == VK_DESCRIPTOR_TYPE_SAMPLER && stage < n_stages) {
         unsigned slot = i - stage * AIR_VK_STAGE_BINDINGS - AIR_VK_SAMPLER_BINDING;
         struct chl_stage_info *si = stages[stage];
         if (si->inline_samplers & (1u << slot)) {
            for (unsigned k = 0; k < si->info.num_resources; k++) {
               const struct air_resource *r = &si->info.resources[k];
               if (r->kind == AIR_RESOURCE_SAMPLER && r->index == slot && r->inline_sampler) {
                  l->immutable[i] = inline_sampler(d, r, sampler_compares(si->nir, i), why, why_size);
                  if (!l->immutable[i])
                     return false;
               }
            }
            b[nb].pImmutableSamplers = &l->immutable[i];
         }
      }
      if (t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && stage < n_stages) {
         unsigned slot = i - stage * AIR_VK_STAGE_BINDINGS - AIR_VK_TEXTURE_BINDING;
         const struct air_shader_info *ai = &stages[stage]->info;
         for (unsigned k = 0; k < ai->num_resources; k++) {
            const struct air_resource *r = &ai->resources[k];
            if (r->kind == AIR_RESOURCE_IMAGE && r->level_slots && slot >= r->level_base && slot < r->level_base + r->level_slots) {
               l->level_src[i] = (uint8_t)(r->index + 1);
               l->level[i] = (uint8_t)(slot - r->level_base + 1);
            }
         }
      }
      if (!b[nb].pImmutableSamplers)
         ndesc++;
      nb++;
   }
   if (ndesc > d->props.limits.maxPerStageDescriptorSamplers + 64 || ndesc > 32) {
      say(why, why_size, "the pipeline binds %u resources and a push-descriptor set holds 32", ndesc);
      return false;
   }
   l->push_bytes = l->bindings.push_bytes;
   /* the signature: stages, push-constant bytes and every binding (number, type, stages, immutable sampler); the same
    * signature is the same VkPipelineLayout for every pipeline of the device */
   struct { uint32_t binding, type, stages, pad; uint64_t sampler; } key_b[AIR_VK_MAX_BINDINGS];
   struct { uint32_t all, push, nb, pad; } key_h = { all, l->push_bytes, nb, 0 };
   memset(key_b, 0, sizeof(key_b));
   for (unsigned k = 0; k < nb; k++) {
      key_b[k].binding = b[k].binding;
      key_b[k].type = b[k].descriptorType;
      key_b[k].stages = b[k].stageFlags;
      key_b[k].sampler = b[k].pImmutableSamplers ? (uint64_t)(uintptr_t)b[k].pImmutableSamplers[0] : 0;
   }
   size_t key_size = sizeof(key_h) + nb * sizeof(key_b[0]);
   uint8_t key[sizeof(key_h) + sizeof(key_b)];
   memcpy(key, &key_h, sizeof(key_h));
   memcpy(key + sizeof(key_h), key_b, nb * sizeof(key_b[0]));
   pthread_mutex_lock(&d->share_lock);
   for (struct chl_shared_layout *sl = d->layouts; sl; sl = sl->next)
      if (sl->key_size == key_size && !memcmp(sl->key, key, key_size)) {
         l->set = sl->set;
         l->layout = sl->layout;
         pthread_mutex_unlock(&d->share_lock);
         return true;
      }
   struct chl_shared_layout *sl = calloc(1, sizeof(*sl));
   VkDescriptorSetLayoutCreateInfo sci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL,
                                           VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, nb, b };
   VkPushConstantRange pr = { all, 0, l->push_bytes };
   VkPipelineLayoutCreateInfo lci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, NULL, l->push_bytes ? 1u : 0u, &pr };
   bool ok = sl && (sl->key = malloc(key_size)) != NULL &&
             d->vk.vkCreateDescriptorSetLayout(d->dev, &sci, NULL, &sl->set) == VK_SUCCESS;
   if (ok) {
      lci.pSetLayouts = &sl->set;
      ok = d->vk.vkCreatePipelineLayout(d->dev, &lci, NULL, &sl->layout) == VK_SUCCESS;
   }
   if (!ok) {
      if (sl && sl->set)
         d->vk.vkDestroyDescriptorSetLayout(d->dev, sl->set, NULL);
      if (sl)
         free(sl->key);
      free(sl);
      pthread_mutex_unlock(&d->share_lock);
      say(why, why_size, "%s would not make the descriptor set layout or the pipeline layout of the pipeline", d->name);
      return false;
   }
   memcpy(sl->key, key, key_size);
   sl->key_size = key_size;
   sl->next = d->layouts;
   d->layouts = sl;
   l->set = sl->set;
   l->layout = sl->layout;
   pthread_mutex_unlock(&d->share_lock);
   return true;
}

static void
free_layout(struct chl_device *d, struct chl_layout *l)
{
   /* the set layout, pipeline layout and constexpr samplers are the device's (shared by signature, make_layout and
    * inline_sampler): they go when the device closes */
   (void)d;
   l->layout = VK_NULL_HANDLE;
   l->set = VK_NULL_HANDLE;
   memset(l->immutable, 0, sizeof(l->immutable));
}

static void
fill_info(struct chl_device *d, struct chl_pipeline_info *info, const struct chl_stage_info *s0, const struct chl_stage_info *s1)
{
   if (!info)
      return;
   memset(info, 0, sizeof(*info));
   info->max_threads_per_group = d->props.limits.maxComputeWorkGroupInvocations;
   info->thread_execution_width = 64;
   const struct chl_stage_info *st[2] = { s0, s1 };
   for (int i = 0; i < 2; i++) {
      if (!st[i] || !st[i]->nir)
         continue;
      info->buffers_used[i] = st[i]->buffers;
      info->textures_used[i] = st[i]->textures;
      info->samplers_used[i] = st[i]->samplers & ~st[i]->inline_samplers;
      for (unsigned k = 0; k < st[i]->info.num_resources; k++) {
         const struct air_resource *r = &st[i]->info.resources[k];
         if (r->argbuf_set != ~0u && r->argbuf_set < 32)
            info->argbufs[i] |= 1u << r->argbuf_set;
      }
   }
   if (s0)
      info->static_threadgroup_bytes = s0->info.threadgroup_bytes;
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Compute pipelines
 */
struct chl_compute_pipeline *
chl_compute_pipeline_create(struct chl_device *d, const struct chl_function *fn, struct chl_pipeline_info *info, char *why,
                            size_t why_size)
{
   if (!fn || !fn->bitcode || fn->size < 4) {
      say(why, why_size, "no function to make a compute pipeline of");
      return NULL;
   }
   struct chl_compute_pipeline *p = calloc(1, sizeof(*p));
   if (!p) {
      say(why, why_size, "out of memory making a compute pipeline");
      return NULL;
   }
   obj_init(&p->o, d, K_COMPUTE);
   pthread_mutex_init(&p->lock, NULL);
   p->mem_ctx = ralloc_context(NULL);
   if (!translate(d, p->mem_ctx, fn, MESA_SHADER_COMPUTE, 0, &p->cs, &p->layout.bindings, why, why_size))
      goto fail;
   for (unsigned i = 0; i < p->cs.info.num_resources; i++)
      if (p->cs.info.resources[i].kind == AIR_RESOURCE_THREADGROUP_BUFFER)
         p->dynamic_threadgroup = true;
   struct chl_stage_info *st[1] = { &p->cs };
   if (!make_layout(d, &p->layout, st, 1, VK_SHADER_STAGE_COMPUTE_BIT, why, why_size))
      goto fail;
   fill_info(d, info, &p->cs, NULL);
   return p;
fail:
   free_layout(d, &p->layout);
   ralloc_free(p->mem_ctx);
   free(p);
   return NULL;
}

static VkPipeline
compute_variant(struct chl_compute_pipeline *p, const unsigned threads[3], unsigned dyn_shared, char *why, size_t why_size)
{
   struct chl_device *d = p->o.dev;
   pthread_mutex_lock(&p->lock);
   for (unsigned i = 0; i < p->n_variants; i++)
      if (!memcmp(p->variant[i].threads, threads, sizeof(p->variant[i].threads)) && p->variant[i].shared == dyn_shared) {
         VkPipeline v = p->variant[i].vk;
         pthread_mutex_unlock(&p->lock);
         return v;
      }
   nir_shader *cs = nir_shader_clone(NULL, p->cs.nir);
   cs->info.workgroup_size[0] = threads[0];
   cs->info.workgroup_size[1] = threads[1];
   cs->info.workgroup_size[2] = threads[2];
   cs->info.workgroup_size_variable = false;
   if (dyn_shared)
      cs->info.shared_size = cs->info.shared_size + dyn_shared;
   VkPipelineShaderStageNirCreateInfoMESA nci = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA, NULL, cs };
   VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                      { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &nci, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                                        VK_NULL_HANDLE, "main", NULL },
                                      p->layout.layout, VK_NULL_HANDLE, 0 };
   VkPipeline v = VK_NULL_HANDLE;
   VkResult r = d->vk.vkCreateComputePipelines(d->dev, VK_NULL_HANDLE, 1, &ci, NULL, &v);
   ralloc_free(cs);
   if (r != VK_SUCCESS) {
      say(why, why_size, "Turnip would not compile %s for %ux%ux%u threads (VkResult %d)",
          p->cs.info.entry_point ? p->cs.info.entry_point : "the kernel", threads[0], threads[1], threads[2], r);
      pthread_mutex_unlock(&p->lock);
      return VK_NULL_HANDLE;
   }
   unsigned slot = p->n_variants < CHL_COMPUTE_VARIANTS ? p->n_variants++ : 0;
   if (p->variant[slot].vk)
      d->vk.vkDestroyPipeline(d->dev, p->variant[slot].vk, NULL);   /* XXX: may be in flight; 8 sizes per kernel is plenty */
   memcpy(p->variant[slot].threads, threads, sizeof(p->variant[slot].threads));
   p->variant[slot].shared = dyn_shared;
   p->variant[slot].vk = v;
   pthread_mutex_unlock(&p->lock);
   return v;
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Command buffers
 */
struct chl_cmdbuf *
chl_cmdbuf_create(struct chl_device *d, char *why, size_t why_size)
{
   pthread_mutex_lock(&d->free_lock);
   struct chl_cmdbuf *cb = d->free_cbs;
   if (cb)
      d->free_cbs = cb->next_free;
   pthread_mutex_unlock(&d->free_lock);
   if (!cb) {
      cb = calloc(1, sizeof(*cb));
      if (!cb) {
         say(why, why_size, "out of memory making a command buffer");
         return NULL;
      }
      cb->dev = d;
      VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, 0 };
      VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
      VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, VK_NULL_HANDLE,
                                         VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
      if (d->vk.vkCreateCommandPool(d->dev, &pci, NULL, &cb->pool) != VK_SUCCESS ||
          (ai.commandPool = cb->pool, d->vk.vkAllocateCommandBuffers(d->dev, &ai, &cb->cb) != VK_SUCCESS) ||
          d->vk.vkCreateFence(d->dev, &fci, NULL, &cb->fence) != VK_SUCCESS) {
         say(why, why_size, "%s would not make a command buffer", d->name);
         free(cb);
         return NULL;
      }
   }
   VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                   NULL };
   d->vk.vkBeginCommandBuffer(cb->cb, &bi);
   cb->committed = cb->done = cb->failed = false;
   cb->done_fn = NULL;
   cb->enc = ENC_NONE;
   cb->cp = NULL;
   cb->rp = NULL;
   cb->ds = NULL;
   memset(cb->bind, 0, sizeof(cb->bind));
   memset(cb->tg_len, 0, sizeof(cb->tg_len));
   return cb;
}

static void
cb_retire(struct chl_cmdbuf *cb)
{
   struct chl_device *d = cb->dev;
   for (unsigned i = 0; i < cb->n_held; i++)
      obj_release(cb->held[i]);
   cb->n_held = 0;
   pthread_mutex_lock(&d->free_lock);
   while (cb->slabs) {
      struct chl_slab *s = cb->slabs;
      cb->slabs = s->next;
      s->next = d->free_slabs;
      d->free_slabs = s;
   }
   cb->slab_used = 0;
   pthread_mutex_unlock(&d->free_lock);
}

static void
cb_recycle(struct chl_cmdbuf *cb)
{
   struct chl_device *d = cb->dev;
   d->vk.vkResetFences(d->dev, 1, &cb->fence);
   d->vk.vkResetCommandPool(d->dev, cb->pool, 0);
   pthread_mutex_lock(&d->free_lock);
   cb->next_free = d->free_cbs;
   d->free_cbs = cb;
   pthread_mutex_unlock(&d->free_lock);
}

int
chl_cmdbuf_commit(struct chl_cmdbuf *cb, chl_done_fn done, void *ctx, char *why, size_t why_size)
{
   struct chl_device *d = cb->dev;
   if (cb->enc == ENC_RENDER)
      chl_render_end(cb);
   cb->enc = ENC_NONE;
   /* what the GPU wrote is visible to the processor once the fence is in */
   VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_HOST_READ_BIT };
   d->vk.vkCmdPipelineBarrier(cb->cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
   d->vk.vkEndCommandBuffer(cb->cb);
   cb->done_fn = done;
   cb->done_ctx = ctx;
   cb->committed = true;

   pthread_mutex_lock(&d->queue_lock);
   pthread_mutex_lock(&d->pro_lock);
   VkResult r = pro_submit_locked(d);
   pthread_mutex_unlock(&d->pro_lock);
   if (r == VK_SUCCESS) {
      VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb->cb, 0, NULL };
      r = d->vk.vkQueueSubmit(d->queue, 1, &si, cb->fence);
   }
   if (r == VK_SUCCESS) {
      pthread_mutex_lock(&d->done_lock);
      cb->next_inflight = NULL;
      if (d->inflight_tail)
         d->inflight_tail->next_inflight = cb;
      else
         d->inflight_head = cb;
      d->inflight_tail = cb;
      pthread_cond_broadcast(&d->done_cv);
      pthread_mutex_unlock(&d->done_lock);
   }
   pthread_mutex_unlock(&d->queue_lock);
   if (r != VK_SUCCESS) {
      say(why, why_size, "the GPU did not take the command buffer (VkResult %d)", r);
      cb_retire(cb);
      cb->failed = cb->done = true;
      if (done)
         done(ctx, -1, "submission failed");
      return -1;
   }
   return 0;
}

int
chl_cmdbuf_wait(struct chl_cmdbuf *cb, char *why, size_t why_size)
{
   struct chl_device *d = cb->dev;
   if (!cb->committed) {
      say(why, why_size, "waiting for a command buffer that was never committed");
      return -1;
   }
   struct timespec deadline;
   clock_gettime(CLOCK_REALTIME, &deadline);
   deadline.tv_sec += CHL_FENCE_TIMEOUT_NS / 1000000000ull;
   pthread_mutex_lock(&d->done_lock);
   int e = 0;
   while (!cb->done && e != ETIMEDOUT)
      e = pthread_cond_timedwait(&d->done_cv, &d->done_lock, &deadline);
   bool done = cb->done, failed = cb->failed;
   pthread_mutex_unlock(&d->done_lock);
   if (!done || failed) {
      say(why, why_size, done ? "the GPU did not finish the command buffer" : "the command buffer did not complete in %llu s",
          CHL_FENCE_TIMEOUT_NS / 1000000000ull);
      return -1;
   }
   return 0;
}

void
chl_cmdbuf_release(struct chl_cmdbuf *cb)
{
   if (!cb)
      return;
   struct chl_device *d = cb->dev;
   if (!cb->committed) {
      d->vk.vkEndCommandBuffer(cb->cb);
      cb_retire(cb);
      cb_recycle(cb);
      return;
   }
   pthread_mutex_lock(&d->done_lock);
   while (!cb->done)
      pthread_cond_wait(&d->done_cv, &d->done_lock);
   pthread_mutex_unlock(&d->done_lock);
   cb_recycle(cb);
}

/* ---- the upload ring ---- */
static bool
ring_alloc(struct chl_cmdbuf *cb, size_t size, struct chl_buffer **buf, size_t *off, void **ptr)
{
   struct chl_device *d = cb->dev;
   size_t a = d->align;
   size_t at = (cb->slab_used + a - 1) & ~(a - 1);
   if (size > CHL_RING_SLAB)
      return false;
   if (!cb->slabs || at + size > CHL_RING_SLAB) {
      pthread_mutex_lock(&d->free_lock);
      struct chl_slab *s = d->free_slabs;
      if (s)
         d->free_slabs = s->next;
      pthread_mutex_unlock(&d->free_lock);
      if (!s) {
         s = calloc(1, sizeof(*s));
         char w[128];
         if (!s || !(s->buf = buffer_new(d, CHL_RING_SLAB, w, sizeof(w)))) {
            free(s);
            return false;
         }
         s->buf->ring = true;
      }
      s->next = cb->slabs;
      cb->slabs = s;
      at = 0;
   }
   *buf = cb->slabs->buf;
   *off = at;
   *ptr = (uint8_t *)cb->slabs->buf->map + at;
   cb->slab_used = at + size;
   return true;
}

/* ---- binding ---- */
void
chl_set_buffer(struct chl_cmdbuf *cb, unsigned stage, unsigned index, struct chl_buffer *buf, size_t offset)
{
   if (stage > 1 || index >= AIR_VK_BUFFER_SLOTS)
      return;
   cb->bind[stage].buf[index] = buf;
   cb->bind[stage].off[index] = offset;
   if (buf && !buf->ring)
      cb_hold(cb, &buf->o);
}

void
chl_set_bytes(struct chl_cmdbuf *cb, unsigned stage, unsigned index, const void *bytes, size_t length)
{
   if (stage > 1 || index >= AIR_VK_BUFFER_SLOTS)
      return;
   struct chl_buffer *b;
   size_t off;
   void *p;
   if (!ring_alloc(cb, length ? length : 4, &b, &off, &p))
      return;
   memcpy(p, bytes, length);
   cb->bind[stage].buf[index] = b;
   cb->bind[stage].off[index] = off;
}

void
chl_set_texture(struct chl_cmdbuf *cb, unsigned stage, unsigned index, struct chl_texture *tex)
{
   if (stage > 1 || index >= AIR_VK_TEXTURE_SLOTS)
      return;
   cb->bind[stage].tex[index] = tex;
   if (tex)
      cb_hold(cb, &tex->o);
}

void
chl_set_sampler(struct chl_cmdbuf *cb, unsigned stage, unsigned index, struct chl_sampler *s)
{
   if (stage > 1 || index >= AIR_VK_SAMPLER_SLOTS)
      return;
   cb->bind[stage].smp[index] = s;
   if (s)
      cb_hold(cb, &s->o);
}

/* Push the descriptors and the offset words a pipeline's layout reads, from what the encoder has bound. */
static int
flush_bindings(struct chl_cmdbuf *cb, VkPipelineBindPoint bp, const struct chl_layout *l, char *why, size_t why_size)
{
   struct chl_device *d = cb->dev;
   VkWriteDescriptorSet w[32];
   VkDescriptorBufferInfo bi[32];
   VkDescriptorImageInfo ii[32];
   uint32_t push[64] = { 0 };
   unsigned nw = 0;
   for (unsigned i = 0; i < AIR_VK_MAX_BINDINGS; i++) {
      if (!l->bindings.type[i] || l->immutable[i])
         continue;
      if (nw == 32) {
         say(why, why_size, "more than 32 resources in one draw or dispatch");
         return -1;
      }
      unsigned stage = i / AIR_VK_STAGE_BINDINGS, local = i % AIR_VK_STAGE_BINDINGS;
      const struct chl_stage_bind *sb = &cb->bind[stage];
      VkDescriptorType t = (VkDescriptorType)(l->bindings.type[i] - 1);
      w[nw] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, i, 0, 1, t, NULL, NULL, NULL };
      if (local < AIR_VK_TEXTURE_BINDING) {
         unsigned n = local;
         struct chl_buffer *b = sb->buf[n];
         if (!b) {
            say(why, why_size, "the %s stage reads [[buffer(%u)]] and nothing is bound there",
                bp == VK_PIPELINE_BIND_POINT_COMPUTE ? "compute" : stage ? "fragment" : "vertex", n);
            return -1;
         }
         size_t off = sb->off[n], aligned = off & ~((size_t)d->align - 1);
         if (off > b->size) {
            say(why, why_size, "[[buffer(%u)]] is bound at offset %zu of a %zu-byte buffer", n, off, b->size);
            return -1;
         }
         VkDeviceSize range = b->size - aligned;
         if (t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && range > d->props.limits.maxUniformBufferRange)
            range = d->props.limits.maxUniformBufferRange;
         bi[nw] = (VkDescriptorBufferInfo){ b->vk, aligned, range };
         w[nw].pBufferInfo = &bi[nw];
         push[stage * 32 + n] = (uint32_t)(off - aligned);
      } else if (local < AIR_VK_SAMPLER_BINDING) {
         unsigned n = local - AIR_VK_TEXTURE_BINDING;
         struct chl_texture *tx = sb->tex[n];
         unsigned level = 0;
         if (l->level_src[i]) {   /* a level slot: that level of the texture it stands for (its last one past the chain) */
            tx = sb->tex[l->level_src[i] - 1];
            level = tx && l->level[i] >= tx->levels ? tx->levels - 1 : l->level[i];
         }
         VkImageView v = VK_NULL_HANDLE;
         if (!tx) {
            tx = d->null_tex;
            level = 0;
         }
         if (tx)
            v = t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? storage_view(tx, level) : tx->view;
         ii[nw] = (VkDescriptorImageInfo){ VK_NULL_HANDLE, v, VK_IMAGE_LAYOUT_GENERAL };
         w[nw].pImageInfo = &ii[nw];
      } else if (local >= AIR_VK_INPUT_BINDING) {
         /* a framebuffer fetch: the colour attachment itself (Turnip reads it from the attachment state, GMEM or sysmem;
          * the descriptor still has to name it) */
         unsigned n = local - AIR_VK_INPUT_BINDING;
         if (cb->enc != ENC_RENDER || !cb->pass_color[n]) {
            say(why, why_size, "the fragment stage reads [[color(%u)]] and the render pass has no colour attachment %u", n, n);
            return -1;
         }
         ii[nw] = (VkDescriptorImageInfo){ VK_NULL_HANDLE, cb->pass_color[n], VK_IMAGE_LAYOUT_GENERAL };
         w[nw].pImageInfo = &ii[nw];
      } else {
         unsigned n = local - AIR_VK_SAMPLER_BINDING;
         struct chl_sampler *s = sb->smp[n];
         ii[nw] = (VkDescriptorImageInfo){ s ? s->vk : d->default_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
         w[nw].pImageInfo = &ii[nw];
      }
      nw++;
   }
   /* the null mask covers the texture slots of the stage the shader reads, bound or not */
   for (unsigned stage = 0; stage < 2; stage++)
      for (unsigned n = 0; n < AIR_VK_TEXTURE_SLOTS; n++)
         if (!cb->bind[stage].tex[n])
            push[stage * 32 + AIR_PUSH_NULL_MASK_WORD] |= 1u << n;
   if (nw)
      d->vk.vkCmdPushDescriptorSetKHR(cb->cb, bp, l->layout, 0, nw, w);
   if (l->push_bytes)
      d->vk.vkCmdPushConstants(cb->cb, l->layout,
                               bp == VK_PIPELINE_BIND_POINT_COMPUTE ? VK_SHADER_STAGE_COMPUTE_BIT
                                                                    : VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, l->push_bytes, push);
   return 0;
}

/* ---- compute encoder ---- */
void
chl_compute_begin(struct chl_cmdbuf *cb)
{
   if (cb->enc == ENC_RENDER)
      chl_render_end(cb);
   full_barrier(cb->dev, cb->cb);
   cb->enc = ENC_COMPUTE;
   cb->cp = NULL;
}

void
chl_set_compute_pipeline(struct chl_cmdbuf *cb, struct chl_compute_pipeline *p)
{
   cb->cp = p;
   cb_hold(cb, &p->o);
}

void
chl_set_threadgroup_memory(struct chl_cmdbuf *cb, unsigned index, size_t length)
{
   if (index < AIR_VK_BUFFER_SLOTS)
      cb->tg_len[index] = (uint32_t)length;
}

int
chl_dispatch(struct chl_cmdbuf *cb, const unsigned groups[3], const unsigned threads[3], char *why, size_t why_size)
{
   struct chl_device *d = cb->dev;
   struct chl_compute_pipeline *p = cb->cp;
   if (!p) {
      say(why, why_size, "a dispatch with no compute pipeline set");
      return -1;
   }
   if (!groups[0] || !groups[1] || !groups[2] || !threads[0] || !threads[1] || !threads[2])
      return 0;   /* Metal: an empty grid does nothing */
   unsigned dyn = 0;
   if (p->dynamic_threadgroup)
      for (unsigned i = 0; i < AIR_VK_BUFFER_SLOTS; i++)
         dyn += (cb->tg_len[i] + 15) & ~15u;
   TRACE("dispatch %s %ux%ux%u: compiling variant", p->cs.info.entry_point, threads[0], threads[1], threads[2]);
   VkPipeline v = compute_variant(p, threads, dyn, why, why_size);
   if (!v)
      return -1;
   TRACE("variant ready, binding");
   d->vk.vkCmdBindPipeline(cb->cb, VK_PIPELINE_BIND_POINT_COMPUTE, v);
   if (flush_bindings(cb, VK_PIPELINE_BIND_POINT_COMPUTE, &p->layout, why, why_size) != 0)
      return -1;
   TRACE("bound, dispatching");
   d->vk.vkCmdDispatch(cb->cb, groups[0], groups[1], groups[2]);
   /* the dispatches of a compute encoder are ordered in Metal: what one writes the next reads */
   VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
   d->vk.vkCmdPipelineBarrier(cb->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                              NULL, 0, NULL);
   return 0;
}

void
chl_compute_end(struct chl_cmdbuf *cb)
{
   full_barrier(cb->dev, cb->cb);
   cb->enc = ENC_NONE;
}

/* ---- blit encoder ---- */
void
chl_copy_buffer(struct chl_cmdbuf *cb, struct chl_buffer *src, size_t src_off, struct chl_buffer *dst, size_t dst_off, size_t size)
{
   struct chl_device *d = cb->dev;
   if (cb->enc == ENC_RENDER)
      chl_render_end(cb);
   full_barrier(d, cb->cb);
   VkBufferCopy c = { src_off, dst_off, size };
   d->vk.vkCmdCopyBuffer(cb->cb, src->vk, dst->vk, 1, &c);
   cb_hold(cb, &src->o);
   cb_hold(cb, &dst->o);
   full_barrier(d, cb->cb);
}

void
chl_fill_buffer(struct chl_cmdbuf *cb, struct chl_buffer *dst, size_t off, size_t size, uint8_t value)
{
   struct chl_device *d = cb->dev;
   if (cb->enc == ENC_RENDER)
      chl_render_end(cb);
   full_barrier(d, cb->cb);
   uint32_t word = value * 0x01010101u;
   if ((off | size) & 3) {
      /* vkCmdFillBuffer works in words: an unaligned fill goes through the ring */
      struct chl_buffer *b;
      size_t ro;
      void *p;
      if (ring_alloc(cb, size, &b, &ro, &p)) {
         memset(p, value, size);
         VkBufferCopy c = { ro, off, size };
         d->vk.vkCmdCopyBuffer(cb->cb, b->vk, dst->vk, 1, &c);
      }
   } else {
      d->vk.vkCmdFillBuffer(cb->cb, dst->vk, off, size, word);
   }
   cb_hold(cb, &dst->o);
   full_barrier(d, cb->cb);
}

static VkImageSubresourceLayers
layers_of(struct chl_texture *t, unsigned level, unsigned slice)
{
   VkImageAspectFlags a = vk_aspects(t->fmt);
   return (VkImageSubresourceLayers){ a, t->base_level + level, t->base_layer + slice, 1 };
}

int
chl_copy_texture(struct chl_cmdbuf *cb, struct chl_texture *src, unsigned slice, unsigned level, const unsigned origin[3],
                 const unsigned size[3], struct chl_texture *dst, unsigned dslice, unsigned dlevel, const unsigned dorigin[3])
{
   struct chl_device *d = cb->dev;
   if (src->fmt->bytes != dst->fmt->bytes || src->fmt->bw != dst->fmt->bw)
      return -1;
   if (cb->enc == ENC_RENDER)
      chl_render_end(cb);
   full_barrier(d, cb->cb);
   VkImageCopy c = { layers_of(src, level, slice), { (int32_t)origin[0], (int32_t)origin[1], (int32_t)origin[2] },
                     layers_of(dst, dlevel, dslice), { (int32_t)dorigin[0], (int32_t)dorigin[1], (int32_t)dorigin[2] },
                     { size[0], size[1], size[2] ? size[2] : 1 } };
   d->vk.vkCmdCopyImage(cb->cb, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
   cb_hold(cb, &src->o);
   cb_hold(cb, &dst->o);
   full_barrier(d, cb->cb);
   return 0;
}

static VkBufferImageCopy
buffer_image(struct chl_texture *t, size_t off, size_t bytes_per_row, size_t bytes_per_image, const unsigned size[3], unsigned slice,
             unsigned level, const unsigned origin[3])
{
   const struct chl_format *f = t->fmt;
   uint32_t row_texels = bytes_per_row ? (uint32_t)(bytes_per_row / f->bytes * f->bw) : 0;
   uint32_t rows = bytes_per_image && bytes_per_row ? (uint32_t)(bytes_per_image / bytes_per_row * f->bh) : 0;
   VkImageSubresourceLayers l = layers_of(t, level, slice);
   if (l.aspectMask == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))
      l.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
   return (VkBufferImageCopy){ off, row_texels, rows, l, { (int32_t)origin[0], (int32_t)origin[1], (int32_t)origin[2] },
                               { size[0], size[1], size[2] ? size[2] : 1 } };
}

int
chl_copy_buffer_to_texture(struct chl_cmdbuf *cb, struct chl_buffer *src, size_t off, size_t bytes_per_row, size_t bytes_per_image,
                           const unsigned size[3], struct chl_texture *dst, unsigned slice, unsigned level, const unsigned origin[3])
{
   struct chl_device *d = cb->dev;
   if (cb->enc == ENC_RENDER)
      chl_render_end(cb);
   full_barrier(d, cb->cb);
   VkBufferImageCopy c = buffer_image(dst, off, bytes_per_row, bytes_per_image, size, slice, level, origin);
   d->vk.vkCmdCopyBufferToImage(cb->cb, src->vk, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
   cb_hold(cb, &src->o);
   cb_hold(cb, &dst->o);
   full_barrier(d, cb->cb);
   return 0;
}

int
chl_copy_texture_to_buffer(struct chl_cmdbuf *cb, struct chl_texture *src, unsigned slice, unsigned level, const unsigned origin[3],
                           const unsigned size[3], struct chl_buffer *dst, size_t off, size_t bytes_per_row, size_t bytes_per_image)
{
   struct chl_device *d = cb->dev;
   if (cb->enc == ENC_RENDER)
      chl_render_end(cb);
   full_barrier(d, cb->cb);
   VkBufferImageCopy c = buffer_image(src, off, bytes_per_row, bytes_per_image, size, slice, level, origin);
   d->vk.vkCmdCopyImageToBuffer(cb->cb, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->vk, 1, &c);
   cb_hold(cb, &src->o);
   cb_hold(cb, &dst->o);
   full_barrier(d, cb->cb);
   return 0;
}

int
chl_generate_mipmaps(struct chl_cmdbuf *cb, struct chl_texture *t)
{
   struct chl_device *d = cb->dev;
   if (t->levels < 2)
      return 0;
   if (cb->enc == ENC_RENDER)
      chl_render_end(cb);
   full_barrier(d, cb->cb);
   for (unsigned l = 1; l < t->levels; l++) {
      int32_t sw = (int32_t)(t->width >> (l - 1)), sh = (int32_t)(t->height >> (l - 1)), dw = (int32_t)(t->width >> l),
              dh = (int32_t)(t->height >> l);
      VkImageBlit b = { layers_of(t, l - 1, 0), { { 0, 0, 0 }, { sw ? sw : 1, sh ? sh : 1, 1 } }, layers_of(t, l, 0),
                        { { 0, 0, 0 }, { dw ? dw : 1, dh ? dh : 1, 1 } } };
      b.srcSubresource.layerCount = b.dstSubresource.layerCount = t->layers;
      d->vk.vkCmdBlitImage(cb->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, t->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_LINEAR);
      full_barrier(d, cb->cb);
   }
   cb_hold(cb, &t->o);
   return 0;
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Destruction
 */
static void
destroy_obj(struct chl_obj *o)
{
   struct chl_device *d = o->dev;
   switch (o->kind) {
   case K_BUFFER: {
      struct chl_buffer *b = (struct chl_buffer *)o;
      d->vk.vkDestroyBuffer(d->dev, b->vk, NULL);
      d->vk.vkFreeMemory(d->dev, b->mem, NULL);
      break;
   }
   case K_TEXTURE: {
      struct chl_texture *t = (struct chl_texture *)o;
      if (t->view)
         d->vk.vkDestroyImageView(d->dev, t->view, NULL);
      for (unsigned i = 0; i < 16; i++)
         if (t->storage[i])
            d->vk.vkDestroyImageView(d->dev, t->storage[i], NULL);
      for (unsigned i = 0; i < t->n_att; i++)
         d->vk.vkDestroyImageView(d->dev, t->att[i].view, NULL);
      if (t->parent) {
         obj_release(&t->parent->o);
      } else {
         d->vk.vkDestroyImage(d->dev, t->image, NULL);
         if (t->mem)
            d->vk.vkFreeMemory(d->dev, t->mem, NULL);
      }
      break;
   }
   case K_SAMPLER: {
      struct chl_sampler *s = (struct chl_sampler *)o;
      d->vk.vkDestroySampler(d->dev, s->vk, NULL);
      d->vk.vkDestroySampler(d->dev, s->vk_compare, NULL);
      break;
   }
   case K_COMPUTE: {
      struct chl_compute_pipeline *p = (struct chl_compute_pipeline *)o;
      for (unsigned i = 0; i < p->n_variants; i++)
         d->vk.vkDestroyPipeline(d->dev, p->variant[i].vk, NULL);
      free_layout(d, &p->layout);
      ralloc_free(p->mem_ctx);
      break;
   }
   case K_RENDER: {
      struct chl_render_pipeline *p = (struct chl_render_pipeline *)o;
      if (p->vk)
         d->vk.vkDestroyPipeline(d->dev, p->vk, NULL);
      free_layout(d, &p->layout);
      ralloc_free(p->mem_ctx);
      break;
   }
   case K_DEPTH_STENCIL:
      break;
   }
   free(o);
}

#include "chl_render.c.inc"
