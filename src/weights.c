#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1   /* O_DIRECT */
#include "weights.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "kernels.h"
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>

/* GGUF */
#define GGUF_MAGIC             0x46554747u
#define GGUF_VERSION_SUPPORTED 3
#define GGUF_DEFAULT_ALIGNMENT 32
/* GGML tensor types */
#define GGML_TYPE_F32  0
#define GGML_TYPE_F16  1
#define GGML_TYPE_Q4_K 12
#define GGML_TYPE_Q6_K 14
/* GGUF metadata types */
#define GGUF_TYPE_UINT8   0
#define GGUF_TYPE_INT8    1
#define GGUF_TYPE_UINT16  2
#define GGUF_TYPE_INT16   3
#define GGUF_TYPE_UINT32  4
#define GGUF_TYPE_INT32   5
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TYPE_BOOL    7
#define GGUF_TYPE_STRING  8
#define GGUF_TYPE_ARRAY   9
#define GGUF_TYPE_UINT64  10
#define GGUF_TYPE_INT64   11
#define GGUF_TYPE_FLOAT64 12
/* ------------------------------------------------------------
 * File helpers
 * ------------------------------------------------------------ */
static int read_full(
    int fd,
    void *buffer,
    size_t size,
    uint64_t offset
)
{
    unsigned char *dst = buffer;
    while (size > 0) {
        ssize_t n = pread(fd, dst, size, (off_t)offset);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (n == 0)
            return -1;
        dst += n;
        offset += (uint64_t)n;
        size -= (size_t)n;
    }
    return 0;
}

/*
 * Expert reads use O_DIRECT (ROASTER_ODIRECT=0 to disable): random expert-sized reads from a
 * model file much larger than RAM go straight from the SSD into our buffer with no page-cache
 * copy or reclaim. O_DIRECT needs 4 KB-aligned file offset/length/memory, and GGUF tensor
 * offsets are only 32-byte aligned, so each read fetches the enclosing aligned range into a
 * per-thread bounce buffer and copies the wanted bytes out. Any failure falls back to a
 * normal buffered pread (and disables O_DIRECT from then on).
 */
#define DIRECT_ALIGN 4096u
#define BOUNCE_SLOTS 16
static void *g_bounce[BOUNCE_SLOTS];
static size_t g_bounce_cap[BOUNCE_SLOTS];
static volatile int g_direct_failed = 0;

static int io_thread_index(void)
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

static void *get_bounce(int tid, size_t need)
{
    if (tid < 0 || tid >= BOUNCE_SLOTS)
        return NULL;
    if (g_bounce_cap[tid] < need) {
        free(g_bounce[tid]);
        g_bounce[tid] = NULL;
        g_bounce_cap[tid] = 0;
        size_t cap = (need + (1u << 20) - 1) / (1u << 20) * (1u << 20);   /* 1 MB steps */
        if (posix_memalign(&g_bounce[tid], DIRECT_ALIGN, cap) != 0) {
            g_bounce[tid] = NULL;
            return NULL;
        }
        g_bounce_cap[tid] = cap;
    }
    return g_bounce[tid];
}

static int read_expert_bytes(Weights *w, void *dst, size_t size, uint64_t off)
{
    if (w->fd_direct >= 0 && !g_direct_failed && size > 0) {
        uint64_t a_off = off & ~(uint64_t)(DIRECT_ALIGN - 1);
        size_t head = (size_t)(off - a_off);
        size_t len = (head + size + DIRECT_ALIGN - 1) & ~(size_t)(DIRECT_ALIGN - 1);
        unsigned char *bounce = get_bounce(io_thread_index(), len);
        if (bounce) {
            size_t got = 0;
            int ok = 1;
            while (got < head + size) {      /* may stop short at end of file: only need head+size */
                ssize_t n = pread(w->fd_direct, bounce + got, len - got, (off_t)(a_off + got));
                if (n < 0) {
                    if (errno == EINTR) continue;
                    ok = 0;
                    break;
                }
                if (n == 0) { ok = 0; break; }
                got += (size_t)n;
            }
            if (ok) {
                memcpy(dst, bounce + head, size);
                return 0;
            }
            if (!g_direct_failed) {
                g_direct_failed = 1;
                fprintf(stderr, "Roaster: O_DIRECT read failed (%s); falling back to buffered reads\n",
                        strerror(errno));
            }
        }
    }
    return read_full(w->fd, dst, size, off);
}
static int read_bytes(
    int fd,
    void *value,
    size_t size,
    uint64_t *pos
)
{
    if (read_full(fd, value, size, *pos) != 0)
        return -1;
    *pos += size;
    return 0;
}
/* ------------------------------------------------------------
 * GGUF strings
 * ------------------------------------------------------------ */
static char *read_string(
    int fd,
    uint64_t *pos
)
{
    uint64_t length;
    if (read_bytes(fd, &length, sizeof(length), pos) != 0)
        return NULL;
    if (length > SIZE_MAX - 1)
        return NULL;
    char *str = malloc((size_t)length + 1);
    if (!str)
        return NULL;
    if (read_full(fd, str, (size_t)length, *pos) != 0) {
        free(str);
        return NULL;
    }
    *pos += length;
    str[length] = '\0';
    return str;
}
/* ------------------------------------------------------------
 * Metadata
 * ------------------------------------------------------------ */
static int skip_string(
    int fd,
    uint64_t *pos
)
{
    uint64_t length;
    if (read_bytes(fd, &length, sizeof(length), pos) != 0)
        return -1;
    if (length > UINT64_MAX - *pos)
        return -1;
    *pos += length;
    return 0;
}
static int skip_value(
    int fd,
    uint32_t type,
    uint64_t *pos
)
{
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
        {
            uint8_t value;
            return read_bytes(fd, &value, sizeof(value), pos);
        }
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
        {
            uint16_t value;
            return read_bytes(fd, &value, sizeof(value), pos);
        }
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        {
            uint32_t value;
            return read_bytes(fd, &value, sizeof(value), pos);
        }
        case GGUF_TYPE_FLOAT32:
        {
            float value;
            return read_bytes(fd, &value, sizeof(value), pos);
        }
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
        {
            uint64_t value;
            return read_bytes(fd, &value, sizeof(value), pos);
        }
        case GGUF_TYPE_STRING:
            return skip_string(fd, pos);
        case GGUF_TYPE_ARRAY:
        {
            uint32_t element_type;
            uint64_t count;
            if (read_bytes(fd, &element_type, sizeof(element_type), pos) != 0)
                return -1;
            if (read_bytes(fd, &count, sizeof(count), pos) != 0)
                return -1;
            for (uint64_t i = 0; i < count; i++) {
                if (skip_value(fd, element_type, pos) != 0)
                    return -1;
            }
            return 0;
        }
        default:
            return -1;
    }
}
/* ------------------------------------------------------------
 * Tensor helpers
 * ------------------------------------------------------------ */
static uint64_t tensor_element_count(
    uint32_t n_dims,
    const uint64_t *dims
)
{
    uint64_t elements = 1;
    for (uint32_t i = 0; i < n_dims; i++) {
        if (dims[i] != 0 &&
            elements > UINT64_MAX / dims[i])
            return 0;
        elements *= dims[i];
    }
    return elements;
}
static uint64_t tensor_size_bytes(
    uint32_t type,
    uint64_t elements
)
{
    switch (type) {
        case GGML_TYPE_F32:
            return elements * 4;
        case GGML_TYPE_F16:
            return elements * 2;
        case GGML_TYPE_Q4_K:
            return ((elements + 255) / 256) * 144;
        case GGML_TYPE_Q6_K:
            return ((elements + 255) / 256) * 210;
        default:
            return 0;
    }
}
static bool is_expert_tensor(
    const char *name
)
{
    return strstr(name, ".ffn_gate_exps.weight") ||
           strstr(name, ".ffn_up_exps.weight")   ||
           strstr(name, ".ffn_down_exps.weight");
}
/*
 * Allocate page-isolated buffers and lock them immediately. Page-aligning and rounding each
 * allocation prevents munlock() on one tensor from unlocking pages shared with another allocation.
 */
static void *malloc_locked(uint64_t requested_size, uint64_t *locked_size)
{
    if (locked_size)
        *locked_size = 0;
    if (requested_size == 0 || requested_size > SIZE_MAX)
        return NULL;
    long page_long = sysconf(_SC_PAGESIZE);
    if (page_long <= 0)
        return NULL;
    size_t page = (size_t)page_long;
    size_t size = (size_t)requested_size;
    if (size > SIZE_MAX - (page - 1))
        return NULL;
    size_t rounded = ((size + page - 1) / page) * page;
    void *ptr = NULL;
    int rc = posix_memalign(&ptr, page, rounded);
    if (rc != 0) {
        errno = rc;
        perror("posix_memalign model weights");
        return NULL;
    }
    if (mlock(ptr, rounded) != 0) {
        perror("mlock model weights (check RLIMIT_MEMLOCK and available RAM)");
        free(ptr);
        return NULL;
    }
    if (locked_size)
        *locked_size = (uint64_t)rounded;
    return ptr;
}
static void unlock_free(void *ptr, uint64_t locked_size)
{
    if (!ptr)
        return;
    if (kernel_vk_free_weights(ptr))      /* driver memory: released with the Vulkan device */
        return;
    /* Drop any Vulkan buffer importing this exact host allocation before free. */
    kernel_vk_release_host_ptr(ptr);
    if (locked_size != 0 && munlock(ptr, (size_t)locked_size) != 0)
        perror("munlock model weights");
    free(ptr);
}
/* ------------------------------------------------------------
 * Resident tensor loading
 * ------------------------------------------------------------ */
static int load_tensor(
    Weights *w,
    TensorInfo *tensor
)
{
    if (tensor->size == 0)
        return -1;
    tensor->data = NULL;
    tensor->locked_size = 0;
    /* GPU-consumed quantized matrices go to driver memory (fast GPU reads). Everything the CPU
     * reads repeatedly (f32 norms/router, the embedding table) stays in normal mlocked memory,
     * because driver memory is uncached for the CPU. */
    if ((tensor->type == GGML_TYPE_Q4_K || tensor->type == GGML_TYPE_Q6_K) &&
        strcmp(tensor->name, "token_embd.weight") != 0) {
        size_t granted = 0;
        void *dm = kernel_vk_alloc_weights((size_t)tensor->size, &granted);
        if (dm)
            tensor->data = dm;
    }
    if (!tensor->data) {
        tensor->data = malloc_locked(tensor->size, &tensor->locked_size);
        if (!tensor->data)
            return -1;
    }
    uint64_t offset = w->tensor_data_offset + tensor->offset;
    if (read_full(w->fd, tensor->data, (size_t)tensor->size, offset) != 0) {
        unlock_free(tensor->data, tensor->locked_size);
        tensor->data = NULL;
        tensor->locked_size = 0;
        return -1;
    }
    return 0;
}
/* ------------------------------------------------------------
 * Expert locations
 * ------------------------------------------------------------ */
static int register_expert_tensor(
    Weights *w,
    TensorInfo *tensor
)
{
    if (tensor->n_dims != 3)
        return -1;
    if (tensor->dims[2] != NUM_EXPERTS)
        return -1;
    if (tensor->size % NUM_EXPERTS != 0)
        return -1;
    int layer;
    if (sscanf(tensor->name, "blk.%d.", &layer) != 1)
        return -1;
    if (layer < 0 || layer >= NUM_LAYERS)
        return -1;
    uint64_t slice_size = tensor->size / NUM_EXPERTS;
    for (int expert = 0; expert < NUM_EXPERTS; expert++) {
        uint64_t offset =
            w->tensor_data_offset +
            tensor->offset +
            (uint64_t)expert * slice_size;
        ExpertLocation *loc =
            &w->expert_location[layer][expert];
        if (strstr(tensor->name, ".ffn_gate_exps.weight")) {
            loc->gate_offset = offset;
            loc->gate_size = slice_size;
            loc->gate_type = tensor->type;
        } else if (strstr(tensor->name, ".ffn_up_exps.weight")) {
            loc->up_offset = offset;
            loc->up_size = slice_size;
            loc->up_type = tensor->type;
        } else if (strstr(tensor->name, ".ffn_down_exps.weight")) {
            loc->down_offset = offset;
            loc->down_size = slice_size;
            loc->down_type = tensor->type;
        }
    }
    return 0;
}
/* ------------------------------------------------------------
 * Expert cache
 * ------------------------------------------------------------ */
static void free_cached_expert(
    CachedExpert *slot
)
{
    if (!slot)
        return;
    if (!slot->in_slab) {
        unlock_free(slot->gate, slot->gate_lock_size);
        unlock_free(slot->up, slot->up_lock_size);
        unlock_free(slot->down, slot->down_lock_size);
    }
    slot->in_slab = 0;
    slot->gate = NULL;
    slot->up = NULL;
    slot->down = NULL;
    slot->gate_lock_size = 0;
    slot->up_lock_size = 0;
    slot->down_lock_size = 0;
    slot->expert_id = -1;
    slot->last_used = 0;
    slot->use_count = 0;
}
static int find_cache_slot(
    LayerCache *cache
)
{
    int victim = 0;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < cache->capacity; i++) {
        CachedExpert *slot =
            &cache->slots[i];
        if (slot->expert_id < 0)
            return i;
        if (slot->last_used < oldest) {
            oldest = slot->last_used;
            victim = i;
        }
    }
    return victim;
}
/*
 * Expert slots keep their (mlocked) buffers when an expert is evicted.
 * Every expert of a layer has the same slice sizes, so a slot is simply
 * overwritten in place: no free/posix_memalign/mlock per miss, and any
 * Vulkan host-pointer import of the buffer stays valid.
 */
static int ensure_buf(void **p, uint64_t *lock_size, uint64_t need)
{
    if (*p && *lock_size >= need)
        return 0;
    if (*p) {
        unlock_free(*p, *lock_size);
        *p = NULL;
        *lock_size = 0;
    }
    *p = malloc_locked(need, lock_size);
    return *p ? 0 : -1;
}

static uint64_t align_up_u64(uint64_t v, uint64_t a)
{
    return (v + a - 1) / a * a;
}

/* Page-aligned allocation without mlock (pages are committed only when first touched). */
static void *malloc_unlocked(uint64_t requested, uint64_t *out_size)
{
    long pl = sysconf(_SC_PAGESIZE);
    size_t page = pl > 0 ? (size_t)pl : 4096u;
    size_t rounded = (size_t)align_up_u64(requested, page);
    void *ptr = NULL;
    int rc = posix_memalign(&ptr, page, rounded);
    if (rc != 0) {
        errno = rc;
        perror("posix_memalign expert slab");
        return NULL;
    }
    *out_size = rounded;
    return ptr;
}

/*
 * Carve this slot's three matrices out of the layer slab (allocated once; the whole layer cache
 * is imported by the GPU once and addressed by offset). Returns 1 = assigned, 0 = not applicable
 * (use individual allocations), -1 = error.
 */
static int slab_assign(LayerCache *cache, CachedExpert *slot, const ExpertLocation *loc)
{
    if (slot->in_slab && slot->gate && slot->up && slot->down)
        return 1;
    long pl = sysconf(_SC_PAGESIZE);
    uint64_t align = pl > 4096 ? (uint64_t)pl : 4096u;   /* also >= Vulkan min host-pointer alignment */
    uint64_t sg = align_up_u64(loc->gate_size, align);
    uint64_t su = align_up_u64(loc->up_size, align);
    uint64_t sd = align_up_u64(loc->down_size, align);
    uint64_t stride = sg + su + sd;
    if (!cache->slab) {
        uint64_t slab_bytes = 0;
        size_t granted = 0;
        void *dm = kernel_vk_alloc_weights((size_t)(stride * (uint64_t)cache->capacity), &granted);
        if (dm) {
            cache->slab = dm;
            slab_bytes = granted;
            cache->slab_driver = 1;
        } else {
            cache->slab = malloc_unlocked(stride * (uint64_t)cache->capacity, &slab_bytes);
            if (!cache->slab)
                return -1;
            /* Import the whole slab once; matrices are then addressed by descriptor offset. */
            (void)kernel_vk_register_region(cache->slab, (size_t)slab_bytes);
        }
        cache->slab_size = slab_bytes;
        cache->slab_stride = stride;
    }
    if (cache->slab_stride != stride)
        return 0;
    if (slot->gate && !slot->in_slab) {            /* drop any earlier individual allocation */
        unlock_free(slot->gate, slot->gate_lock_size);
        unlock_free(slot->up, slot->up_lock_size);
        unlock_free(slot->down, slot->down_lock_size);
    }
    uint64_t idx = (uint64_t)(slot - cache->slots);
    char *base = (char *)cache->slab + idx * stride;
    /* Lock only this slot's range, so memory is committed as the cache actually fills. */
    if (!cache->slab_driver && mlock(base, (size_t)stride) != 0) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            perror("mlock expert slot (continuing unlocked; raise --ulimit memlock)");
        }
    }
    slot->gate = base;
    slot->up = base + sg;
    slot->down = base + sg + su;
    slot->gate_lock_size = sg;
    slot->up_lock_size = su;
    slot->down_lock_size = sd;
    slot->in_slab = 1;
    return 1;
}

static int ensure_slot_buffers(
    LayerCache *cache,
    CachedExpert *slot,
    const ExpertLocation *loc
)
{
    if (loc->gate_size == 0 || loc->up_size == 0 || loc->down_size == 0)
        return -1;
    /* Always use the per-layer slab: one allocation and ONE GPU import per layer instead of
     * three imports per slot (which can exceed maxMemoryAllocationCount = 4096). */
    {
        int rc = slab_assign(cache, slot, loc);
        if (rc != 0)
            return rc > 0 ? 0 : -1;
    }
    if (ensure_buf(&slot->gate, &slot->gate_lock_size, loc->gate_size) != 0 ||
        ensure_buf(&slot->up,   &slot->up_lock_size,   loc->up_size)   != 0 ||
        ensure_buf(&slot->down, &slot->down_lock_size, loc->down_size) != 0)
        return -1;
    return 0;
}

/* Forget which expert a slot holds but keep its buffers for reuse. */
static void release_slot_entry(LayerCache *cache, CachedExpert *slot)
{
    if (slot->expert_id >= 0 && slot->expert_id < NUM_EXPERTS)
        cache->slot_by_expert[slot->expert_id] = -1;
    slot->expert_id = -1;
    slot->last_used = 0;
    slot->use_count = 0;
}

static void claim_slot(
    Weights *w,
    LayerCache *cache,
    CachedExpert *slot,
    int slot_index,
    int expert,
    const ExpertLocation *loc
)
{
    slot->expert_id = expert;
    slot->gate_type = loc->gate_type;
    slot->up_type = loc->up_type;
    slot->down_type = loc->down_type;
    slot->last_used = ++w->clock;
    slot->use_count = 1;
    cache->slot_by_expert[expert] = slot_index;
}

static int load_expert_into_slot(
    Weights *w,
    int layer,
    int expert,
    int slot_index
)
{
    LayerCache *cache =
        &w->cache[layer];
    CachedExpert *slot =
        &cache->slots[slot_index];
    ExpertLocation *loc =
        &w->expert_location[layer][expert];
    if (loc->gate_size == 0 ||
        loc->up_size == 0 ||
        loc->down_size == 0) {
        fprintf(
            stderr,
            "Missing expert weights: layer=%d expert=%d\n",
            layer,
            expert
        );
        return -1;
    }
    if (ensure_slot_buffers(cache, slot, loc) != 0) {
        free_cached_expert(slot);
        return -1;
    }
    if (read_expert_bytes(w, slot->gate, (size_t)loc->gate_size, loc->gate_offset) != 0 ||
        read_expert_bytes(w, slot->up,   (size_t)loc->up_size,   loc->up_offset)   != 0 ||
        read_expert_bytes(w, slot->down, (size_t)loc->down_size, loc->down_offset) != 0) {
        free_cached_expert(slot);
        return -1;
    }
    claim_slot(w, cache, slot, slot_index, expert, loc);
    return 0;
}

/* ------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------ */

int weights_load(
    Weights *w,
    const char *path,
    int max_experts_per_layer
)
{
    if (!w || !path)
        return -1;
    if (max_experts_per_layer <= 0 ||
        max_experts_per_layer > NUM_EXPERTS) {
        fprintf(
            stderr,
            "Invalid max experts per layer: %d\n",
            max_experts_per_layer
        );
        return -1;
    }
    memset(w, 0, sizeof(*w));
    w->fd = -1;
    w->fd_direct = -1;
    w->alignment = GGUF_DEFAULT_ALIGNMENT;
    w->max_experts_per_layer =
        max_experts_per_layer;
    w->fd = open(path, O_RDONLY);
    if (w->fd < 0) {
        perror("open GGUF");
        return -1;
    }
    /* O_DIRECT for expert reads; if unavailable (-1) buffered reads are used instead. */
    w->fd_direct = open(path, O_RDONLY | O_DIRECT);
    uint64_t pos = 0;
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    if (read_bytes(w->fd, &magic, sizeof(*&magic), &pos) != 0 ||
        read_bytes(w->fd, &version, sizeof(*&version), &pos) != 0 ||
        read_bytes(w->fd, &tensor_count, sizeof(*&tensor_count), &pos) != 0 ||
        read_bytes(w->fd, &metadata_count, sizeof(*&metadata_count), &pos) != 0) {
        fprintf(stderr, "Failed to read GGUF header\n");
        weights_free(w);
        return -1;
    }
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic\n");
        weights_free(w);
        return -1;
    }
    if (version != GGUF_VERSION_SUPPORTED) {
        fprintf(
            stderr,
            "Unsupported GGUF version: %u\n",
            version
        );
        weights_free(w);
        return -1;
    }
    if (tensor_count == 0) {
        fprintf(stderr, "GGUF contains no tensors\n");
        weights_free(w);
        return -1;
    }
    /* Metadata */
    for (uint64_t i = 0; i < metadata_count; i++) {
        char *key =
            read_string(w->fd, &pos);
        if (!key) {
            weights_free(w);
            return -1;
        }
        uint32_t type;
        if (read_bytes(w->fd, &type, sizeof(*&type), &pos) != 0) {
            free(key);
            weights_free(w);
            return -1;
        }
        if (strcmp(key, "general.alignment") == 0 &&
            type == GGUF_TYPE_UINT32) {
            uint32_t alignment;
            if (read_bytes(w->fd, &alignment, sizeof(*&alignment), &pos) != 0) {
                free(key);
                weights_free(w);
                return -1;
            }
            if (alignment != 0)
                w->alignment = alignment;
        } else {
            if (skip_value(
                    w->fd,
                    type,
                    &pos) != 0) {
                fprintf(
                    stderr,
                    "Failed skipping metadata: %s\n",
                    key
                );
                free(key);
                weights_free(w);
                return -1;
            }
        }
        free(key);
    }
    /* Tensor table */
    w->tensor_count = tensor_count;
    w->tensors = calloc(
        (size_t)tensor_count,
        sizeof(TensorInfo)
    );
    if (!w->tensors) {
        weights_free(w);
        return -1;
    }
    for (uint64_t i = 0; i < tensor_count; i++) {
        TensorInfo *tensor =
            &w->tensors[i];
        tensor->name =
            read_string(w->fd, &pos);
        if (!tensor->name) {
            weights_free(w);
            return -1;
        }
        if (read_bytes(w->fd, &tensor->n_dims, sizeof(*&tensor->n_dims), &pos) != 0) {
            weights_free(w);
            return -1;
        }
        if (tensor->n_dims > 4) {
            fprintf(
                stderr,
                "Tensor has too many dimensions: %s\n",
                tensor->name
            );
            weights_free(w);
            return -1;
        }
        for (uint32_t d = 0;
             d < tensor->n_dims;
             d++) {
            if (read_bytes(w->fd, &tensor->dims[d], sizeof(*&tensor->dims[d]), &pos) != 0) {
                weights_free(w);
                return -1;
            }
        }
        if (read_bytes(w->fd, &tensor->type, sizeof(*&tensor->type), &pos) != 0) {
            weights_free(w);
            return -1;
        }
        if (read_bytes(w->fd, &tensor->offset, sizeof(*&tensor->offset), &pos) != 0) {
            weights_free(w);
            return -1;
        }
        tensor->elements =
            tensor_element_count(
                tensor->n_dims,
                tensor->dims
            );
        if (tensor->elements == 0) {
            weights_free(w);
            return -1;
        }
        tensor->size =
            tensor_size_bytes(
                tensor->type,
                tensor->elements
            );
        if (tensor->size == 0) {
            fprintf(
                stderr,
                "Unsupported tensor type %u: %s\n",
                tensor->type,
                tensor->name
            );
            weights_free(w);
            return -1;
        }
    }
    /* Tensor data section */
    uint64_t aligned_pos =
        (pos + w->alignment - 1) /
        w->alignment *
        w->alignment;
    w->tensor_data_offset =
        aligned_pos;
    /* Expert locations */
    for (uint64_t i = 0;
         i < tensor_count;
         i++) {
        TensorInfo *tensor =
            &w->tensors[i];
        if (!is_expert_tensor(tensor->name))
            continue;
        if (register_expert_tensor(
                w,
                tensor) != 0) {
            fprintf(
                stderr,
                "Failed registering expert tensor: %s\n",
                tensor->name
            );
            weights_free(w);
            return -1;
        }
    }
    /* Resident tensors */
    for (uint64_t i = 0;
         i < tensor_count;
         i++) {
        TensorInfo *tensor =
            &w->tensors[i];
        if (is_expert_tensor(tensor->name))
            continue;
        if (load_tensor(w, tensor) != 0) {
            fprintf(
                stderr,
                "Failed loading resident tensor: %s\n",
                tensor->name
            );
            weights_free(w);
            return -1;
        }
}
    /* Resident tensors were read through the page cache: drop that copy (they are mlocked in
     * our own buffers) and disable readahead for the random buffered fallback reads. */
    (void)posix_fadvise(w->fd, 0, 0, POSIX_FADV_DONTNEED);
    (void)posix_fadvise(w->fd, 0, 0, POSIX_FADV_RANDOM);
    /* Expert caches */
    w->cache = calloc(
        NUM_LAYERS,
        sizeof(LayerCache)
    );
    if (!w->cache) {
        weights_free(w);
        return -1;
    }
    for (int layer = 0;
         layer < NUM_LAYERS;
         layer++) {
        LayerCache *cache =
            &w->cache[layer];
        cache->capacity =
            max_experts_per_layer;
        cache->slots = calloc(
            (size_t)cache->capacity,
            sizeof(CachedExpert)
        );
        cache->slot_by_expert = malloc(
            NUM_EXPERTS * sizeof(int)
        );
        if (!cache->slots ||
            !cache->slot_by_expert) {
            weights_free(w);
            return -1;
        }
        for (int expert = 0;
             expert < NUM_EXPERTS;
             expert++) {
            cache->slot_by_expert[expert] = -1;
        }
        for (int slot = 0;
             slot < cache->capacity;
             slot++) {
            cache->slots[slot].expert_id = -1;
        }
    }
    return 0;
}
TensorInfo *weights_find_tensor(
    Weights *w,
    const char *name
)
{
    if (!w || !name)
        return NULL;
    for (uint64_t i = 0;
         i < w->tensor_count;
         i++) {
        if (strcmp(
                w->tensors[i].name,
                name) == 0) {
            return &w->tensors[i];
        }
    }
    return NULL;
}
void *weights_tensor_data(
    Weights *w,
    const char *name
)
{
    TensorInfo *tensor =
        weights_find_tensor(w, name);
    if (!tensor)
        return NULL;
    return tensor->data;
}
CachedExpert *weights_get_expert(
    Weights *w,
    int layer,
    int expert
)
{
    if (!w)
        return NULL;
    if (layer < 0 ||
        layer >= NUM_LAYERS)
        return NULL;
    if (expert < 0 ||
        expert >= NUM_EXPERTS)
        return NULL;
    if (!w->cache)
        return NULL;
    LayerCache *cache =
        &w->cache[layer];
    /* Cache hit */
    int slot_index =
        cache->slot_by_expert[expert];
    if (slot_index >= 0) {
        CachedExpert *slot =
            &cache->slots[slot_index];
        slot->last_used = ++w->clock;
        slot->use_count++;
        return slot;
    }
    /* Cache miss */
    slot_index =
        find_cache_slot(cache);
    CachedExpert *slot =
        &cache->slots[slot_index];
    /* Evict old expert (buffers are kept and overwritten in place) */
    if (slot->expert_id >= 0)
        release_slot_entry(cache, slot);
    /* Load requested expert */
    if (load_expert_into_slot(
            w,
            layer,
            expert,
            slot_index) != 0) {
        return NULL;
    }
    return slot;
}
/*
 * Make a set of experts resident with all missing reads issued in parallel
 * (3 preads per missing expert). Hits are touched first so they cannot be
 * picked as victims. If the set cannot fit in the cache this is a no-op and
 * callers fall back to weights_get_expert() one at a time.
 * Returns 0 on success (or no-op), -1 if a load failed.
 */
#define PREFETCH_MAX 32
int weights_prefetch_experts(
    Weights *w,
    int layer,
    const int *expert_ids,
    int count
)
{
    if (!w || !w->cache || !expert_ids ||
        layer < 0 || layer >= NUM_LAYERS ||
        count <= 0 || count > PREFETCH_MAX)
        return -1;
    LayerCache *cache = &w->cache[layer];
    if (count > cache->capacity)
        return 0;
    for (int i = 0; i < count; i++) {
        int e = expert_ids[i];
        if (e < 0 || e >= NUM_EXPERTS)
            continue;
        int si = cache->slot_by_expert[e];
        if (si >= 0) {
            CachedExpert *slot = &cache->slots[si];
            slot->last_used = ++w->clock;
            slot->use_count++;
        }
    }
    int miss_e[PREFETCH_MAX], miss_s[PREFETCH_MAX];
    int nm = 0, failed = 0;
    for (int i = 0; i < count; i++) {
        int e = expert_ids[i];
        if (e < 0 || e >= NUM_EXPERTS || cache->slot_by_expert[e] >= 0)
            continue;
        const ExpertLocation *loc = &w->expert_location[layer][e];
        int si = find_cache_slot(cache);
        CachedExpert *slot = &cache->slots[si];
        if (slot->expert_id >= 0)
            release_slot_entry(cache, slot);
        if (ensure_slot_buffers(cache, slot, loc) != 0) {
            free_cached_expert(slot);
            failed = 1;
            continue;
        }
        /* Reserve now so the next victim search skips this slot. */
        claim_slot(w, cache, slot, si, e, loc);
        miss_e[nm] = e;
        miss_s[nm] = si;
        nm++;
    }
    int rc[PREFETCH_MAX * 3] = {0};
    /* Reads are I/O-bound, so use their own thread team: with OMP_NUM_THREADS=1 the
     * default team would issue one read at a time (queue depth 1). */
    const int io_threads = nm * 3 < 12 ? nm * 3 : 12;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(io_threads) if (nm > 0)
    for (int t = 0; t < nm * 3; t++) {
        int m = t / 3;
        CachedExpert *slot = &cache->slots[miss_s[m]];
        const ExpertLocation *loc = &w->expert_location[layer][miss_e[m]];
        void *dst;
        uint64_t size, off;
        switch (t % 3) {
            case 0:  dst = slot->gate; size = loc->gate_size; off = loc->gate_offset; break;
            case 1:  dst = slot->up;   size = loc->up_size;   off = loc->up_offset;   break;
            default: dst = slot->down; size = loc->down_size; off = loc->down_offset; break;
        }
        rc[t] = read_expert_bytes(w, dst, (size_t)size, off);
    }
    for (int m = 0; m < nm; m++) {
        if (rc[3 * m] || rc[3 * m + 1] || rc[3 * m + 2]) {
            CachedExpert *slot = &cache->slots[miss_s[m]];
            cache->slot_by_expert[miss_e[m]] = -1;
            free_cached_expert(slot);
            failed = 1;
        }
    }
    return failed ? -1 : 0;
}

int weights_get_loaded_experts(
    Weights *w,
    int layer,
    int *expert_ids,
    int max_ids
) {
    if (w == NULL ||
        w->cache == NULL ||
        layer < 0 ||
        layer >= NUM_LAYERS) {
        return -1;
    }
    LayerCache *cache = &w->cache[layer];
    int loaded = 0;
    for (int i = 0; i < cache->capacity; i++) {
        CachedExpert *expert = &cache->slots[i];
        // An expert is loaded only when its weights are present.
        if (expert->expert_id < 0 ||
            expert->gate == NULL ||
            expert->up == NULL ||
            expert->down == NULL) {
            continue;
        }
        if (expert_ids != NULL && loaded < max_ids) {
            expert_ids[loaded] = expert->expert_id;
        }
        loaded++;
    }
    return loaded;
}
void weights_free(
    Weights *w
)
{
    if (!w)
        return;
    if (w->tensors) {
        for (uint64_t i = 0;
             i < w->tensor_count;
             i++) {
            free(w->tensors[i].name);
            unlock_free(w->tensors[i].data, w->tensors[i].locked_size);
        }
        free(w->tensors);
    }
    if (w->cache) {
        for (int layer = 0;
             layer < NUM_LAYERS;
             layer++) {
            LayerCache *cache =
                &w->cache[layer];
            if (cache->slab) {
                for (int slot = 0; cache->slots && slot < cache->capacity; slot++) {
                    CachedExpert *cs = &cache->slots[slot];
                    if (cs->in_slab) {
                        kernel_vk_release_host_ptr(cs->gate);
                        kernel_vk_release_host_ptr(cs->up);
                        kernel_vk_release_host_ptr(cs->down);
                    }
                }
                unlock_free(cache->slab, cache->slab_size);
                cache->slab = NULL;
            }
            if (cache->slots) {
                for (int slot = 0;
                     slot < cache->capacity;
                     slot++) {
                    free_cached_expert(
                        &cache->slots[slot]
                    );
                }
                free(cache->slots);
            }
            free(cache->slot_by_expert);
        }
        free(w->cache);
    }
    if (w->fd >= 0)
        close(w->fd);
    if (w->fd_direct >= 0)
        close(w->fd_direct);
    for (int i = 0; i < BOUNCE_SLOTS; i++) {          /* O_DIRECT bounce buffers */
        free(g_bounce[i]);
        g_bounce[i] = NULL;
        g_bounce_cap[i] = 0;
    }
    g_direct_failed = 0;
    /* All imported weight allocations have been released above. */
    kernel_vk_shutdown();
    memset(w, 0, sizeof(*w));
    w->fd = -1;
    w->fd_direct = -1;
}