#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stdint.h>
#include <stddef.h>

#define NUM_LAYERS  40
#define NUM_EXPERTS 256

typedef struct {
    char *name;

    uint32_t type;
    uint32_t n_dims;
    uint64_t dims[4];

    uint64_t elements;
    uint64_t offset;
    uint64_t size;

    void *data;
    uint64_t locked_size;
} TensorInfo;

typedef struct {
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;

    uint64_t gate_size;
    uint64_t up_size;
    uint64_t down_size;

    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
} ExpertLocation;

typedef struct {
    int expert_id;

    void *gate;
    void *up;
    void *down;

    uint64_t last_used;
    uint64_t use_count;

    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    uint64_t gate_lock_size, up_lock_size, down_lock_size;
    int in_slab;   /* buffers are carved from LayerCache.slab (not individually allocated) */
} CachedExpert;

typedef struct {
    CachedExpert *slots;
    int *slot_by_expert;
    int capacity;
    /* One allocation holding every slot of this layer. */
    void *slab;
    uint64_t slab_size;
    uint64_t slab_stride;
    int slab_driver;      /* slab lives in Vulkan driver memory (no mlock, never freed here) */
    int slab_tried;       /* a slab allocation was already attempted (do not retry after a failure) */
} LayerCache;

typedef struct {
    int fd;
    int fd_direct;   /* O_DIRECT descriptor for expert reads (-1 = unavailable/disabled) */

    uint64_t tensor_data_offset;
    uint64_t alignment;

    TensorInfo *tensors;
    uint64_t tensor_count;

    ExpertLocation expert_location[NUM_LAYERS][NUM_EXPERTS];

    LayerCache *cache;
    int max_experts_per_layer;

    uint64_t clock;
} Weights;

int weights_load(
    Weights *w,
    const char *path,
    int max_experts_per_layer
);

TensorInfo *weights_find_tensor(
    Weights *w,
    const char *name
);

void *weights_tensor_data(
    Weights *w,
    const char *name
);

CachedExpert *weights_get_expert(
    Weights *w,
    int layer,
    int expert
);

/* Load all missing experts of one layer's routed set in parallel. */
int weights_prefetch_experts(
    Weights *w,
    int layer,
    const int *expert_ids,
    int count
);

int weights_get_loaded_experts(
    Weights *w,
    int layer,
    int *expert_ids,
    int max_ids
);

void weights_free(Weights *w);

#endif