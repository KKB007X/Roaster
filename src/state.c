#include "state.h"
#include "kvwindow.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define STATE_MAGIC   0x41545352u      /* "RSTA" */
#define STATE_VERSION 2u              /* v2: position is absolute (may exceed the window) */

typedef struct {
    uint32_t magic, version, num_layers, reserved;
    uint64_t context_length, position;
} StateHeader;

static int put(FILE *f, const void *p, size_t n)
{
    return n == 0 || fwrite(p, 1, n, f) == n ? 0 : -1;
}

static int get(FILE *f, void *p, size_t n)
{
    return n == 0 || fread(p, 1, n, f) == n ? 0 : -1;
}

int model_state_save(const Model *m, FILE *f)
{
    if (!m || !f || !m->initialized || !m->kv_cache || !m->ssm_state)
        return -1;
    StateHeader h = { STATE_MAGIC, STATE_VERSION, MODEL_NUM_LAYERS, 0,
                      (uint64_t)m->context_length, (uint64_t)m->position };
    if (put(f, &h, sizeof h))
        return -1;
    for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
        if (model_is_attention_layer(layer)) {
            const KVCache *kv = &m->kv_cache[layer];
            uint64_t len = kv->length;
            size_t bytes = (size_t)len * KV_FLOATS_PER_TOKEN * sizeof(float);
            if (len > kv->capacity || put(f, &len, sizeof len) ||
                put(f, kv->k, bytes) || put(f, kv->v, bytes))
                return -1;
        } else {
            const SSMState *s = &m->ssm_state[layer];
            if (put(f, s->state, s->state_elements * sizeof(float)) ||
                put(f, s->conv_state, s->conv_elements * sizeof(float)))
                return -1;
        }
    }
    return 0;
}

int model_state_load(Model *m, FILE *f)
{
    if (!m || !f || !m->initialized || !m->kv_cache || !m->ssm_state)
        return -1;
    StateHeader h;
    if (get(f, &h, sizeof h) || h.magic != STATE_MAGIC || h.version != STATE_VERSION ||
        h.num_layers != MODEL_NUM_LAYERS)
        return -1;                            /* a window of a different size is fine if the data fits */
    for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
        if (model_is_attention_layer(layer)) {
            KVCache *kv = &m->kv_cache[layer];
            uint64_t len;
            if (get(f, &len, sizeof len) || len > kv->capacity)
                return -1;
            size_t bytes = (size_t)len * KV_FLOATS_PER_TOKEN * sizeof(float);
            if (get(f, kv->k, bytes) || get(f, kv->v, bytes))
                return -1;
            kv->length = (size_t)len;
        } else {
            SSMState *s = &m->ssm_state[layer];
            if (get(f, s->state, s->state_elements * sizeof(float)) ||
                get(f, s->conv_state, s->conv_elements * sizeof(float)))
                return -1;
        }
    }
    /* window + tokens on SSD must add up to the absolute position */
    if ((uint64_t)kvw_len(m) + (uint64_t)kvw_spilled(m) != h.position)
        return -1;
    m->position = (size_t)h.position;
    return 0;
}

size_t model_ram_bytes(const Model *m)
{
    if (!m || !m->initialized)
        return 0;
    size_t total = 0;
    for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
        if (model_is_attention_layer(layer)) {
            total += 2 * m->kv_cache[layer].capacity * KV_FLOATS_PER_TOKEN * sizeof(float);
        } else {
            const SSMState *s = &m->ssm_state[layer];
            total += (s->state_elements + s->conv_elements) * sizeof(float);
        }
    }
    total += 16 * m->context_length * sizeof(float);                         /* attention scores */
    total += (size_t)MODEL_VOCAB_SIZE * (sizeof(float) * 2 + sizeof(int32_t)); /* logits + sampler */
    total += (size_t)1 << 20;                                                /* small buffers */
    return total;
}

/* ------------------------------------------------------------------ snapshot */

struct ModelSnapshot {
    size_t position;
    size_t spilled;         /* tokens on SSD at snapshot time */
    size_t floats;
    float *data;            /* recurrent state + conv history of every recurrent layer */
};

ModelSnapshot *model_snapshot_take(const Model *m)
{
    if (!m || !m->initialized)
        return NULL;
    size_t floats = 0;
    for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++)
        if (!model_is_attention_layer(layer))
            floats += m->ssm_state[layer].state_elements + m->ssm_state[layer].conv_elements;
    ModelSnapshot *s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->data = malloc((floats ? floats : 1) * sizeof(float));
    if (!s->data) {
        free(s);
        return NULL;
    }
    s->floats = floats;
    s->position = m->position;
    s->spilled = kvw_spilled(m);
    float *w = s->data;
    for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
        if (!model_is_attention_layer(layer)) {
            const SSMState *st = &m->ssm_state[layer];
            memcpy(w, st->state, st->state_elements * sizeof(float));
            w += st->state_elements;
            memcpy(w, st->conv_state, st->conv_elements * sizeof(float));
            w += st->conv_elements;
        }
    }
    return s;
}

int model_snapshot_restore(Model *m, const ModelSnapshot *s)
{
    if (!m || !s || !m->initialized)
        return -1;
    const float *r = s->data;
    for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
        if (!model_is_attention_layer(layer)) {
            SSMState *st = &m->ssm_state[layer];
            memcpy(st->state, r, st->state_elements * sizeof(float));
            r += st->state_elements;
            memcpy(st->conv_state, r, st->conv_elements * sizeof(float));
            r += st->conv_elements;
        }
    }
    /* KV: drop the newer tokens and bring back whatever the request pushed out to SSD. */
    return kvw_rollback(m, s->position, s->spilled);
}

void model_snapshot_free(ModelSnapshot *s)
{
    if (!s)
        return;
    free(s->data);
    free(s);
}
