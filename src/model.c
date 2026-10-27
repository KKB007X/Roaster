#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "model.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>

#include "kernels.h"

/* ============================================================
 * Internal helpers
 * ============================================================ */

static bool checked_mul_size(
    size_t a,
    size_t b,
    size_t *result
)
{
    if (a != 0 && b > SIZE_MAX / a)
        return false;
    *result = a * b;
    return true;
}

/* Persistent per-slot state uses its own page-aligned mapping so mlock/munlock
 * never affects unrelated malloc allocations sharing a page. */

typedef struct {
    size_t mapping_size;
    size_t rounded_size;
    int locked;
} LockedBufferHeader;

static size_t locked_buffer_page_size(void)
{
    long value = sysconf(_SC_PAGESIZE);
    return value > 0 ? (size_t)value : 4096u;
}

static void *locked_calloc_checked(size_t count, size_t size)
{
    size_t bytes;
    if (!checked_mul_size(count, size, &bytes) || bytes == 0)
        return NULL;
    size_t page = locked_buffer_page_size();
    if (bytes > SIZE_MAX - (page - 1))
        return NULL;
    size_t rounded = ((bytes + page - 1) / page) * page;
    if (rounded > SIZE_MAX - page)
        return NULL;
    size_t mapping_size = page + rounded;
    void *base = mmap(NULL, mapping_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED)
        return NULL;
    LockedBufferHeader *header = (LockedBufferHeader *)base;
    header->mapping_size = mapping_size;
    header->rounded_size = rounded;
    header->locked = 0;
    void *data = (char *)base + page;
    /* mmap is zero-filled, matching calloc semantics. */
    if (mlock(data, rounded) == 0) {
        header->locked = 1;
    } else {
        fprintf(stderr,
                "WARNING: mlock failed for persistent state (%zu bytes): %s\n",
                rounded, strerror(errno));
    }
    return data;
}

static void locked_buffer_free(void *data)
{
    if (!data)
        return;
    size_t page = locked_buffer_page_size();
    void *base = (char *)data - page;
    LockedBufferHeader *header = (LockedBufferHeader *)base;
    size_t mapping_size = header->mapping_size;
    size_t rounded_size = header->rounded_size;
    int locked = header->locked;
    if (locked)
        (void)munlock(data, rounded_size);
    (void)munmap(base, mapping_size);
}

static void *calloc_checked(size_t count, size_t size)
{
    size_t bytes;
    if (!checked_mul_size(count, size, &bytes))
        return NULL;
    return calloc(1, bytes);
}

/* ============================================================
 * Layer type
 * ============================================================ */

static int tensor_matvec_f32_or_quant(const TensorInfo *t, const void *w, const float *x, float *y, size_t rows, size_t cols)
{
    if (!t || !w || !x || !y) return -1;
    if (t->type == 0) { kernel_matvec_f32((const float *)w, x, y, rows, cols); return 0; }
    if (t->type == 12) { kernel_matvec_q4_k(w,x,y,rows,cols); return 0; }
    if (t->type == 14) { kernel_matvec_q6_k(w,x,y,rows,cols); return 0; }
    fprintf(stderr, "ERROR: unsupported tensor type %u\n", t->type);
    return -1;
}

/* ============================================================
 * Tensor reference cache
 *
 * Every tensor the forward pass needs is looked up once at init instead of
 * a snprintf + linear strcmp scan over the whole GGUF tensor table (and a
 * second scan inside weights_tensor_data) on every use of every token.
 * ============================================================ */

typedef struct {
    const TensorInfo *t;
    const void *d;
} TRef;

typedef struct {
    const float *attn_norm;
    const float *post_norm;
    /* MoE */
    TRef router, sh_gate_inp, sh_gate, sh_up, sh_down;
    /* recurrent (Gated DeltaNet) layers */
    TRef qkv, gate, alpha, beta, ssm_out;
    const float *conv, *dt, *a, *ssm_norm;
    /* full-attention layers */
    TRef q, k, v, o;
    const float *qnorm, *knorm;
} LayerRefs;

typedef struct {
    TRef embd, out;
    const float *out_norm;
    LayerRefs layer[MODEL_NUM_LAYERS];
} ModelRefs;

static TRef tref(Weights *w, int layer, const char *suffix)
{
    char name[128];
    if (layer >= 0)
        snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    else
        snprintf(name, sizeof(name), "%s", suffix);
    TRef r;
    r.t = weights_find_tensor(w, name);
    r.d = r.t ? r.t->data : NULL;
    return r;
}

static const float *fref(Weights *w, int layer, const char *suffix)
{
    return (const float *)tref(w, layer, suffix).d;
}

static int mv(const TRef *r, const float *x, float *y, size_t rows, size_t cols)
{
    if (!r->t || !r->d) {
        fprintf(stderr, "ERROR: required tensor missing from model\n");
        return -1;
    }
    return tensor_matvec_f32_or_quant(r->t, r->d, x, y, rows, cols);
}

/* Queue one matvec as a batch job when the tensor is Q4_K/Q6_K, else run it now (e.g. f32). */
static int add_ref_job(KernelMatvecJob *jobs, int *nj, const TRef *r,
                       const float *x, float *y, size_t rows, size_t cols)
{
    if (!r->t || !r->d) {
        fprintf(stderr, "ERROR: required tensor missing from model\n");
        return -1;
    }
    if (r->t->type == 12 || r->t->type == 14) {
        jobs[(*nj)++] = (KernelMatvecJob){ r->t->type == 14, r->d, x, y, rows, cols };
        return 0;
    }
    return tensor_matvec_f32_or_quant(r->t, r->d, x, y, rows, cols);
}

/* Several matvecs sharing the same input vector: one GPU round trip instead of n. */
static int mv_batch(const TRef *const *refs, float *const *outs, const size_t *rows,
                    int n, const float *x, size_t cols)
{
    KernelMatvecJob jobs[8];
    int nj = 0;
    for (int i = 0; i < n; i++)
        if (add_ref_job(jobs, &nj, refs[i], x, outs[i], rows[i], cols) != 0)
            return -1;
    if (nj)
        kernel_matvec_quant_batch(jobs, nj);
    return 0;
}

static int build_refs(Model *m)
{
    ModelRefs *R = calloc(1, sizeof(*R));
    if (!R)
        return -1;
    Weights *w = m->weights;
    R->embd = tref(w, -1, "token_embd.weight");
    R->out = tref(w, -1, "output.weight");
    R->out_norm = fref(w, -1, "output_norm.weight");
    for (int l = 0; l < MODEL_NUM_LAYERS; l++) {
        LayerRefs *L = &R->layer[l];
        L->attn_norm = fref(w, l, "attn_norm.weight");
        L->post_norm = fref(w, l, "post_attention_norm.weight");
        L->router = tref(w, l, "ffn_gate_inp.weight");
        L->sh_gate_inp = tref(w, l, "ffn_gate_inp_shexp.weight");
        L->sh_gate = tref(w, l, "ffn_gate_shexp.weight");
        L->sh_up = tref(w, l, "ffn_up_shexp.weight");
        L->sh_down = tref(w, l, "ffn_down_shexp.weight");
        if (model_is_attention_layer(l)) {
            L->q = tref(w, l, "attn_q.weight");
            L->k = tref(w, l, "attn_k.weight");
            L->v = tref(w, l, "attn_v.weight");
            L->o = tref(w, l, "attn_output.weight");
            L->qnorm = fref(w, l, "attn_q_norm.weight");
            L->knorm = fref(w, l, "attn_k_norm.weight");
        } else {
            L->qkv = tref(w, l, "attn_qkv.weight");
            L->gate = tref(w, l, "attn_gate.weight");
            L->alpha = tref(w, l, "ssm_alpha.weight");
            L->beta = tref(w, l, "ssm_beta.weight");
            L->ssm_out = tref(w, l, "ssm_out.weight");
            L->conv = fref(w, l, "ssm_conv1d.weight");
            L->dt = fref(w, l, "ssm_dt.bias");
            L->a = fref(w, l, "ssm_a");
            L->ssm_norm = fref(w, l, "ssm_norm.weight");
        }
    }
    m->refs = R;
    return 0;
}

/*
 * Gated DeltaNet step, one pass over the 128x128 state per head.
 *
 * Reference algorithm (per value head, S is [key_dim r][value_dim c]):
 *   S      <- d * S
 *   pred   =  S^T k
 *   delta  =  (v - pred) * beta
 *   S      <- S + k (x) delta
 *   out    =  scale * S^T q
 *
 * Substituting S_old, A = S_old^T k, B = S_old^T q, kq = k.q:
 *   pred   = d*A
 *   delta  = (v - d*A) * beta
 *   S_new  = d*S_old + k (x) delta
 *   out    = scale * (d*B + kq*delta)
 * so A and B come from ONE row-major read of the state, then S_new is
 * written in one more pass (the state is 64 KB per head and stays in L2).
 * The original did four passes and one of them walked columns with a
 * 512-byte stride. Heads are independent and run in parallel.
 */
static void recurrent_delta_step(
    SSMState *ssm,
    const float *q,
    const float *k,
    const float *v,
    const float *decay,
    const float *beta,
    float *out)
{
    const float scale = 1.0f / sqrtf((float)MODEL_LINEAR_HEAD_DIM);
    /* Not parallel: 30 layers x ~0.1 ms of work is less than the cost of waking an OpenMP team
     * (measured: 4 threads made this phase slower). */
    for (int vh = 0; vh < MODEL_LINEAR_V_HEADS; vh++) {
        const int kh = vh % MODEL_LINEAR_K_HEADS;
        const float *qh  = q + kh * MODEL_LINEAR_HEAD_DIM;
        const float *khv = k + kh * MODEL_LINEAR_HEAD_DIM;
        const float *vhv = v + vh * MODEL_LINEAR_HEAD_DIM;
        float *st = ssm->state +
            (size_t)vh * MODEL_LINEAR_HEAD_DIM * MODEL_LINEAR_HEAD_DIM;
        const float d = expf(decay[vh]);
        const float b = beta[vh];
        float A[MODEL_LINEAR_HEAD_DIM];
        float B[MODEL_LINEAR_HEAD_DIM];
        float delta[MODEL_LINEAR_HEAD_DIM];
        float kq = 0.0f;
        for (int c = 0; c < MODEL_LINEAR_HEAD_DIM; c++) {
            A[c] = 0.0f;
            B[c] = 0.0f;
        }
        for (int r = 0; r < MODEL_LINEAR_HEAD_DIM; r++) {
            const float kr = khv[r];
            const float qr = qh[r];
            const float *row = st + r * MODEL_LINEAR_HEAD_DIM;
            kq += kr * qr;
            for (int c = 0; c < MODEL_LINEAR_HEAD_DIM; c++) {
                A[c] += row[c] * kr;
                B[c] += row[c] * qr;
            }
        }
        for (int c = 0; c < MODEL_LINEAR_HEAD_DIM; c++) {
            delta[c] = (vhv[c] - d * A[c]) * b;
            out[vh * MODEL_LINEAR_HEAD_DIM + c] =
                (d * B[c] + kq * delta[c]) * scale;
        }
        for (int r = 0; r < MODEL_LINEAR_HEAD_DIM; r++) {
            const float kr = khv[r];
            float *row = st + r * MODEL_LINEAR_HEAD_DIM;
            for (int c = 0; c < MODEL_LINEAR_HEAD_DIM; c++)
                row[c] = d * row[c] + kr * delta[c];
        }
    }
}

bool model_is_attention_layer(int layer)
{
    return layer >= 0 &&
           layer < MODEL_NUM_LAYERS &&
           ((layer + 1) % MODEL_FULL_ATTN_INTERVAL) == 0;
}

bool model_is_recurrent_layer(int layer)
{
    if (layer < 0 || layer >= MODEL_NUM_LAYERS)
        return false;
    return !model_is_attention_layer(layer);
}

/* ============================================================
 * KV cache
 * ============================================================ */

static void kv_cache_free(KVCache *kv)
{
    if (!kv)
        return;
    locked_buffer_free(kv->k);
    locked_buffer_free(kv->v);
    kv->k = NULL;
    kv->v = NULL;
    kv->capacity = 0;
    kv->length = 0;
}

static int kv_cache_init(
    KVCache *kv,
    size_t context_length
)
{
    size_t elements;
    memset(kv, 0, sizeof(*kv));
    if (!checked_mul_size(
            2,
            context_length,
            &elements))
    {
        return -1;
    }
    if (!checked_mul_size(
            elements,
            256,
            &elements))
    {
        return -1;
    }
    kv->capacity = context_length;
    kv->k = locked_calloc_checked(elements, sizeof(float));
    if (!kv->k)
        return -1;
    kv->v = locked_calloc_checked(elements, sizeof(float));
    if (!kv->v) {
        locked_buffer_free(kv->k);
        kv->k = NULL;
        return -1;
    }
    kv->length = 0;
    return 0;
}

/* ============================================================
 * SSM state
 * ============================================================ */

static void ssm_state_free(SSMState *state)
{
    if (!state)
        return;
    locked_buffer_free(state->state);
    locked_buffer_free(state->conv_state);
    state->state = NULL;
    state->conv_state = NULL;
    state->state_elements = 0;
    state->conv_elements = 0;
}

static int ssm_state_init(SSMState *state)
{
    size_t elements;
    memset(state, 0, sizeof(*state));
    /*
     * Gated DeltaNet state: [value_head][key_dim][value_dim]
     * 32 x 128 x 128 floats.
     */
    if (!checked_mul_size(
            MODEL_LINEAR_V_HEADS,
            MODEL_LINEAR_HEAD_DIM,
            &elements))
    {
        return -1;
    }
    if (!checked_mul_size(
            elements,
            MODEL_LINEAR_HEAD_DIM,
            &elements))
    {
        return -1;
    }
    state->state = locked_calloc_checked(
        elements,
        sizeof(float)
    );
    if (!state->state)
        return -1;
    state->state_elements = elements;
    /*
     * Causal convolution history.
     *
     * kernel_size = 4
     * history      = 3 previous tokens
     *
     * Each token has MODEL_LINEAR_QKV_SIZE values.
     */
    if (!checked_mul_size(
            MODEL_SSM_CONV_KERNEL - 1,
            MODEL_LINEAR_QKV_SIZE,
            &elements))
    {
        ssm_state_free(state);
        return -1;
    }
    state->conv_state = locked_calloc_checked(
        elements,
        sizeof(float)
    );
    if (!state->conv_state) {
        ssm_state_free(state);
        return -1;
    }
    state->conv_elements = elements;
    return 0;
}

/* ============================================================
 * Model buffers
 * ============================================================ */

static int allocate_model_buffers(Model *model)
{
    model->hidden = calloc_checked(
        MODEL_HIDDEN_SIZE,
        sizeof(float)
    );
    if (!model->hidden)
        return -1;
    model->residual = calloc_checked(
        MODEL_HIDDEN_SIZE,
        sizeof(float)
    );
    if (!model->residual)
        return -1;
    model->normed = calloc_checked(
        MODEL_HIDDEN_SIZE,
        sizeof(float)
    );
    if (!model->normed)
        return -1;
    model->linear_q = calloc_checked(
        MODEL_LINEAR_Q_SIZE,
        sizeof(float)
    );
    if (!model->linear_q) return -1;
    model->linear_k = calloc_checked(
        MODEL_LINEAR_K_SIZE,
        sizeof(float)
    );
    if (!model->linear_k) return -1;
    model->linear_v = calloc_checked(
        MODEL_LINEAR_V_SIZE,
        sizeof(float)
    );
    if (!model->linear_v) return -1;
    model->linear_gate = calloc_checked(
        MODEL_LINEAR_GATE_SIZE,
        sizeof(float)
    );
    if (!model->linear_gate) return -1;
    model->ffn_output = calloc_checked(
        MODEL_HIDDEN_SIZE,
        sizeof(float)
    );
    if (!model->ffn_output)
        return -1;
    model->router_logits = calloc_checked(
        MODEL_NUM_EXPERTS,
        sizeof(float)
    );
    if (!model->router_logits)
        return -1;
    model->expert_ids = calloc_checked(
        MODEL_TOP_K,
        sizeof(int)
    );
    if (!model->expert_ids)
        return -1;
    model->expert_weights = calloc_checked(
        MODEL_TOP_K,
        sizeof(float)
    );
    if (!model->expert_weights)
        return -1;
    model->logits = calloc_checked(
        MODEL_VOCAB_SIZE,
        sizeof(float)
    );
    if (!model->logits)
        return -1;
    model->sampling_ids = calloc_checked(
        MODEL_VOCAB_SIZE,
        sizeof(int32_t)
    );
    if (!model->sampling_ids)
        return -1;
    model->sampling_logits = calloc_checked(
        MODEL_VOCAB_SIZE,
        sizeof(float)
    );
    if (!model->sampling_logits)
        return -1;
    return 0;
}

/* ============================================================
 * Embedding
 * ============================================================ */

static int model_embedding(
    Model *model,
    int32_t token,
    float *output
)
{
    if (!model || !output || !model->refs)
        return MODEL_FORWARD_ERROR;
    if (token < 0 || token >= MODEL_VOCAB_SIZE)
        return MODEL_FORWARD_ERROR;
    const ModelRefs *R = (const ModelRefs *)model->refs;
    if (!R->embd.t || !R->embd.d)
        return MODEL_FORWARD_ERROR;
    if (R->embd.t->type != 12) {
        fprintf(
            stderr,
            "ERROR: token_embd.weight has unexpected type %u\n",
            R->embd.t->type
        );
        return MODEL_FORWARD_ERROR;
    }
    kernel_embedding_row_q4_k(
        R->embd.d,
        (size_t)token,
        output,
        MODEL_HIDDEN_SIZE
    );
    return MODEL_FORWARD_OK;
}

/* ============================================================
 * Model initialization
 * ============================================================ */

int model_init(
    Model *model,
    Weights *weights,
    size_t context_length,
    float temperature,
    float top_p,
    int top_k
)
{
    if (!model || !weights)
        return -1;
    if (context_length == 0)
        return -1;
    if (!isfinite(temperature) || temperature < 0.0f)
        return -1;
    if (!isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f)
        return -1;
    if (top_k <= 0 || top_k > MODEL_VOCAB_SIZE)
        return -1;
    memset(model, 0, sizeof(*model));
    model->weights = weights;
    model->context_length = context_length;
    model->temperature = temperature;
    model->top_p = top_p;
    model->top_k = top_k;
    /* Colibri-style routing controls are opt-in. The cache records routes;
       expert substitution is intentionally not enabled without a verified
       routing policy implementation. */
    model->cache_route = false;
    model->route_j = 2;
    model->route_m = 12;
    model->route_alpha = 1.0f;
    model->route_p = 0.0f;
    model->position = 0;
    model->rng_state = (((uint64_t)time(NULL)) << 32) ^
                       (uint64_t)(uintptr_t)model ^
                       UINT64_C(0x9e3779b97f4a7c15);
    if (model->rng_state == 0)
        model->rng_state = UINT64_C(0x2545f4914f6cdd1d);
    model->qkv = calloc_checked(
        MODEL_LINEAR_QKV_SIZE,
        sizeof(float)
    );
    if (!model->qkv)
        goto fail;
    model->recurrent_output = calloc_checked(
        MODEL_LINEAR_QKV_SIZE,
        sizeof(float)
    );
    if (!model->recurrent_output)
        goto fail;
    model->attn_q = calloc_checked(
        MODEL_ATTN_Q_SIZE,
        sizeof(float)
    );
    if (!model->attn_q)
        goto fail;
    model->attn_k = calloc_checked(
        MODEL_ATTN_K_SIZE,
        sizeof(float)
    );
    if (!model->attn_k)
        goto fail;
    model->attn_v = calloc_checked(
        MODEL_ATTN_V_SIZE,
        sizeof(float)
    );
    if (!model->attn_v)
        goto fail;
    model->kv_cache = calloc(
        MODEL_NUM_LAYERS,
        sizeof(KVCache)
    );
    if (!model->kv_cache)
        goto fail;
    for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
        if (!model_is_attention_layer(layer))
            continue;
        if (kv_cache_init(
                &model->kv_cache[layer],
                context_length) != 0)
        {
            goto fail;
        }
    }
    model->ssm_state = calloc(
        MODEL_NUM_LAYERS,
        sizeof(SSMState)
    );
    if (!model->ssm_state)
        goto fail;
    for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
        if (!model_is_recurrent_layer(layer))
            continue;
        if (ssm_state_init(
                &model->ssm_state[layer]) != 0)
        {
            goto fail;
        }
    }
    if (allocate_model_buffers(model) != 0)
        goto fail;
    {
        size_t scratch;
        if (!checked_mul_size(16, context_length, &scratch))
            goto fail;
        model->attn_scratch = calloc_checked(scratch, sizeof(float));
        if (!model->attn_scratch)
            goto fail;
    }
    if (build_refs(model) != 0)
        goto fail;
    model->initialized = true;
    return 0;
fail:
    model_free(model);
    return -1;
}

/* ============================================================
 * Reset
 * ============================================================ */

void model_reset(Model *model)
{
    if (!model || !model->initialized)
        return;
    model->position = 0;
    /* KV contents are never read past kv->length, so only the length is
     * reset (zeroing the whole cache was O(context) per request). */
    if (model->kv_cache) {
        for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
            if (model_is_attention_layer(layer))
                model->kv_cache[layer].length = 0;
        }
    }
    if (model->ssm_state) {
        for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
            if (!model_is_recurrent_layer(layer))
                continue;
            SSMState *state = &model->ssm_state[layer];
            if (state->state) {
                memset(
                    state->state,
                    0,
                    state->state_elements * sizeof(float)
                );
            }
            if (state->conv_state) {
                memset(
                    state->conv_state,
                    0,
                    state->conv_elements * sizeof(float)
                );
            }
        }
    }
}

/* ============================================================
 * Full attention helpers
 * ============================================================ */

static void apply_rope_64(float *x, size_t heads, size_t head_dim, size_t position)
{
    const float theta = 10000000.0f;
    const size_t rotary_dim = 64;
    const size_t half = rotary_dim / 2;
    /* cos/sin depend only on (position, j): compute once, not once per head. */
    float cs[32], sn[32];
    for (size_t j = 0; j < half; j++) {
        float inv_freq = powf(theta, -2.0f * (float)j / (float)rotary_dim);
        float angle = (float)position * inv_freq;
        cs[j] = cosf(angle);
        sn[j] = sinf(angle);
    }
    for (size_t h = 0; h < heads; h++) {
        float *v = x + h * head_dim;
        for (size_t j = 0; j < half; j++) {
            float a = v[j];
            float b = v[j + half];
            v[j] = a * cs[j] - b * sn[j];
            v[j + half] = b * cs[j] + a * sn[j];
        }
    }
}

static int full_attention(Model *m, int layer)
{
    const LayerRefs *R = &((const ModelRefs *)m->refs)->layer[layer];
    KVCache *kv = &m->kv_cache[layer];
    if (m->position >= kv->capacity || !m->attn_scratch)
        return -1;
    /*
     * Q projection. Physical layout: [Q0 | G0 | Q1 | G1 | ... | Q15 | G15],
     * each Q/G block 256 values.
     */
    float q_gate[MODEL_ATTN_Q_SIZE];
    {
        const TRef *refs[3] = { &R->q, &R->k, &R->v };
        float *outs[3] = { q_gate, m->attn_k, m->attn_v };
        const size_t rws[3] = { MODEL_ATTN_Q_SIZE, MODEL_ATTN_K_SIZE, MODEL_ATTN_V_SIZE };
        if (mv_batch(refs, outs, rws, 3, m->normed, MODEL_HIDDEN_SIZE) != 0)
            return -1;
    }
    float *gate = m->attn_q + MODEL_ATTN_Q_SIZE / 2;
    for (int h = 0; h < MODEL_QK_HEAD_COUNT; h++) {
        memcpy(m->attn_q + h * 256, q_gate + h * 512, 256 * sizeof(float));
        memcpy(gate + h * 256, q_gate + h * 512 + 256, 256 * sizeof(float));
    }
    if (!R->qnorm || !R->knorm)
        return -1;
    for (int h = 0; h < MODEL_QK_HEAD_COUNT; h++)
        kernel_rmsnorm(m->attn_q + h * 256, m->attn_q + h * 256, R->qnorm, 256, 1e-6f);
    for (int h = 0; h < 2; h++)
        kernel_rmsnorm(m->attn_k + h * 256, m->attn_k + h * 256, R->knorm, 256, 1e-6f);
    /* Partial RoPE: only the first 64 dims are rotated. */
    apply_rope_64(m->attn_q, 16, 256, m->position);
    apply_rope_64(m->attn_k, 2, 256, m->position);
    memcpy(kv->k + m->position * 512, m->attn_k, 512 * sizeof(float));
    memcpy(kv->v + m->position * 512, m->attn_v, 512 * sizeof(float));
    kv->length = m->position + 1;
    /*
     * Causal GQA: 16 Q heads, 2 KV heads (8 Q heads per KV head).
     * Heads are independent; each gets its own score row in persistent
     * scratch (no per-call calloc/free) and softmax is done in place.
     */
    const size_t len = kv->length;
    const size_t stride = kv->capacity;
    const float inv_sqrt = 1.0f / sqrtf(256.0f);
    float attn_out[4096];
    memset(attn_out, 0, sizeof(attn_out));
    #pragma omp parallel for schedule(static) if (len >= 256)
    for (int qh = 0; qh < 16; qh++) {
        const int kvh = qh / 8;
        const float *q = m->attn_q + qh * 256;
        float *scores = m->attn_scratch + (size_t)qh * stride;
        float max_score = -INFINITY;
        for (size_t t = 0; t < len; t++) {
            const float *k = kv->k + t * 512 + kvh * 256;
            float s = kernel_dot_f32(q, k, 256) * inv_sqrt;
            scores[t] = s;
            if (s > max_score)
                max_score = s;
        }
        float sum = 0.0f;
        for (size_t t = 0; t < len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        float *out = attn_out + qh * 256;
        for (size_t t = 0; t < len; t++) {
            const float *v = kv->v + t * 512 + kvh * 256;
            const float p = scores[t] * inv;
            for (int d = 0; d < 256; d++)
                out[d] += p * v[d];
        }
    }
    /* Output gate (not normalized or RoPE'd). */
    for (int i = 0; i < 4096; i++) {
        float g = gate[i];
        attn_out[i] *= 1.0f / (1.0f + expf(-g));
    }
    return mv(&R->o, attn_out, m->residual, MODEL_HIDDEN_SIZE, 4096);
}

/* ============================================================
 * MoE helpers
 * ============================================================ */

static void select_top8(const float *logits, int *ids, float *weights)
{
    for (int i = 0; i < MODEL_TOP_K; i++) {
        ids[i] = -1;
        weights[i] = -INFINITY;
    }
    for (int e = 0; e < MODEL_NUM_EXPERTS; e++) {
        float v = logits[e];
        int p = MODEL_TOP_K;
        for (int i = 0; i < MODEL_TOP_K; i++) {
            if (v > weights[i]) { p = i; break; }
        }
        if (p == MODEL_TOP_K) continue;
        for (int j = MODEL_TOP_K - 1; j > p; j--) {
            weights[j] = weights[j - 1];
            ids[j] = ids[j - 1];
        }
        weights[p] = v;
        ids[p] = e;
    }
    float maxv = weights[0];
    float sum = 0.0f;
    for (int i = 0; i < MODEL_TOP_K; i++) {
        weights[i] = expf(weights[i] - maxv);
        sum += weights[i];
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = 0; i < MODEL_TOP_K; i++) weights[i] *= inv;
}

static int expert_is_resident(const Weights *weights, int layer, int expert_id)
{
    if (!weights || !weights->cache || layer < 0 ||
        layer >= MODEL_NUM_LAYERS || expert_id < 0 ||
        expert_id >= MODEL_NUM_EXPERTS)
        return 0;
    const LayerCache *cache = &weights->cache[layer];
    if (!cache->slots || cache->capacity <= 0)
        return 0;
    for (int i = 0; i < cache->capacity; ++i) {
        const CachedExpert *entry = &cache->slots[i];
        if (entry->expert_id == expert_id && entry->gate &&
            entry->up && entry->down)
            return 1;
    }
    return 0;
}

static int expert_was_selected(const int *ids, int count, int expert_id)
{
    for (int i = 0; i < count; ++i) {
        if (ids[i] == expert_id)
            return 1;
    }
    return 0;
}

/* Colibri-style max-rank cache-aware routing.
 * Keep true top-J, prefer resident experts within top-M, then fill by rank.
 * The selected gate masses are re-normalized; ROUTE_ALPHA scales substitutions.
 */

static void select_cache_route(Model *m, int layer)
{
    int rank_ids[MODEL_NUM_EXPERTS];
    float rank_logits[MODEL_NUM_EXPERTS];
    unsigned char used[MODEL_NUM_EXPERTS] = {0};
    int rank_cap = m->route_m;
    if (rank_cap < MODEL_TOP_K)
        rank_cap = MODEL_TOP_K;
    if (rank_cap > MODEL_NUM_EXPERTS)
        rank_cap = MODEL_NUM_EXPERTS;
    /* Rank experts by router logit, descending. */
    for (int rank = 0; rank < rank_cap; ++rank) {
        int best = -1;
        float best_value = -INFINITY;
        for (int e = 0; e < MODEL_NUM_EXPERTS; ++e) {
            float value = m->router_logits[e];
            if (used[e] || !isfinite(value))
                continue;
            if (best < 0 || value > best_value) {
                best = e;
                best_value = value;
            }
        }
        /* Deterministic fallback if logits contain NaN/Inf everywhere. */
        if (best < 0) {
            for (int e = 0; e < MODEL_NUM_EXPERTS; ++e) {
                if (!used[e]) {
                    best = e;
                    best_value = isfinite(m->router_logits[e])
                        ? m->router_logits[e] : -FLT_MAX;
                    break;
                }
            }
        }
        if (best < 0) {
            rank_ids[rank] = rank % MODEL_NUM_EXPERTS;
            rank_logits[rank] = -INFINITY;
        } else {
            rank_ids[rank] = best;
            rank_logits[rank] = best_value;
            used[best] = 1;
        }
    }
    int m_window = rank_cap;
    if (m->route_p > 0.0f && m->route_p < 1.0f) {
        float max_logit = rank_logits[0];
        double total = 0.0;
        for (int i = 0; i < rank_cap; ++i) {
            if (isfinite(rank_logits[i]))
                total += exp((double)rank_logits[i] - max_logit);
        }
        if (total > 0.0 && isfinite(total)) {
            double cumulative = 0.0;
            m_window = MODEL_TOP_K;
            for (int i = 0; i < rank_cap; ++i) {
                if (isfinite(rank_logits[i]))
                    cumulative += exp((double)rank_logits[i] - max_logit);
                m_window = i + 1;
                if (cumulative >= (double)m->route_p * total)
                    break;
            }
            if (m_window < MODEL_TOP_K)
                m_window = MODEL_TOP_K;
            if (m_window > rank_cap)
                m_window = rank_cap;
        }
    }
    int sacred = m->route_j;
    if (sacred < 0)
        sacred = 0;
    if (sacred > MODEL_TOP_K)
        sacred = MODEL_TOP_K;
    int chosen = 0;
    for (int i = 0; i < sacred && chosen < MODEL_TOP_K; ++i) {
        m->expert_ids[chosen] = rank_ids[i];
        m->expert_weights[chosen] = rank_logits[i];
        ++chosen;
    }
    /* Fill remaining slots with resident candidates from the rank window. */
    for (int r = sacred; r < m_window && chosen < MODEL_TOP_K; ++r) {
        int eid = rank_ids[r];
        if (expert_was_selected(m->expert_ids, chosen, eid))
            continue;
        if (expert_is_resident(m->weights, layer, eid)) {
            m->expert_ids[chosen] = eid;
            m->expert_weights[chosen] = rank_logits[r];
            ++chosen;
        }
    }
    /* Fill any remaining positions with the highest-ranked candidates. */
    for (int r = 0; r < m_window && chosen < MODEL_TOP_K; ++r) {
        int eid = rank_ids[r];
        if (expert_was_selected(m->expert_ids, chosen, eid))
            continue;
        m->expert_ids[chosen] = eid;
        m->expert_weights[chosen] = rank_logits[r];
        ++chosen;
    }
    /* Defensive fallback; m_window is clamped to at least top-k. */
    for (int r = 0; chosen < MODEL_TOP_K && r < rank_cap; ++r) {
        int eid = rank_ids[r];
        if (expert_was_selected(m->expert_ids, chosen, eid))
            continue;
        m->expert_ids[chosen] = eid;
        m->expert_weights[chosen] = rank_logits[r];
        ++chosen;
    }
    /* Apply the same selected-set softmax as the baseline, then penalize only
       substituted experts and renormalize. */
    float max_selected = m->expert_weights[0];
    for (int i = 1; i < MODEL_TOP_K; ++i) {
        if (m->expert_weights[i] > max_selected)
            max_selected = m->expert_weights[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < MODEL_TOP_K; ++i) {
        float weight = expf(m->expert_weights[i] - max_selected);
        if (m->route_alpha > 0.0f && m->route_alpha < 1.0f &&
            !expert_was_selected(rank_ids, MODEL_TOP_K, m->expert_ids[i]))
            weight *= m->route_alpha;
        m->expert_weights[i] = weight;
        sum += weight;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = 0; i < MODEL_TOP_K; ++i)
        m->expert_weights[i] *= inv;
}

static int moe_forward(Model *m, int layer)
{
    const LayerRefs *R = &((const ModelRefs *)m->refs)->layer[layer];
    if (mv(&R->router, m->normed, m->router_logits,
           MODEL_NUM_EXPERTS, MODEL_HIDDEN_SIZE) != 0)
        return -1;
    if (m->cache_route)
        select_cache_route(m, layer);
    else
        select_top8(m->router_logits, m->expert_ids, m->expert_weights);
    /* Issue every missing expert read for this layer at once (parallel preads). */
    (void)weights_prefetch_experts(m->weights, layer, m->expert_ids, MODEL_TOP_K);

    CachedExpert *ex[MODEL_TOP_K];
    for (int k = 0; k < MODEL_TOP_K; k++) {
        ex[k] = weights_get_expert(m->weights, layer, m->expert_ids[k]);
        if (!ex[k] || !ex[k]->gate || !ex[k]->up || !ex[k]->down) return -1;
        if ((ex[k]->gate_type != 12 && ex[k]->gate_type != 14) ||
            (ex[k]->up_type != 12 && ex[k]->up_type != 14) ||
            (ex[k]->down_type != 12 && ex[k]->down_type != 14))
            return -1;
    }
    float g[MODEL_TOP_K][512], u[MODEL_TOP_K][512], eh[MODEL_TOP_K][512], d[MODEL_TOP_K][2048];
    float sg[512], su[512], sh[512], sd[2048];
    KernelMatvecJob jobs[2 * MODEL_TOP_K + 2];

    /*
     * Fast path: gate/up -> SiLU*up -> down for the 8 routed experts and the shared expert in ONE
     * GPU submit (the generic path below needs two round trips). Falls back to the generic path when the GPU path is unavailable.
     */
    float shared_gate_raw = 0.0f;
    int fused = 0;
    if (R->sh_gate.t && R->sh_up.t && R->sh_down.t && R->sh_gate.d && R->sh_up.d && R->sh_down.d &&
        (R->sh_gate.t->type == 12 || R->sh_gate.t->type == 14) &&
        (R->sh_up.t->type == 12 || R->sh_up.t->type == 14) &&
        (R->sh_down.t->type == 12 || R->sh_down.t->type == 14)) {
        KernelFfnExpert fx[MODEL_TOP_K + 1];
        for (int k = 0; k < MODEL_TOP_K; k++)
            fx[k] = (KernelFfnExpert){ ex[k]->gate, ex[k]->up, ex[k]->down,
                                       ex[k]->gate_type == 14, ex[k]->up_type == 14,
                                       ex[k]->down_type == 14, d[k] };
        fx[MODEL_TOP_K] = (KernelFfnExpert){ R->sh_gate.d, R->sh_up.d, R->sh_down.d,
                                             R->sh_gate.t->type == 14, R->sh_up.t->type == 14,
                                             R->sh_down.t->type == 14, sd };
        fused = kernel_ffn_fused(fx, MODEL_TOP_K + 1, m->normed, 512, 2048);
    }
    if (!fused) {
    /* Round trip 1: gate+up of all 8 routed experts and the shared expert (same input). */
    int nj = 0;
    for (int k = 0; k < MODEL_TOP_K; k++) {
        jobs[nj++] = (KernelMatvecJob){ ex[k]->gate_type == 14, ex[k]->gate, m->normed, g[k], 512, 2048 };
        jobs[nj++] = (KernelMatvecJob){ ex[k]->up_type == 14,   ex[k]->up,   m->normed, u[k], 512, 2048 };
    }
    if (add_ref_job(jobs, &nj, &R->sh_gate, m->normed, sg, 512, 2048) != 0) return -1;
    if (add_ref_job(jobs, &nj, &R->sh_up,   m->normed, su, 512, 2048) != 0) return -1;
    /* The GPU runs the batch while the CPU computes the shared-expert gate logit. */
    kernel_matvec_quant_begin(jobs, nj);
    if (mv(&R->sh_gate_inp, m->normed, &shared_gate_raw, 1, 2048) != 0) {
        kernel_matvec_quant_end();
        return -1;
    }
    kernel_matvec_quant_end();

    for (int k = 0; k < MODEL_TOP_K; k++)
        for (int i = 0; i < 512; i++)
            eh[k][i] = (g[k][i] / (1.0f + expf(-g[k][i]))) * u[k][i];
    for (int i = 0; i < 512; i++)
        sh[i] = (sg[i] / (1.0f + expf(-sg[i]))) * su[i];

    /* Round trip 2: down projections (each has its own input). */
    nj = 0;
    for (int k = 0; k < MODEL_TOP_K; k++)
        jobs[nj++] = (KernelMatvecJob){ ex[k]->down_type == 14, ex[k]->down, eh[k], d[k], 2048, 512 };
    if (add_ref_job(jobs, &nj, &R->sh_down, sh, sd, 2048, 512) != 0) return -1;
    kernel_matvec_quant_batch(jobs, nj);

    }
    if (fused && mv(&R->sh_gate_inp, m->normed, &shared_gate_raw, 1, 2048) != 0)
        return -1;
    float shared_gate = 1.0f / (1.0f + expf(-shared_gate_raw));
    memset(m->ffn_output, 0, MODEL_HIDDEN_SIZE * sizeof(float));
    for (int k = 0; k < MODEL_TOP_K; k++) {
        float w = m->expert_weights[k];
        for (int i = 0; i < 2048; i++) m->ffn_output[i] += w * d[k][i];
    }
    for (int i = 0; i < 2048; i++) m->ffn_output[i] += shared_gate * sd[i];
    return 0;
}

static uint64_t model_rng_next(Model *model)
{
    /* xorshift64*: independent RNG state per slot. */
    uint64_t x = model->rng_state;
    if (x == 0)
        x = UINT64_C(0x2545f4914f6cdd1d);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    model->rng_state = x;
    return x * UINT64_C(0x2545f4914f6cdd1d);
}

static double model_rng_uniform(Model *model)
{
    /* Use the high 53 bits to produce a value in [0, 1). */
    return (double)(model_rng_next(model) >> 11) * (1.0 / 9007199254740992.0);
}

static int sample_token(Model *m, const float *logits, size_t n)
{
    if (!m || !logits || n == 0)
        return -1;
    /* Temperature zero means deterministic greedy decoding. */
    if (m->temperature == 0.0f) {
        size_t best = 0;
        for (size_t i = 1; i < n; ++i) {
            if (logits[i] > logits[best])
                best = i;
        }
        return (int)best;
    }
    size_t limit = n;
    if (m->top_k > 0 && (size_t)m->top_k < limit)
        limit = (size_t)m->top_k;
    if (limit == 0 || !m->sampling_ids || !m->sampling_logits)
        return -1;
    /* Maintain the top-k candidates in descending logit order. */
    size_t count = 0;
    for (size_t token = 0; token < n; ++token) {
        float value = logits[token] / m->temperature;
        if (!isfinite(value))
            continue;
        if (count == limit && value <= m->sampling_logits[limit - 1])
            continue;
        size_t insert_at = 0;
        while (insert_at < count &&
               m->sampling_logits[insert_at] >= value)
            ++insert_at;
        if (insert_at >= limit)
            continue;
        size_t new_count = count < limit ? count + 1 : count;
        for (size_t j = new_count; j > insert_at + 1; --j) {
            m->sampling_logits[j - 1] = m->sampling_logits[j - 2];
            m->sampling_ids[j - 1] = m->sampling_ids[j - 2];
        }
        m->sampling_logits[insert_at] = value;
        m->sampling_ids[insert_at] = (int32_t)token;
        count = new_count;
    }
    if (count == 0)
        return -1;
    /* Stable softmax over the retained candidates. */
    float max_logit = m->sampling_logits[0];
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double p = exp((double)m->sampling_logits[i] - max_logit);
        m->sampling_logits[i] = (float)p;
        sum += p;
    }
    if (!(sum > 0.0) || !isfinite(sum))
        return (int)m->sampling_ids[0];
    for (size_t i = 0; i < count; ++i)
        m->sampling_logits[i] = (float)(m->sampling_logits[i] / sum);
    /* Nucleus filtering: retain the smallest prefix reaching top_p. */
    size_t nucleus_count = count;
    if (m->top_p < 1.0f) {
        double cumulative = 0.0;
        for (size_t i = 0; i < count; ++i) {
            cumulative += m->sampling_logits[i];
            if (cumulative >= m->top_p) {
                nucleus_count = i + 1;
                break;
            }
        }
    }
    double nucleus_sum = 0.0;
    for (size_t i = 0; i < nucleus_count; ++i)
        nucleus_sum += m->sampling_logits[i];
    if (!(nucleus_sum > 0.0))
        return (int)m->sampling_ids[0];
    double draw = model_rng_uniform(m) * nucleus_sum;
    double cumulative = 0.0;
    for (size_t i = 0; i < nucleus_count; ++i) {
        cumulative += m->sampling_logits[i];
        if (draw < cumulative)
            return (int)m->sampling_ids[i];
    }
    return (int)m->sampling_ids[nucleus_count - 1];
}

/* ============================================================
 * Forward pass
 * ============================================================ */

/*
 * want_logits == 0 is used for prompt prefill: every prompt token except the
 * last only needs to update KV / recurrent state, so the 248k x 2048 LM head
 * and the sampler are skipped entirely.
 */
static int32_t forward_impl(Model *model, int32_t input_token, int want_logits)
{
    if (!model || !model->initialized || !model->refs)
        return MODEL_FORWARD_ERROR;
    if (input_token < 0 || input_token >= MODEL_VOCAB_SIZE)
        return MODEL_FORWARD_ERROR;
    if (model->position >= model->context_length)
        return MODEL_FORWARD_ERROR;
    const ModelRefs *R = (const ModelRefs *)model->refs;
    if (model_embedding(model, input_token, model->hidden) != MODEL_FORWARD_OK)
        return MODEL_FORWARD_ERROR;
    for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++) {
        const LayerRefs *L = &R->layer[layer];
        if (!L->attn_norm || !L->post_norm)
            return MODEL_FORWARD_ERROR;
        kernel_rmsnorm(model->normed, model->hidden, L->attn_norm,
                       MODEL_HIDDEN_SIZE, 1e-6f);
        if (model_is_attention_layer(layer)) {
            if (full_attention(model, layer) != 0)
                return MODEL_FORWARD_ERROR;
        } else {
            float alpha[MODEL_LINEAR_V_HEADS];
            float beta[MODEL_LINEAR_V_HEADS];
            {
                /* qkv, gate, alpha and beta share the same input: ONE GPU round trip. */
                const TRef *refs[4] = { &L->qkv, &L->gate, &L->alpha, &L->beta };
                float *outs[4] = { model->qkv, model->linear_gate, alpha, beta };
                const size_t rws[4] = { MODEL_LINEAR_QKV_SIZE, MODEL_LINEAR_GATE_SIZE,
                                        MODEL_LINEAR_V_HEADS, MODEL_LINEAR_V_HEADS };
                if (mv_batch(refs, outs, rws, 4, model->normed, MODEL_HIDDEN_SIZE) != 0)
                    return MODEL_FORWARD_ERROR;
            }
            if (!L->conv || !L->dt || !L->a || !L->ssm_norm)
                return MODEL_FORWARD_ERROR;
            kernel_conv1d_causal_f32(
                model->qkv, model->ssm_state[layer].conv_state,
                L->conv, model->recurrent_output,
                MODEL_LINEAR_QKV_SIZE, MODEL_SSM_CONV_KERNEL);
            kernel_silu_array(model->recurrent_output,
                              MODEL_LINEAR_QKV_SIZE);
            float *history = model->ssm_state[layer].conv_state;
            memmove(history, history + MODEL_LINEAR_QKV_SIZE,
                    (MODEL_SSM_CONV_KERNEL - 2) *
                    MODEL_LINEAR_QKV_SIZE * sizeof(float));
            memcpy(history + (MODEL_SSM_CONV_KERNEL - 2) *
                   MODEL_LINEAR_QKV_SIZE,
                   model->qkv,
                   MODEL_LINEAR_QKV_SIZE * sizeof(float));
            memcpy(model->linear_q, model->recurrent_output,
                   MODEL_LINEAR_Q_SIZE * sizeof(float));
            memcpy(model->linear_k,
                   model->recurrent_output + MODEL_LINEAR_Q_SIZE,
                   MODEL_LINEAR_K_SIZE * sizeof(float));
            memcpy(model->linear_v,
                   model->recurrent_output + MODEL_LINEAR_Q_SIZE +
                   MODEL_LINEAR_K_SIZE,
                   MODEL_LINEAR_V_SIZE * sizeof(float));
            for (int h = 0; h < MODEL_LINEAR_K_HEADS; h++) {
                kernel_l2norm(model->linear_q + h * MODEL_LINEAR_HEAD_DIM,
                              model->linear_q + h * MODEL_LINEAR_HEAD_DIM,
                              MODEL_LINEAR_HEAD_DIM, 1e-6f);
                kernel_l2norm(model->linear_k + h * MODEL_LINEAR_HEAD_DIM,
                              model->linear_k + h * MODEL_LINEAR_HEAD_DIM,
                              MODEL_LINEAR_HEAD_DIM, 1e-6f);
            }
            for (int h = 0; h < MODEL_LINEAR_V_HEADS; h++) {
                beta[h] = 1.0f / (1.0f + expf(-beta[h]));
                float x = alpha[h] + L->dt[h];
                alpha[h] = L->a[h] * (x > 20.0f ? x : log1pf(expf(x)));
            }
            recurrent_delta_step(&model->ssm_state[layer],
                                 model->linear_q, model->linear_k,
                                 model->linear_v, alpha, beta,
                                 model->recurrent_output);
            for (int h = 0; h < MODEL_LINEAR_V_HEADS; h++) {
                float *x = model->recurrent_output +
                           h * MODEL_LINEAR_HEAD_DIM;
                const float *z = model->linear_gate +
                                 h * MODEL_LINEAR_HEAD_DIM;
                kernel_rmsnorm(x, x, L->ssm_norm, MODEL_LINEAR_HEAD_DIM, 1e-6f);
                for (int j = 0; j < MODEL_LINEAR_HEAD_DIM; j++) {
                    float sz = 1.0f / (1.0f + expf(-z[j]));
                    x[j] *= z[j] * sz;
                }
            }
            if (mv(&L->ssm_out, model->recurrent_output,
                   model->residual,
                   MODEL_HIDDEN_SIZE, MODEL_LINEAR_V_SIZE) != 0)
                return MODEL_FORWARD_ERROR;
        }
        /* Attention/DeltaNet residual */
        for (size_t i = 0; i < MODEL_HIDDEN_SIZE; i++)
            model->hidden[i] += model->residual[i];
        kernel_rmsnorm(model->normed, model->hidden, L->post_norm,
                       MODEL_HIDDEN_SIZE, 1e-6f);
        if (moe_forward(model, layer) != 0)
            return MODEL_FORWARD_ERROR;
        /* MoE residual */
        for (size_t i = 0; i < MODEL_HIDDEN_SIZE; i++)
            model->hidden[i] += model->ffn_output[i];
    }
    if (!want_logits) {
        model->position++;
        return MODEL_FORWARD_OK;
    }
    if (!R->out_norm)
        return MODEL_FORWARD_ERROR;
    kernel_rmsnorm(model->normed, model->hidden, R->out_norm,
                   MODEL_HIDDEN_SIZE, 1e-6f);
    if (mv(&R->out, model->normed, model->logits,
           MODEL_VOCAB_SIZE, MODEL_HIDDEN_SIZE) != 0)
        return MODEL_FORWARD_ERROR;
    int sampled = sample_token(model, model->logits, MODEL_VOCAB_SIZE);
    if (sampled < 0 || sampled >= MODEL_VOCAB_SIZE)
        return MODEL_FORWARD_ERROR;
    model->position++;
    return (int32_t)sampled;
}

int32_t model_forward(Model *model, int32_t input_token)
{
    return forward_impl(model, input_token, 1);
}

/* Advance state by one prompt token without computing logits. */
int model_prefill(Model *model, int32_t input_token)
{
    int32_t rc = forward_impl(model, input_token, 0);
    return rc == MODEL_FORWARD_OK ? MODEL_FORWARD_OK : MODEL_FORWARD_ERROR;
}

/* ============================================================
 * Free
 * ============================================================ */

void model_free(Model *model)
{
    if (!model)
        return;
    if (model->kv_cache) {
        for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++)
            kv_cache_free(&model->kv_cache[layer]);
        free(model->kv_cache);
    }
    if (model->ssm_state) {
        for (int layer = 0; layer < MODEL_NUM_LAYERS; layer++)
            ssm_state_free(&model->ssm_state[layer]);
        free(model->ssm_state);
    }
    free(model->hidden);
    free(model->residual);
    free(model->normed);
    free(model->linear_q);
    free(model->linear_k);
    free(model->linear_v);
    free(model->linear_gate);
    free(model->qkv);
    free(model->attn_q);
    free(model->attn_k);
    free(model->attn_v);
    free(model->recurrent_output);
    free(model->ffn_output);
    free(model->router_logits);
    free(model->expert_ids);
    free(model->expert_weights);
    free(model->logits);
    free(model->sampling_ids);
    free(model->sampling_logits);
    free(model->attn_scratch);
    free(model->refs);
    /*
     * model->weights is shared and not owned here.
     */
    memset(model, 0, sizeof(*model));
}