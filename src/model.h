#ifndef MODEL_H
#define MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "weights.h"
#include "kvspill.h"

#define MODEL_HIDDEN_SIZE        2048
#define MODEL_VOCAB_SIZE         248320
#define MODEL_NUM_LAYERS         40

#define MODEL_LINEAR_K_HEADS     16
#define MODEL_LINEAR_V_HEADS     32
#define MODEL_LINEAR_HEAD_DIM    128

#define MODEL_NUM_EXPERTS        256
#define MODEL_TOP_K              8

#define MODEL_LINEAR_Q_SIZE      2048
#define MODEL_LINEAR_K_SIZE      2048
#define MODEL_LINEAR_V_SIZE      4096
#define MODEL_LINEAR_QKV_SIZE    8192
#define MODEL_LINEAR_GATE_SIZE   4096

#define MODEL_ATTN_Q_SIZE        8192
#define MODEL_ATTN_K_SIZE        512
#define MODEL_ATTN_V_SIZE        512

#define MODEL_SSM_INNER_SIZE     4096
#define MODEL_SSM_STATE_SIZE     128
#define MODEL_SSM_GROUP_COUNT    16
#define MODEL_SSM_TIME_RANK      32
#define MODEL_SSM_CONV_KERNEL    4

#define MODEL_FULL_ATTN_INTERVAL 4

#define MODEL_QK_HEAD_COUNT      16
#define MODEL_V_HEAD_COUNT       32

#define MODEL_FORWARD_OK         0
#define MODEL_FORWARD_ERROR     -1

typedef struct {
    float *k;
    float *v;
    size_t capacity;
    size_t length;
} KVCache;

typedef struct {
    float *state;
    float *conv_state;
    size_t state_elements;
    size_t conv_elements;
} SSMState;

typedef struct {
    /* Shared weights. Not owned by Model. */
    Weights *weights;

    /* Per-slot generation configuration. */
    size_t context_length;
    float temperature;
    float top_p;
    int top_k;
    uint64_t rng_state;

    /* Colibri-style routing controls (opt-in). */
    bool cache_route;
    int route_j;
    int route_m;
    float route_alpha;
    float route_p;

    /* Absolute sequence position: RoPE position = total tokens processed since the last reset.
     * It keeps growing when the attention window slides. */
    size_t position;

    /* Context sliding. The KV window holds `position - spill->count` tokens; the ones that slid
     * out live on SSD in `spill` (owned by the slot, not by Model). The first `slide_keep`
     * tokens always stay in the window (attention sinks). Recurrent layers need nothing: their
     * state already summarises everything. */
    bool slide;
    int slide_keep;
    size_t slide_pin;    /* tokens of the pinned system prompt: never slid out (set per conversation) */
    KVSpill *spill;

    /* Full-attention KV caches. */
    KVCache *kv_cache;

    /* Recurrent SSM state. */
    SSMState *ssm_state;

    /* Main hidden-state buffers. */
    float *hidden;
    float *residual;
    float *normed;

    /* Attention / recurrent intermediates. */
    float *qkv;
    float *linear_q;
    float *linear_k;
    float *linear_v;
    float *linear_gate;
    float *attn_q;
    float *attn_k;
    float *attn_v;
    float *attn_gate;
    float *recurrent_output;

    /* MoE output and router working buffers. */
    float *ffn_output;
    float *router_logits;
    int *expert_ids;
    float *expert_weights;

    /* Final vocabulary logits. */
    float *logits;

    /* Sampling buffers. */
    int32_t *sampling_ids;
    float *sampling_logits;

    /* Per-head attention score scratch: 16 * context_length floats. */
    float *attn_scratch;

    /* Pre-resolved tensor pointers (opaque; see model.c). */
    void *refs;

    bool initialized;
} Model;

int model_init(
    Model *model,
    Weights *weights,
    size_t context_length,
    float temperature,
    float top_p,
    int top_k
);

int32_t model_forward(Model *model, int32_t input_token);
/* Process a prompt token without computing logits or sampling.
 * Returns MODEL_FORWARD_OK or MODEL_FORWARD_ERROR. */
int model_prefill(Model *model, int32_t input_token);
/* Advance the context by n prompt tokens without computing logits, in chunks of up to
 * model_prefill_max_chunk() tokens (batched matmuls, GPU when available, MoE experts grouped).
 * Uses one scratch area shared by all slots: call it only while holding the slot manager's step lock.
 * On failure the context is inconsistent (roll back or reset it). */
int model_prefill_batch(Model *model, const int32_t *tokens, size_t n);
int model_prefill_max_chunk(void);
/* Free the shared prefill scratch and the GPU staging buffers it grew. Hold the step lock. */
void model_prefill_release(void);
void model_reset(Model *model);
/* Change the attention window. Growing brings spilled tokens back from SSD, shrinking pushes the
 * oldest ones out to SSD. Set model->slide / slide_keep first. 0 = ok, -1 = failed (unchanged
 * apart from tokens already moved to SSD, which stay recoverable). */
int model_set_context(Model *model, size_t new_ctx);
void model_free(Model *model);
bool model_is_attention_layer(int layer);
bool model_is_recurrent_layer(int layer);

#endif