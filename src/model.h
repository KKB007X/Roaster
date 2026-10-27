#ifndef MODEL_H
#define MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "weights.h"

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

    /* Current sequence position. */
    size_t position;

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
void model_reset(Model *model);
void model_free(Model *model);
bool model_is_attention_layer(int layer);
bool model_is_recurrent_layer(int layer);

#endif