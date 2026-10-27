#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "slots.h"
#include "protocol.h"

#define PROMPT_BUFFER_SIZE 16384

void slot_config_defaults(SlotConfig *c)
{
    c->context_length = DEFAULT_CONTEXT;
    c->cache_route    = DEFAULT_CACHE_ROUTE;
    c->route_j        = DEFAULT_ROUTE_J;
    c->route_m        = DEFAULT_ROUTE_M;
    c->route_alpha    = DEFAULT_ROUTE_ALPHA;
    c->route_p        = DEFAULT_ROUTE_P;
    c->temperature    = DEFAULT_TEMPERATURE;
    c->top_p          = DEFAULT_TOP_P;
    c->top_k          = DEFAULT_TOP_K;
    c->max_tokens     = DEFAULT_MAX_TOKENS;
    c->think          = DEFAULT_THINK;
}

const char *slot_config_check(const SlotConfig *c)
{
    static char msg[96];
    if (c->context_length == 0)
        return "Context length must be greater than zero.";
    if (c->cache_route != 0 && c->cache_route != 1)
        return "cache-route must be 0 or 1.";
    if (c->route_j < 0 || c->route_j > MODEL_TOP_K) {
        snprintf(msg, sizeof msg, "route-j must be between 0 and %d.", MODEL_TOP_K);
        return msg;
    }
    if (c->route_m < MODEL_TOP_K || c->route_m > MODEL_NUM_EXPERTS) {
        snprintf(msg, sizeof msg, "route-m must be between %d and %d.", MODEL_TOP_K, MODEL_NUM_EXPERTS);
        return msg;
    }
    if (!isfinite(c->route_alpha) || c->route_alpha < 0.0f || c->route_alpha > 1.0f)
        return "route-alpha must be between 0 and 1.";
    if (!isfinite(c->route_p) || c->route_p < 0.0f || c->route_p > 1.0f)
        return "route-p must be between 0 and 1.";
    if (!isfinite(c->temperature) || c->temperature < 0.0f)
        return "temperature must be finite and >= 0.";
    if (!isfinite(c->top_p) || c->top_p <= 0.0f || c->top_p > 1.0f)
        return "top-p must be in (0, 1].";
    if (c->top_k <= 0 || c->top_k > MODEL_VOCAB_SIZE) {
        snprintf(msg, sizeof msg, "top-k must be between 1 and %d.", MODEL_VOCAB_SIZE);
        return msg;
    }
    if (c->max_tokens <= 0)
        return "max-tokens must be greater than zero.";
    if (c->think != 0 && c->think != 1)
        return "think must be 0 (off) or 1 (on).";
    return NULL;
}

static void fair_lock_init(FairLock *l)
{
    pthread_mutex_init(&l->m, NULL);
    pthread_cond_init(&l->c, NULL);
    l->next = l->serving = 0;
}

static void fair_lock_destroy(FairLock *l)
{
    pthread_cond_destroy(&l->c);
    pthread_mutex_destroy(&l->m);
}

static void fair_lock(FairLock *l)
{
    pthread_mutex_lock(&l->m);
    unsigned long ticket = l->next++;
    while (l->serving != ticket)
        pthread_cond_wait(&l->c, &l->m);
    pthread_mutex_unlock(&l->m);
}

static void fair_unlock(FairLock *l)
{
    pthread_mutex_lock(&l->m);
    l->serving++;
    pthread_cond_broadcast(&l->c);
    pthread_mutex_unlock(&l->m);
}

int slots_init(SlotManager *sm, Weights *weights, const Tokenizer *tokenizer, int max_slots)
{
    memset(sm, 0, sizeof(*sm));
    sm->slots = calloc((size_t)max_slots, sizeof(InferenceSlot));
    if (!sm->slots)
        return -1;
    sm->weights = weights;
    sm->tokenizer = tokenizer;
    sm->max_slots = max_slots;
    pthread_mutex_init(&sm->table_lock, NULL);
    fair_lock_init(&sm->step_lock);
    /* The tokenizer builds its byte<->unicode tables lazily on first use; do that once here so
     * concurrent slot_run() calls only ever read them. */
    int32_t *warm = NULL;
    size_t wn = tokenizer_encode(tokenizer, " a\n", &warm);
    if (warm) {
        free(tokenizer_decode(tokenizer, warm, wn));
        free(warm);
    }
    return 0;
}

void slots_free(SlotManager *sm)
{
    if (sm->slots) {
        for (int i = 0; i < sm->max_slots; i++)
            if (sm->slots[i].active)
                model_free(&sm->slots[i].model);
        free(sm->slots);
        pthread_mutex_destroy(&sm->table_lock);
        fair_lock_destroy(&sm->step_lock);
    }
    memset(sm, 0, sizeof(*sm));
}

int slot_is_active(const SlotManager *sm, int id)
{
    return id >= 0 && id < sm->max_slots && sm->slots[id].active;
}

static int slot_create_locked(SlotManager *sm, const SlotConfig *config)
{
    int id = -1;
    for (int i = 0; i < sm->max_slots; i++) {
        if (!sm->slots[i].active) {
            id = i;
            break;
        }
    }
    if (id < 0) {
        fprintf(stderr, "Maximum number of slots reached\n");
        return -1;
    }
    InferenceSlot *slot = &sm->slots[id];
    memset(slot, 0, sizeof(*slot));
    if (model_init(&slot->model, sm->weights, config->context_length,
                   config->temperature, config->top_p, config->top_k) != 0) {
        fprintf(stderr, "Failed to initialize slot %d\n", id);
        return -1;
    }
    /* Routing configuration is per slot; expert weights remain shared. */
    slot->model.cache_route = config->cache_route != 0;
    slot->model.route_j = config->route_j;
    slot->model.route_m = config->route_m;
    slot->model.route_alpha = config->route_alpha;
    slot->model.route_p = config->route_p;

    slot->config = *config;
    slot->active = 1;
    printf("Created slot %d\n", id);
    printf("  Context:     %zu\n", config->context_length);
    printf("  Cache-aware routing: %s\n", config->cache_route ? "enabled" : "disabled");
    printf("  Route J/M:    %d/%d\n", config->route_j, config->route_m);
    printf("  Route alpha:  %.2f\n", config->route_alpha);
    printf("  Route p:      %.2f\n", config->route_p);
    printf("  Temperature: %.2f\n", config->temperature);
    printf("  Top-p:       %.2f\n", config->top_p);
    printf("  Top-k:       %d\n", config->top_k);
    printf("  Max tokens:  %d\n", config->max_tokens);
    printf("  Thinking:    %s\n", config->think ? "on" : "off");
    return id;
}

static void slot_destroy_locked(SlotManager *sm, int id)
{
    if (!slot_is_active(sm, id)) {
        fprintf(stderr, "Invalid or inactive slot\n");
        return;
    }
    model_free(&sm->slots[id].model);
    memset(&sm->slots[id], 0, sizeof(sm->slots[id]));
    printf("Destroyed slot %d\n", id);
}

int slot_create(SlotManager *sm, const SlotConfig *config)
{
    pthread_mutex_lock(&sm->table_lock);
    int id = slot_create_locked(sm, config);
    pthread_mutex_unlock(&sm->table_lock);
    return id;
}

void slot_destroy(SlotManager *sm, int id)
{
    pthread_mutex_lock(&sm->table_lock);
    slot_destroy_locked(sm, id);
    pthread_mutex_unlock(&sm->table_lock);
}

/* One model step under the round-robin lock. Everything else (tokenizing, decoding, writing to
 * the client) happens outside it, so a slow client never holds up the other slots. */
static int32_t step_forward(SlotManager *sm, Model *m, int32_t token, int prefill, double *wait)
{
    double t0 = omp_get_wtime();
    fair_lock(&sm->step_lock);
    *wait += omp_get_wtime() - t0;
    int32_t r;
    if (g_slots_abort)
        r = MODEL_FORWARD_ERROR;
    else if (prefill)
        r = model_prefill(m, token) == MODEL_FORWARD_OK ? 0 : MODEL_FORWARD_ERROR;
    else
        r = model_forward(m, token);
    fair_unlock(&sm->step_lock);
    return r;
}

/* Qwen3.x chat turn. With thinking on the assistant turn is opened with "<think>\n" and the model
 * reasons first; with thinking off an empty think block is pre-filled so it answers directly.
 * Template text is tokenized with special-token parsing; the user's text never is, so typing
 * "<|im_end|>" in a prompt cannot inject control tokens. */
static size_t build_prompt_tokens(const Tokenizer *tok, const char *user, int think, int32_t **out)
{
    const char *head = "<|im_start|>user\n";
    const char *tail = think
        ? "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        : "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    int32_t *a = NULL, *b = NULL, *c = NULL;
    *out = NULL;
    size_t na = tokenizer_encode_special(tok, head, &a);
    size_t nb = user[0] ? tokenizer_encode(tok, user, &b) : 0;
    size_t nc = tokenizer_encode_special(tok, tail, &c);
    if (!a || !c || (user[0] && !b)) {
        free(a); free(b); free(c);
        return 0;
    }
    int32_t *all = malloc((na + nb + nc) * sizeof(int32_t));
    if (all) {
        memcpy(all, a, na * sizeof(int32_t));
        if (nb)
            memcpy(all + na, b, nb * sizeof(int32_t));
        memcpy(all + na + nb, c, nc * sizeof(int32_t));
        *out = all;
    }
    free(a); free(b); free(c);
    return all ? na + nb + nc : 0;
}

/* Write model text; drops bytes that collide with the stream markers and, after a think tag,
 * the blank lines the template puts there. */
static void emit_text(FILE *out, const char *s, int *skip_nl)
{
    for (; *s; s++) {
        unsigned char ch = (unsigned char)*s;
        if (ch >= 1 && ch <= 6)
            continue;
        if (*skip_nl) {
            if (ch == '\n')
                continue;
            *skip_nl = 0;
        }
        fputc(ch, out);
    }
}

volatile sig_atomic_t g_slots_abort = 0;

int slot_run(SlotManager *sm, int id, const char *user_prompt, FILE *out, FILE *info)
{
    if (!slot_is_active(sm, id)) {
        fprintf(info, "Invalid or inactive slot\n");
        return -1;
    }
    InferenceSlot *slot = &sm->slots[id];
    const Tokenizer *tok = sm->tokenizer;
    int max_tokens = slot->config.max_tokens;

    if (strlen(user_prompt) >= PROMPT_BUFFER_SIZE - 128) {
        fprintf(info, "Prompt is too long\n");
        return -1;
    }
    int32_t *tokens = NULL;
    size_t token_count = build_prompt_tokens(tok, user_prompt, slot->config.think, &tokens);
    if (!tokens || token_count == 0) {
        fprintf(info, "Prompt tokenization failed\n");
        free(tokens);
        return -1;
    }
    if (token_count > slot->config.context_length ||
        (size_t)max_tokens > slot->config.context_length - token_count) {
        fprintf(info, "Prompt plus requested generation exceeds slot context\n");
        free(tokens);
        return -1;
    }

    /* A slot is reused across requests: reset recurrent state, attention cache and position. */
    model_reset(&slot->model);

    /* Prefill: every prompt token; only the last one produces the first output token. */
    int32_t next_token = MODEL_FORWARD_ERROR;
    double wait_seconds = 0.0;
    double prefill_start = omp_get_wtime();
    for (size_t i = 0; i < token_count; i++) {
        if (g_slots_abort) {
            free(tokens);
            return -1;
        }
        next_token = step_forward(sm, &slot->model, tokens[i], i + 1 < token_count, &wait_seconds);
        if (next_token == MODEL_FORWARD_ERROR) {
            if (!g_slots_abort)
                fprintf(info, "\nForward pass failed during prefill after %.3f seconds\n",
                        omp_get_wtime() - prefill_start);
            free(tokens);
            return -1;
        }
    }
    double prefill_seconds = omp_get_wtime() - prefill_start;
    free(tokens);

    /* Decode timing starts once prefill has produced the first output token. */
    double generation_start = omp_get_wtime();
    size_t generated_tokens = 0, think_tokens = 0;
    enum { ST_ANSWER, ST_THINK, ST_TOOL } state = ST_ANSWER;
    int skip_nl = 0, stopped = 0;
    if (slot->config.think) {              /* the prompt already opened <think> */
        fputc(STREAM_THINK_BEGIN, out);
        state = ST_THINK;
    }
    for (int i = 0; i < max_tokens; i++) {
        if (next_token < 0 || (uint64_t)next_token >= tok->vocab_size) {
            fprintf(info, "\nInvalid generated token: %d\n", next_token);
            return -1;
        }
        generated_tokens++;
        if (tokenizer_is_stop(tok, next_token)) {
            stopped = 1;
            break;
        }
        if (next_token == tok->think_open_id) {
            fputc(STREAM_THINK_BEGIN, out);
            state = ST_THINK;
            skip_nl = 1;
        } else if (next_token == tok->think_close_id) {
            fputc(STREAM_THINK_END, out);
            state = ST_ANSWER;
            skip_nl = 1;
        } else if (next_token == tok->tool_open_id) {
            fputc(STREAM_TOOL_BEGIN, out);
            state = ST_TOOL;
        } else if (next_token == tok->tool_close_id) {
            fputc(STREAM_TOOL_END, out);
            state = ST_ANSWER;
        } else if (tokenizer_is_special(tok, next_token)) {
            fputc(STREAM_SPECIAL_BEGIN, out);
            fputs(tok->tokens[next_token].text, out);
            fputc(STREAM_SPECIAL_END, out);
        } else {
            char *decoded = tokenizer_decode(tok, &next_token, 1);
            if (decoded) {
                emit_text(out, decoded, &skip_nl);
                free(decoded);
            }
        }
        if (state == ST_THINK)
            think_tokens++;
        fflush(out);                            /* stream every token immediately */
        if (ferror(out) || g_slots_abort)       /* client disconnected or server stopping */
            return -1;
        if (i + 1 < max_tokens) {
            next_token = step_forward(sm, &slot->model, next_token, 0, &wait_seconds);
            if (next_token == MODEL_FORWARD_ERROR) {
                if (g_slots_abort)
                    return -1;
                fprintf(info, "\nForward pass failed during generation after %.3f seconds\n",
                        omp_get_wtime() - generation_start);
                return -1;
            }
        }
    }
    double generation_seconds = omp_get_wtime() - generation_start;
    fputc('\n', out);

    double prefill_tps = prefill_seconds > 0.0 ? (double)token_count / prefill_seconds : 0.0;
    /* The first output token is computed during prefill, so it is not a decode token. */
    size_t decode_tokens = generated_tokens > 0 ? generated_tokens - 1 : 0;
    double generation_tps = generation_seconds > 0.0 ? (double)decode_tokens / generation_seconds : 0.0;

    if (!stopped)
        fprintf(info, "\nStopped at max-tokens%s.\n",
                state == ST_THINK ? " while still thinking (raise --max-tokens or use --think 0)" : "");
    fprintf(info,
            "\nPerformance:\n"
            "  Prompt tokens:  %zu\n"
            "  Prefill time:   %.3f s\n"
            "  Prefill speed:  %.2f tokens/s\n"
            "  Output tokens:  %zu (including end token if generated)\n"
            "  Think tokens:   %zu\n"
            "  Decode tokens:  %zu (excluding first output token)\n"
            "  Decode time:    %.3f s\n"
            "  Generation TPS: %.2f tokens/s\n"
            "  Turn wait:      %.3f s (queued behind other slots)\n",
            token_count, prefill_seconds, prefill_tps,
            generated_tokens, think_tokens, decode_tokens, generation_seconds, generation_tps, wait_seconds);
    return 0;
}