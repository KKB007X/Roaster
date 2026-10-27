#ifndef SLOTS_H
#define SLOTS_H

#include <stddef.h>
#include <stdio.h>
#include <signal.h>
#include <pthread.h>
#include "tokenizer.h"
#include "weights.h"
#include "model.h"

/* Defaults for `slot create` */
#define DEFAULT_CONTEXT        4096
#define DEFAULT_CACHE_ROUTE    0
#define DEFAULT_ROUTE_J        2
#define DEFAULT_ROUTE_M        12
#define DEFAULT_ROUTE_ALPHA    1.0f
#define DEFAULT_ROUTE_P        0.0f
#define DEFAULT_TEMPERATURE    0.8f
#define DEFAULT_TOP_P          0.95f
#define DEFAULT_TOP_K          20
#define DEFAULT_MAX_TOKENS     2048   /* thinking models need room for the reasoning */
#define DEFAULT_THINK          1

/* Per-slot settings. Weights, expert cache and tokenizer are shared by all slots. */
typedef struct {
    size_t context_length;
    int cache_route;
    int route_j;
    int route_m;
    float route_alpha;
    float route_p;
    float temperature;
    float top_p;
    int top_k;
    int max_tokens;
    int think;          /* 1 = reasoning mode (<think>), 0 = empty think block */
} SlotConfig;

typedef struct {
    int active;
    SlotConfig config;
    Model model;
} InferenceSlot;

/* FIFO ticket lock: waiters are served strictly in arrival order, so slots that re-queue after
 * every token take turns (round robin) instead of one slot barging in repeatedly. */
typedef struct {
    pthread_mutex_t m;
    pthread_cond_t c;
    unsigned long next, serving;
} FairLock;

typedef struct {
    Weights *weights;
    const Tokenizer *tokenizer;
    InferenceSlot *slots;
    int max_slots;
    pthread_mutex_t table_lock;   /* slot_create / slot_destroy */
    FairLock step_lock;           /* one model step (token) at a time, in round-robin order */
} SlotManager;

void slot_config_defaults(SlotConfig *config);

/* NULL if the config is valid, otherwise a message describing the problem. */
const char *slot_config_check(const SlotConfig *config);

int  slots_init(SlotManager *sm, Weights *weights, const Tokenizer *tokenizer, int max_slots);
void slots_free(SlotManager *sm);

int  slot_create(SlotManager *sm, const SlotConfig *config);   /* returns slot id, or -1 */
void slot_destroy(SlotManager *sm, int id);
int  slot_is_active(const SlotManager *sm, int id);

/* Set (e.g. from a signal handler) to make a running slot_run() stop as soon as possible. */
extern volatile sig_atomic_t g_slots_abort;

/* Thread-safe: any number of threads may call slot_run on different slots at once. They take turns
 * at token granularity (one prefill/decode token each, round robin).
 * Run one prompt on a slot (fresh conversation). The reply is streamed to `out`; errors and the
 * performance summary go to `info`. Stops early if writing to `out` fails (client went away). */
int  slot_run(SlotManager *sm, int id, const char *user_prompt, FILE *out, FILE *info);

#endif