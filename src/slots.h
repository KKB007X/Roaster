#ifndef SLOTS_H
#define SLOTS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <signal.h>
#include <stdatomic.h>
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
#define DEFAULT_SLIDE          1      /* slide the attention window instead of failing when full */
#define DEFAULT_KEEP           64     /* tokens at the start that are never slid out */
/* Unloaded contexts and slid-out context (KV spill) live here: use a real SSD, not tmpfs. */
#define DEFAULT_STATE_DIR      "/var/tmp/roaster-state"

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
    int slide;          /* 1 = when the window is full, slide it and spill old KV to SSD */
    int keep;           /* tokens at the start of the context that are never slid out */
} SlotConfig;

typedef struct {
    int active;
    int loaded;                 /* 1 = context (KV cache, recurrent state) is in RAM, 0 = unloaded */
    int broken;                 /* a failure left the context inconsistent: reset it */
    int32_t pending;            /* last sampled token not yet fed to the model, -1 = none */
    size_t saved_position;      /* position of the context while it is unloaded */
    size_t saved_window;        /* window tokens of the context while it is unloaded */
    char *system;               /* system prompt for the next fresh conversation (malloc) */
    size_t pinned;              /* tokens of the system prompt pinned in the window */
    char spill_path[320];       /* SSD file for tokens that slid out of the window */
    KVSpill spill;              /* lives as long as the slot (survives unload/load) */
    char state_path[320];       /* where `unload` saves the context */
    atomic_int running;         /* a request is executing on this slot */
    atomic_int stop;            /* a stop was requested for the running request */
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
    char state_dir[256];          /* unloaded contexts are saved here */
    char save_root[256];          /* if set, save/restore paths are confined to this directory */
    pthread_mutex_t table_lock;   /* slot_create / slot_destroy / slot_stop */
    FairLock step_lock;           /* one model step (token) at a time, in round-robin order */
} SlotManager;

typedef struct {
    int loaded, broken, running, has_pending;
    size_t position;              /* absolute position = total tokens in the conversation */
    size_t window;                /* tokens in the attention window (RAM) */
    size_t spilled;               /* tokens that slid out to SSD */
    size_t spill_bytes;           /* size of the spill file */
    size_t pinned;                /* system-prompt tokens that are never slid out */
    SlotConfig config;
    size_t context_length;
    size_t ram_bytes;             /* RAM held by the context (0 while unloaded) */
    size_t state_bytes;           /* size of the saved context file (0 if none) */
} SlotInfo;

void slot_config_defaults(SlotConfig *config);

/* NULL if the config is valid, otherwise a message describing the problem. */
const char *slot_config_check(const SlotConfig *config);

int  slots_init(SlotManager *sm, Weights *weights, const Tokenizer *tokenizer, int max_slots,
                const char *state_dir);
void slots_free(SlotManager *sm);

int  slot_create(SlotManager *sm, const SlotConfig *config);   /* returns slot id, or -1 */
void slot_destroy(SlotManager *sm, int id);
int  slot_is_active(const SlotManager *sm, int id);

/* Set (e.g. from a signal handler) to make a running slot_run() stop as soon as possible. */
extern volatile sig_atomic_t g_slots_abort;

/* Thread-safe: any number of threads may call slot_run / slot_continue on different slots at
 * once. They take turns at token granularity (one prefill/decode token each, round robin).
 * The reply is streamed to `out`; errors and the performance summary go to `info`. Both stop
 * early if writing to `out` fails (client went away); the context stays usable.
 *
 * slot_run       starts a fresh conversation (the old context is discarded).
 * slot_continue  appends a user turn to the existing context and answers it; with an empty
 *                context it behaves like slot_run. If the context is too full the request fails
 *                with "exceeds slot context" and nothing changes. */
int  slot_run(SlotManager *sm, int id, const char *user_prompt, FILE *out, FILE *info);
int  slot_continue(SlotManager *sm, int id, const char *user_prompt, FILE *out, FILE *info);

/* Context control. Call these from the thread that owns the slot (never while it is running);
 * they return 0 on success, otherwise -1 and a message in `err`. */
int  slot_update(SlotManager *sm, int id, const SlotConfig *sampling, char *err, size_t cap);
/* Replace the whole configuration of an idle slot: context window, routing, sampling, sliding.
 * Growing the window brings slid-out context back from SSD; shrinking it pushes the oldest out.
 * The context window of an unloaded slot cannot be changed (load it first). */
int  slot_configure(SlotManager *sm, int id, const SlotConfig *config, char *err, size_t cap);
/* Set (or clear with "") the system prompt used when a fresh conversation starts. It is wrapped in a
 * system turn and PINNED: it stays in the attention window when the window slides. It does not
 * touch the running context (reset it to apply the change). */
int  slot_set_system(SlotManager *sm, int id, const char *text, char *err, size_t cap);
int  slot_reset(SlotManager *sm, int id, char *err, size_t cap);     /* empty context, drops saved state */

/* Confine slot_save_dir / slot_restore_dir to `root` (paths become relative to it, ".." refused). */
void slots_set_save_root(SlotManager *sm, const char *root);
/* Write the whole context of an idle slot (attention window, recurrent state, position, the part
 * that slid out to SSD, system prompt) into a directory, creating it. An older snapshot in that
 * directory is replaced. Works for loaded and unloaded slots. *bytes = size written. */
int  slot_save_dir(SlotManager *sm, int id, const char *path, size_t *bytes, char *err, size_t cap);
/* Replace the context of an idle, loaded slot with a snapshot written by slot_save_dir(). The
 * window becomes at least as large as the saved one; anything that fits more is brought back
 * from the snapshot's SSD part. Settings (sampling, routing, ...) are kept. */
int  slot_restore_dir(SlotManager *sm, int id, const char *path, char *err, size_t cap);
int  slot_unload(SlotManager *sm, int id, int discard, char *err, size_t cap);
int  slot_load(SlotManager *sm, int id, char *err, size_t cap);
int  slot_info(const SlotManager *sm, int id, SlotInfo *out);

/* Thread-safe, callable from any thread: ask the running request on this slot to finish after the
 * current token. The partial answer stays in the context. Returns 1 if a request was running. */
int  slot_stop(SlotManager *sm, int id);

#endif
