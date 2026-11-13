#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <omp.h>
#include <limits.h>
#include <sys/vfs.h>
#include "slots.h"
#include "kvwindow.h"
#include "state.h"
#include "protocol.h"

#define PROMPT_BUFFER_SIZE 16384
#define SLOT_FILE_MAGIC    0x544f4c53u      /* "SLOT" */

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
    c->slide          = DEFAULT_SLIDE;
    c->keep           = DEFAULT_KEEP;
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
    if (c->slide != 0 && c->slide != 1)
        return "slide must be 0 or 1.";
    if (c->keep < 0 || c->keep > 100000000)
        return "keep must be >= 0.";
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

static void set_err(char *err, size_t cap, const char *msg)
{
    if (err && cap)
        snprintf(err, cap, "%s", msg);
}

int slots_init(SlotManager *sm, Weights *weights, const Tokenizer *tokenizer, int max_slots,
               const char *state_dir)
{
    memset(sm, 0, sizeof(*sm));
    sm->slots = calloc((size_t)max_slots, sizeof(InferenceSlot));
    if (!sm->slots)
        return -1;
    sm->weights = weights;
    sm->tokenizer = tokenizer;
    sm->max_slots = max_slots;
    snprintf(sm->state_dir, sizeof sm->state_dir, "%s", state_dir && *state_dir ? state_dir : DEFAULT_STATE_DIR);
    if (mkdir(sm->state_dir, 0700) != 0 && errno != EEXIST)
        fprintf(stderr, "Warning: cannot create state dir %s: %s (unloading contexts will fail)\n",
                sm->state_dir, strerror(errno));
    {   /* slid-out context should go to SSD, not RAM */
        struct statfs sf;
        if (statfs(sm->state_dir, &sf) == 0 && (unsigned long)sf.f_type == 0x01021994UL)
            fprintf(stderr, "Warning: state dir %s is on tmpfs (RAM). Slid-out context would be kept "
                            "in memory; use --state-dir on a real SSD.\n", sm->state_dir);
    }
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

static void slot_remove_state_file(InferenceSlot *slot)
{
    if (slot->state_path[0])
        unlink(slot->state_path);
}

void slots_free(SlotManager *sm)
{
    if (sm->slots) {
        for (int i = 0; i < sm->max_slots; i++) {
            InferenceSlot *s = &sm->slots[i];
            if (!s->active)
                continue;
            if (s->loaded)
                model_free(&s->model);
            slot_remove_state_file(s);
            kvspill_clear(&s->spill);
            free(s->system);
        }
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

/* Allocate the model state of a slot from its config. */
static int slot_model_setup(SlotManager *sm, InferenceSlot *slot)
{
    const SlotConfig *config = &slot->config;
    if (model_init(&slot->model, sm->weights, config->context_length,
                   config->temperature, config->top_p, config->top_k) != 0)
        return -1;
    /* Routing configuration is per slot; expert weights remain shared. */
    slot->model.cache_route = config->cache_route != 0;
    slot->model.route_j = config->route_j;
    slot->model.route_m = config->route_m;
    slot->model.route_alpha = config->route_alpha;
    slot->model.route_p = config->route_p;
    /* Context sliding: the spill file belongs to the slot and survives unload/load. */
    slot->model.slide = config->slide != 0;
    slot->model.slide_keep = config->keep;
    slot->model.spill = &slot->spill;
    slot->model.slide_pin = slot->pinned;
    return 0;
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
    slot->config = *config;
    slot->pending = -1;
    snprintf(slot->state_path, sizeof slot->state_path, "%s/roaster-%ld-slot%d.state",
             sm->state_dir, (long)getpid(), id);
    snprintf(slot->spill_path, sizeof slot->spill_path, "%s/roaster-%ld-slot%d.kv",
             sm->state_dir, (long)getpid(), id);
    kvspill_init(&slot->spill, slot->spill_path);
    if (slot_model_setup(sm, slot) != 0) {
        fprintf(stderr, "Failed to initialize slot %d\n", id);
        memset(slot, 0, sizeof(*slot));
        return -1;
    }
    slot->loaded = 1;
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
    printf("  Sliding:     %s (keep first %d tokens)\n", config->slide ? "on" : "off", config->keep);
    return id;
}

static void slot_destroy_locked(SlotManager *sm, int id)
{
    if (!slot_is_active(sm, id)) {
        fprintf(stderr, "Invalid or inactive slot\n");
        return;
    }
    InferenceSlot *slot = &sm->slots[id];
    if (slot->loaded)
        model_free(&slot->model);
    slot_remove_state_file(slot);
    kvspill_clear(&slot->spill);             /* the slid-out context dies with the client */
    free(slot->system);
    memset(slot, 0, sizeof(*slot));
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

/* ============================================================ context control */

static InferenceSlot *idle_slot(SlotManager *sm, int id, char *err, size_t cap)
{
    if (!slot_is_active(sm, id)) {
        set_err(err, cap, "invalid or inactive slot");
        return NULL;
    }
    InferenceSlot *slot = &sm->slots[id];
    if (atomic_load(&slot->running)) {
        set_err(err, cap, "slot is busy with a running request");
        return NULL;
    }
    return slot;
}

int slot_update(SlotManager *sm, int id, const SlotConfig *s, char *err, size_t cap)
{
    InferenceSlot *slot = idle_slot(sm, id, err, cap);
    if (!slot)
        return -1;
    SlotConfig merged = slot->config;
    merged.temperature = s->temperature;
    merged.top_p = s->top_p;
    merged.top_k = s->top_k;
    merged.max_tokens = s->max_tokens;
    merged.think = s->think;
    const char *e = slot_config_check(&merged);
    if (e) {
        set_err(err, cap, e);
        return -1;
    }
    slot->config = merged;
    if (slot->loaded) {
        slot->model.temperature = merged.temperature;
        slot->model.top_p = merged.top_p;
        slot->model.top_k = merged.top_k;
    }
    return 0;
}

int slot_configure(SlotManager *sm, int id, const SlotConfig *nc, char *err, size_t cap)
{
    InferenceSlot *slot = idle_slot(sm, id, err, cap);
    if (!slot)
        return -1;
    const char *e = slot_config_check(nc);
    if (e) {
        set_err(err, cap, e);
        return -1;
    }
    const int ctx_change = nc->context_length != slot->config.context_length;
    if (slot->loaded) {
        Model *m = &slot->model;
        const int old_slide = m->slide, old_keep = m->slide_keep;
        m->slide = nc->slide != 0;
        m->slide_keep = nc->keep;
        if (ctx_change && model_set_context(m, nc->context_length) != 0) {
            m->slide = old_slide;
            m->slide_keep = old_keep;
            set_err(err, cap, "could not resize the context window (out of memory, SSD error, "
                              "or the window is not larger than the preserved head)");
            return -1;
        }
        m->cache_route = nc->cache_route != 0;
        m->route_j = nc->route_j;
        m->route_m = nc->route_m;
        m->route_alpha = nc->route_alpha;
        m->route_p = nc->route_p;
        m->temperature = nc->temperature;
        m->top_p = nc->top_p;
        m->top_k = nc->top_k;
    } else if (ctx_change) {
        set_err(err, cap, "the context is unloaded; load it before changing the context window");
        return -1;
    }
    slot->config = *nc;
    return 0;
}

int slot_set_system(SlotManager *sm, int id, const char *text, char *err, size_t cap)
{
    InferenceSlot *slot = idle_slot(sm, id, err, cap);
    if (!slot)
        return -1;
    char *copy = NULL;
    if (text && text[0]) {
        copy = strdup(text);
        if (!copy) {
            set_err(err, cap, "out of memory");
            return -1;
        }
    }
    free(slot->system);
    slot->system = copy;
    return 0;
}

int slot_reset(SlotManager *sm, int id, char *err, size_t cap)
{
    InferenceSlot *slot = idle_slot(sm, id, err, cap);
    if (!slot)
        return -1;
    if (slot->loaded)
        model_reset(&slot->model);
    slot->pending = -1;
    slot->broken = 0;
    slot->saved_position = 0;
    slot->saved_window = slot->pinned = 0;
    slot_remove_state_file(slot);
    kvspill_clear(&slot->spill);
    return 0;
}

typedef struct {
    uint32_t magic, version;
    int32_t pending;
    uint32_t reserved;
} SlotFileHeader;

static int save_state_file(const InferenceSlot *slot, const char *path)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0)
        return -1;
    FILE *f = fdopen(fd, "wb");
    if (!f) {
        close(fd);
        return -1;
    }
    SlotFileHeader h = { SLOT_FILE_MAGIC, 1, slot->pending, 0 };
    int ok = fwrite(&h, sizeof h, 1, f) == 1 && model_state_save(&slot->model, f) == 0;
    if (fclose(f) != 0)
        ok = 0;
    if (!ok)
        unlink(path);
    return ok ? 0 : -1;
}

static int save_slot_state(const InferenceSlot *slot)
{
    return save_state_file(slot, slot->state_path);
}

int slot_unload(SlotManager *sm, int id, int discard, char *err, size_t cap)
{
    InferenceSlot *slot = idle_slot(sm, id, err, cap);
    if (!slot)
        return -1;
    if (!slot->loaded) {
        if (discard) {                       /* already out of RAM: just forget the saved context */
            slot_remove_state_file(slot);
            kvspill_clear(&slot->spill);
            slot->pending = -1;
            slot->broken = 0;
            slot->saved_position = 0;
            slot->saved_window = slot->pinned = 0;
        }
        return 0;
    }
    if (discard) {
        slot_remove_state_file(slot);
        kvspill_clear(&slot->spill);
        slot->pending = -1;
        slot->broken = 0;
        slot->saved_position = 0;
        slot->saved_window = slot->pinned = 0;
    } else {
        if (save_slot_state(slot) != 0) {
            char m[160];
            snprintf(m, sizeof m, "could not save the context: %s (nothing was unloaded)",
                     errno ? strerror(errno) : "write error");
            set_err(err, cap, m);
            return -1;
        }
        slot->saved_position = slot->model.position;
        slot->saved_window = kvw_len(&slot->model);
    }
    model_free(&slot->model);
    slot->loaded = 0;
    return 0;
}

int slot_load(SlotManager *sm, int id, char *err, size_t cap)
{
    InferenceSlot *slot = idle_slot(sm, id, err, cap);
    if (!slot)
        return -1;
    if (slot->loaded)
        return 0;
    if (slot_model_setup(sm, slot) != 0) {
        set_err(err, cap, "could not allocate memory for the context");
        return -1;
    }
    FILE *f = fopen(slot->state_path, "rb");
    if (f) {
        SlotFileHeader h;
        int ok = fread(&h, sizeof h, 1, f) == 1 && h.magic == SLOT_FILE_MAGIC && h.version == 1 &&
                 model_state_load(&slot->model, f) == 0;
        fclose(f);
        if (!ok) {
            model_free(&slot->model);
            set_err(err, cap, "the saved context is unreadable or does not match this slot (reset it)");
            return -1;
        }
        slot->pending = h.pending;
        unlink(slot->state_path);            /* from now on the context lives in RAM only */
    } else {
        slot->pending = -1;                  /* nothing saved: start with an empty context */
        slot->pinned = 0;
        slot->model.slide_pin = 0;
        kvspill_clear(&slot->spill);
    }
    slot->loaded = 1;
    return 0;
}

/* ============================================================ save / restore a context to a directory */

#define CTX_META    "roaster-context.meta"
#define CTX_STATE   "state.bin"
#define CTX_SPILL   "spill.kv"
#define CTX_INDEX   "spill.idx"
#define CTX_VERSION 1

void slots_set_save_root(SlotManager *sm, const char *root)
{
    snprintf(sm->save_root, sizeof sm->save_root, "%s", root ? root : "");
    size_t n = strlen(sm->save_root);
    while (n > 1 && sm->save_root[n - 1] == '/')
        sm->save_root[--n] = '\0';
}

static int att_layer_count(void)
{
    int n = 0;
    for (int l = 0; l < MODEL_NUM_LAYERS; l++)
        if (model_is_attention_layer(l))
            n++;
    return n;
}

static void model_fingerprint(const SlotManager *sm, char *out, size_t cap)
{
    struct stat st;
    unsigned long long size = fstat(sm->weights->fd, &st) == 0 ? (unsigned long long)st.st_size : 0;
    snprintf(out, cap, "%llu-%llu", size, (unsigned long long)sm->weights->tensor_count);
}

/* "invalid path" errors are reported with that text (HTTP 400). */
static int resolve_dir(const SlotManager *sm, const char *in, char *out, size_t cap, char *err, size_t ecap)
{
    if (!in || !*in || strchr(in, '\n')) {
        set_err(err, ecap, "invalid path: a directory is required");
        return -1;
    }
    for (const char *p = in; *p;) {            /* refuse ".." components */
        while (*p == '/') p++;
        const char *e = p;
        while (*e && *e != '/') e++;
        if (e - p == 2 && p[0] == '.' && p[1] == '.') {
            set_err(err, ecap, "invalid path: '..' is not allowed");
            return -1;
        }
        p = e;
    }
    int n;
    if (sm->save_root[0]) {
        while (*in == '/') in++;
        n = *in ? snprintf(out, cap, "%s/%s", sm->save_root, in) : snprintf(out, cap, "%s", sm->save_root);
    } else {
        n = snprintf(out, cap, "%s", in);
    }
    if (n < 0 || (size_t)n >= cap - 64) {
        set_err(err, ecap, "invalid path: too long");
        return -1;
    }
    while (n > 1 && out[n - 1] == '/')
        out[--n] = '\0';
    return 0;
}

static int mkdir_p(const char *path)
{
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0700) != 0 && errno != EEXIST)
                return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0700) != 0 && errno != EEXIST)
        return -1;
    struct stat st;
    return stat(tmp, &st) == 0 && S_ISDIR(st.st_mode) ? 0 : -1;
}

static void escape_text(FILE *f, const char *s)
{
    for (; s && *s; s++) {
        if (*s == '\n') fputs("\\n", f);
        else if (*s == '\r') fputs("\\r", f);
        else if (*s == '\\') fputs("\\\\", f);
        else fputc(*s, f);
    }
}

static char *unescape_text(const char *s)
{
    char *o = malloc(strlen(s) + 1), *w = o;
    if (!o) return NULL;
    for (; *s; s++) {
        if (*s == '\\' && s[1]) {
            s++;
            *w++ = *s == 'n' ? '\n' : *s == 'r' ? '\r' : *s;
        } else {
            *w++ = *s;
        }
    }
    *w = '\0';
    return o;
}

typedef struct {
    int version, model_layers, att_layers;
    char model[64];
    size_t position, window, spilled, pinned, context;
    char *system;
} CtxMeta;

static int read_meta(const char *path, CtxMeta *m)
{
    memset(m, 0, sizeof *m);
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    int magic = 0;
    while ((len = getline(&line, &cap, f)) >= 0) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        if (sscanf(line, "roaster-context %d", &m->version) == 1) magic = 1;
        else if (!strncmp(line, "model=", 6)) snprintf(m->model, sizeof m->model, "%s", line + 6);
        else if (!strncmp(line, "model_layers=", 13)) m->model_layers = atoi(line + 13);
        else if (!strncmp(line, "att_layers=", 11)) m->att_layers = atoi(line + 11);
        else if (!strncmp(line, "position=", 9)) m->position = (size_t)strtoull(line + 9, NULL, 10);
        else if (!strncmp(line, "window=", 7)) m->window = (size_t)strtoull(line + 7, NULL, 10);
        else if (!strncmp(line, "spilled=", 8)) m->spilled = (size_t)strtoull(line + 8, NULL, 10);
        else if (!strncmp(line, "pinned=", 7)) m->pinned = (size_t)strtoull(line + 7, NULL, 10);
        else if (!strncmp(line, "context_length=", 15)) m->context = (size_t)strtoull(line + 15, NULL, 10);
        else if (!strncmp(line, "system=", 7)) { free(m->system); m->system = unescape_text(line + 7); }
    }
    free(line);
    fclose(f);
    if (!magic) {
        free(m->system);
        m->system = NULL;
        return -2;
    }
    return 0;
}

static long long file_size(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 ? (long long)st.st_size : 0;
}

int slot_save_dir(SlotManager *sm, int id, const char *path, size_t *bytes, char *err, size_t cap)
{
    InferenceSlot *slot = idle_slot(sm, id, err, cap);
    if (!slot)
        return -1;
    if (bytes)
        *bytes = 0;
    if (slot->broken) {
        set_err(err, cap, "the context is inconsistent after a failure; reset it first");
        return -1;
    }
    char dir[PATH_MAX], f_meta[PATH_MAX + 64], f_state[PATH_MAX + 64], f_spill[PATH_MAX + 64],
         f_index[PATH_MAX + 64], m[PATH_MAX + 320];
    if (resolve_dir(sm, path, dir, sizeof dir, err, cap) != 0)
        return -1;
    struct stat st;
    if (!slot->loaded && stat(slot->state_path, &st) != 0) {
        set_err(err, cap, "the context is empty: nothing to save");
        return -1;
    }
    if (mkdir_p(dir) != 0) {
        snprintf(m, sizeof m, "could not create the directory: %s", strerror(errno));
        set_err(err, cap, m);
        return -1;
    }
    snprintf(f_meta, sizeof f_meta, "%s/" CTX_META, dir);
    snprintf(f_state, sizeof f_state, "%s/" CTX_STATE, dir);
    snprintf(f_spill, sizeof f_spill, "%s/" CTX_SPILL, dir);
    snprintf(f_index, sizeof f_index, "%s/" CTX_INDEX, dir);
    unlink(f_meta);                             /* an interrupted save is never taken for a valid one */
    int rc = slot->loaded ? save_state_file(slot, f_state)
                          : kvspill_copy_file(slot->state_path, f_state);
    if (rc != 0) {
        snprintf(m, sizeof m, "could not write the context to %s: %s", dir, strerror(errno));
        set_err(err, cap, m);
        return -1;
    }
    int na = att_layer_count();
    if (slot->spill.count > 0) {
        if (kvspill_save(&slot->spill, na, f_spill, f_index) != 0) {
            snprintf(m, sizeof m, "could not write the spilled context to %s: %s", dir, strerror(errno));
            set_err(err, cap, m);
            unlink(f_state);
            return -1;
        }
    } else {
        unlink(f_spill);
        unlink(f_index);
    }
    FILE *f = fopen(f_meta, "w");
    char fp[64];
    model_fingerprint(sm, fp, sizeof fp);
    int ok = f != NULL;
    if (f) {
        fprintf(f, "roaster-context %d\nmodel=%s\nmodel_layers=%d\natt_layers=%d\n"
                   "position=%zu\nwindow=%zu\nspilled=%zu\npinned=%zu\ncontext_length=%zu\nsystem=",
                CTX_VERSION, fp, MODEL_NUM_LAYERS, na,
                slot->loaded ? slot->model.position : slot->saved_position,
                slot->loaded ? kvw_len(&slot->model) : slot->saved_window,
                (size_t)slot->spill.count, slot->pinned, slot->config.context_length);
        escape_text(f, slot->system);
        fputc('\n', f);
        ok = fclose(f) == 0;
    }
    if (!ok) {
        snprintf(m, sizeof m, "could not write the context metadata to %s: %s", dir, strerror(errno));
        set_err(err, cap, m);
        unlink(f_meta);
        return -1;
    }
    if (bytes)
        *bytes = (size_t)(file_size(f_state) + file_size(f_spill) + file_size(f_index) + file_size(f_meta));
    return 0;
}

int slot_restore_dir(SlotManager *sm, int id, const char *path, char *err, size_t cap)
{
    InferenceSlot *slot = idle_slot(sm, id, err, cap);
    if (!slot)
        return -1;
    if (!slot->loaded) {
        set_err(err, cap, "the context is unloaded; load it before restoring a saved context");
        return -1;
    }
    char dir[PATH_MAX], f_meta[PATH_MAX + 64], f_state[PATH_MAX + 64], f_spill[PATH_MAX + 64],
         f_index[PATH_MAX + 64], m[PATH_MAX + 400];
    if (resolve_dir(sm, path, dir, sizeof dir, err, cap) != 0)
        return -1;
    snprintf(f_meta, sizeof f_meta, "%s/" CTX_META, dir);
    snprintf(f_state, sizeof f_state, "%s/" CTX_STATE, dir);
    snprintf(f_spill, sizeof f_spill, "%s/" CTX_SPILL, dir);
    snprintf(f_index, sizeof f_index, "%s/" CTX_INDEX, dir);

    /* ---- validate everything before touching the live context ---- */
    CtxMeta mt;
    int rc = read_meta(f_meta, &mt);
    if (rc != 0) {
        snprintf(m, sizeof m, rc == -1 ? "no saved context found in %s" : "invalid saved context in %s", dir);
        set_err(err, cap, m);
        return -1;
    }
    char fp[64];
    model_fingerprint(sm, fp, sizeof fp);
    const int na = att_layer_count();
    if (mt.version != CTX_VERSION || strcmp(mt.model, fp) != 0 || mt.model_layers != MODEL_NUM_LAYERS ||
        mt.att_layers != na) {
        set_err(err, cap, "invalid saved context: it was made by a different model or Roaster version");
        free(mt.system);
        return -1;
    }
    if (file_size(f_state) <= 0 || mt.window > mt.position || mt.window + mt.spilled != mt.position ||
        (mt.spilled > 0 && (file_size(f_spill) <= 0 || file_size(f_index) <= 0))) {
        set_err(err, cap, "invalid saved context: files are missing or inconsistent");
        free(mt.system);
        return -1;
    }

    /* ---- the window is never smaller than the saved one ---- */
    Model *mod = &slot->model;
    const size_t cur = slot->config.context_length;
    const size_t target = mt.context > cur ? mt.context : cur;
    if (target > cur) {
        if (model_set_context(mod, target) != 0) {
            set_err(err, cap, "could not resize the context window for the saved context");
            free(mt.system);
            return -1;
        }
        slot->config.context_length = target;
    }

    /* ---- replace the context ---- */
    model_reset(mod);                           /* also empties the SSD spill and the pin */
    slot->pending = -1;
    slot->broken = 0;
    slot->pinned = 0;
    slot->saved_position = slot->saved_window = 0;
    slot_remove_state_file(slot);
    int ok = 1;
    int32_t pending = -1;
    if (mt.spilled > 0 && kvspill_load(&slot->spill, na, f_spill, f_index) != 0)
        ok = 0;
    if (ok) {
        FILE *f = fopen(f_state, "rb");
        SlotFileHeader h;
        ok = f && fread(&h, sizeof h, 1, f) == 1 && h.magic == SLOT_FILE_MAGIC && h.version == 1 &&
             model_state_load(mod, f) == 0 && mod->position == mt.position &&
             kvw_spilled(mod) == mt.spilled;
        if (f) {
            if (ok) pending = h.pending;
            fclose(f);
        }
    }
    if (!ok) {
        model_reset(mod);                       /* leave a clean, empty context behind */
        kvspill_clear(&slot->spill);
        snprintf(m, sizeof m, "could not restore the saved context from %s (the previous context was "
                              "discarded)", dir);
        set_err(err, cap, m);
        free(mt.system);
        return -1;
    }
    slot->pending = pending;
    slot->pinned = mod->slide_pin = mt.pinned;
    free(slot->system);
    slot->system = mt.system && mt.system[0] ? mt.system : NULL;
    if (!slot->system)
        free(mt.system);
    /* A window with free room takes back what the snapshot kept on SSD. */
    if (kvw_spilled(mod) > 0 && kvw_len(mod) < mod->context_length && kvw_refill(mod) != 0)
        fprintf(stderr, "Warning: could not bring spilled context back into the window\n");
    return 0;
}

int slot_info(const SlotManager *sm, int id, SlotInfo *o)
{
    if (!slot_is_active(sm, id))
        return -1;
    const InferenceSlot *slot = &sm->slots[id];
    memset(o, 0, sizeof *o);
    o->loaded = slot->loaded;
    o->broken = slot->broken;
    o->running = atomic_load(&slot->running);
    o->has_pending = slot->pending >= 0;
    o->context_length = slot->config.context_length;
    o->config = slot->config;
    o->spilled = (size_t)slot->spill.count;
    o->pinned = slot->pinned;
    o->spill_bytes = (size_t)kvspill_bytes(&slot->spill);
    if (slot->loaded) {
        o->position = slot->model.position;
        o->window = kvw_len(&slot->model);
        o->ram_bytes = model_ram_bytes(&slot->model);
    } else {
        o->position = slot->saved_position;
        o->window = slot->saved_window;
        struct stat st;
        if (stat(slot->state_path, &st) == 0)
            o->state_bytes = (size_t)st.st_size;
    }
    return 0;
}

int slot_stop(SlotManager *sm, int id)
{
    int hit = 0;
    pthread_mutex_lock(&sm->table_lock);
    if (slot_is_active(sm, id) && atomic_load(&sm->slots[id].running)) {
        atomic_store(&sm->slots[id].stop, 1);
        hit = 1;
    }
    pthread_mutex_unlock(&sm->table_lock);
    return hit;
}

/* ============================================================ running a request */

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

/* A whole chunk of prompt tokens (no logits) under ONE step-lock hold. Other slots wait for the
 * chunk, so the chunk size trades prefill speed against fairness: ROASTER_PREFILL_CHUNK (tokens,
 * default 256, max model_prefill_max_chunk()) lowers it, e.g. to 64 when several slots are busy. */
static int step_prefill_batch(SlotManager *sm, Model *m, const int32_t *tokens, size_t n, double *wait)
{
    double t0 = omp_get_wtime();
    fair_lock(&sm->step_lock);
    *wait += omp_get_wtime() - t0;
    int r = g_slots_abort ? MODEL_FORWARD_ERROR : model_prefill_batch(m, tokens, n);
    fair_unlock(&sm->step_lock);
    return r;
}

/* ROASTER_PREFILL=0 selects the old one-token-at-a-time prefill (for A/B comparisons). */
static int batched_prefill_enabled(void)
{
    const char *e = getenv("ROASTER_PREFILL");
    return !(e && e[0] == '0');
}

static size_t prefill_chunk_size(void)
{
    static size_t chunk;
    if (!chunk) {
        size_t max = (size_t)model_prefill_max_chunk(), v = max;
        const char *e = getenv("ROASTER_PREFILL_CHUNK");
        if (e && atol(e) > 0)
            v = (size_t)atol(e);
        chunk = v > max ? max : v;
    }
    return chunk;
}

/* Qwen3.x chat turn. With thinking on the assistant turn is opened with "<think>\n" and the model
 * reasons first; with thinking off an empty think block is pre-filled so it answers directly.
 * When continuing, the previous assistant turn is closed with "<|im_end|>\n" first.
 * Template text is tokenized with special-token parsing; the user's text never is, so typing
 * "<|im_end|>" in a prompt cannot inject control tokens. */
static size_t build_prompt_tokens(const Tokenizer *tok, const char *user, const char *system,
                                  int think, int continuing, int32_t **out, size_t *sys_tokens)
{
    const char *head = continuing ? "<|im_end|>\n<|im_start|>user\n" : "<|im_start|>user\n";
    const char *tail = think
        ? "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        : "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    int32_t *a = NULL, *b = NULL, *c = NULL, *s0 = NULL, *s1 = NULL, *s2 = NULL;
    size_t n0 = 0, n1 = 0, n2 = 0;
    *out = NULL;
    *sys_tokens = 0;
    /* A fresh conversation starts with the system turn; its tokens get pinned by the caller. */
    if (!continuing && system && system[0]) {
        n0 = tokenizer_encode_special(tok, "<|im_start|>system\n", &s0);
        n1 = tokenizer_encode(tok, system, &s1);
        n2 = tokenizer_encode_special(tok, "<|im_end|>\n", &s2);
        if (!s0 || !s1 || !s2) {
            free(s0); free(s1); free(s2);
            return 0;
        }
    }
    size_t na = tokenizer_encode_special(tok, head, &a);
    size_t nb = user[0] ? tokenizer_encode(tok, user, &b) : 0;
    size_t nc = tokenizer_encode_special(tok, tail, &c);
    if (!a || !c || (user[0] && !b)) {
        free(a); free(b); free(c); free(s0); free(s1); free(s2);
        return 0;
    }
    int32_t *all = malloc((n0 + n1 + n2 + na + nb + nc) * sizeof(int32_t));
    if (all) {
        int32_t *w = all;
        if (n0) { memcpy(w, s0, n0 * sizeof(int32_t)); w += n0; }
        if (n1) { memcpy(w, s1, n1 * sizeof(int32_t)); w += n1; }
        if (n2) { memcpy(w, s2, n2 * sizeof(int32_t)); w += n2; }
        memcpy(w, a, na * sizeof(int32_t)); w += na;
        if (nb) { memcpy(w, b, nb * sizeof(int32_t)); w += nb; }
        memcpy(w, c, nc * sizeof(int32_t));
        *out = all;
        *sys_tokens = n0 + n1 + n2;
    }
    free(a); free(b); free(c); free(s0); free(s1); free(s2);
    return all ? n0 + n1 + n2 + na + nb + nc : 0;
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

/* Undo a half-processed prompt so the context is exactly as it was before the request. */
static void rollback_prompt(InferenceSlot *slot, ModelSnapshot *snap, int continuing, int32_t old_pending)
{
    if (!continuing) {                       /* nothing to protect: the context was fresh anyway */
        model_reset(&slot->model);
        slot->pending = -1;
        slot->pinned = 0;
    } else if (snap) {
        if (model_snapshot_restore(&slot->model, snap) != 0)
            slot->broken = 1;                /* could not undo a slide (SSD error): needs a reset */
        slot->pending = old_pending;
    } else {
        slot->broken = 1;                    /* no snapshot (out of memory): needs a reset */
    }
}

enum { RUN_FRESH = 0, RUN_CONTINUE = 1 };

static int run_impl(SlotManager *sm, InferenceSlot *slot, const char *user_prompt, int mode,
                    FILE *out, FILE *info)
{
    const Tokenizer *tok = sm->tokenizer;
    Model *m = &slot->model;
    int max_tokens = slot->config.max_tokens;

    if (!slot->loaded) {
        fprintf(info, "Context is unloaded; load it first\n");
        return -1;
    }
    if (slot->broken) {
        fprintf(info, "Context was left inconsistent by an interrupted request; reset it first\n");
        return -1;
    }
    if (strlen(user_prompt) >= PROMPT_BUFFER_SIZE - 128) {
        fprintf(info, "Prompt is too long\n");
        return -1;
    }

    int continuing = mode == RUN_CONTINUE && (m->position > 0 || slot->pending >= 0);
    int32_t *tokens = NULL;
    size_t sys_tokens = 0;
    size_t token_count = build_prompt_tokens(tok, user_prompt, slot->system, slot->config.think, continuing,
                                             &tokens, &sys_tokens);
    if (!tokens || token_count == 0) {
        fprintf(info, "Prompt tokenization failed\n");
        free(tokens);
        return -1;
    }

    /* The token sampled at the end of the previous turn was never fed to the model: feed it first.
     * (A stop token is not fed; the turn is closed by the explicit <|im_end|> in the template.) */
    int feed_pending = continuing && slot->pending >= 0 && !tokenizer_is_stop(tok, slot->pending);
    size_t total = token_count + (feed_pending ? 1 : 0);
    size_t ctx = slot->config.context_length;
    size_t used = continuing ? kvw_len(m) : 0;      /* tokens resident in the attention window */
    /* With sliding on there is no hard limit: old KV entries slide out to SSD as needed. */
    if (!slot->config.slide &&
        (used > ctx || total > ctx - used || (size_t)max_tokens > ctx - used - total)) {
        if (continuing)
            fprintf(info, "Prompt plus requested generation exceeds slot context "
                          "(%zu of %zu tokens already in use; reset the context or use a larger one)\n",
                    used, ctx);
        else
            fprintf(info, "Prompt plus requested generation exceeds slot context\n");
        free(tokens);
        return -1;
    }
    int32_t *feed = tokens;
    if (feed_pending) {
        feed = malloc(total * sizeof(int32_t));
        if (!feed) {
            fprintf(info, "Out of memory\n");
            free(tokens);
            return -1;
        }
        feed[0] = slot->pending;
        memcpy(feed + 1, tokens, token_count * sizeof(int32_t));
        free(tokens);
    }

    if (!continuing) {                       /* fresh conversation: reset state, attention cache and position */
        model_reset(m);
        slot->pending = -1;
        slot->pinned = m->slide_pin = sys_tokens;   /* pin the system prompt before anything can slide */
    }
    int32_t old_pending = slot->pending;
    ModelSnapshot *snap = continuing ? model_snapshot_take(m) : NULL;

    /* Prefill: every prompt token; only the last one produces the first output token. */
    int32_t next_token = MODEL_FORWARD_ERROR;
    double wait_seconds = 0.0;
    int stopped_req = 0, stopped = 0, skip_gen = 0;
    double prefill_start = omp_get_wtime();
    /* Everything except the last few tokens goes through the batched prefill in chunks; the last token
     * must go through step_forward because it is the one that samples the first output token. */
    const size_t chunk = prefill_chunk_size();
    const size_t batch_min = batched_prefill_enabled() ? 8 : (size_t)-1;
    int used_batch = 0;
    const uint64_t req0 = sm->weights->stat_requests, load0 = sm->weights->stat_loads;
    size_t i = 0;
    while (i < total) {
        if (g_slots_abort) {
            free(feed);
            model_snapshot_free(snap);
            return -1;
        }
        if (atomic_load(&slot->stop)) {      /* stop while reading the prompt: roll it back */
            rollback_prompt(slot, snap, continuing, old_pending);
            stopped_req = skip_gen = 1;
            break;
        }
        size_t left = total - i;
        if (left > batch_min) {
            size_t n = left - 1 < chunk ? left - 1 : chunk;
            next_token = step_prefill_batch(sm, m, feed + i, n, &wait_seconds) == MODEL_FORWARD_OK
                         ? 0 : MODEL_FORWARD_ERROR;
            i += n;
            used_batch = 1;
        } else {
            next_token = step_forward(sm, m, feed[i], i + 1 < total, &wait_seconds);
            i++;
        }
        if (next_token == MODEL_FORWARD_ERROR) {
            if (!g_slots_abort) {
                rollback_prompt(slot, snap, continuing, old_pending);
                fprintf(info, "\nForward pass failed during prefill after %.3f seconds%s\n",
                        omp_get_wtime() - prefill_start,
                        slot->broken ? " (context must be reset)" : "");
            }
            free(feed);
            model_snapshot_free(snap);
            return -1;
        }
    }
    double prefill_seconds = omp_get_wtime() - prefill_start;
    if (used_batch) {                        /* decode should not carry prefill's large scratch around */
        fair_lock(&sm->step_lock);
        model_prefill_release();
        fair_unlock(&sm->step_lock);
    }
    const uint64_t req1 = sm->weights->stat_requests, load1 = sm->weights->stat_loads;
    free(feed);
    model_snapshot_free(snap);

    /* Decode timing starts once prefill has produced the first output token. */
    double generation_start = omp_get_wtime();
    size_t generated_tokens = 0, think_tokens = 0;
    enum { ST_ANSWER, ST_THINK, ST_TOOL } state = ST_ANSWER;
    int skip_nl = 0;
    if (!skip_gen && slot->config.think) {   /* the prompt already opened <think> */
        fputc(STREAM_THINK_BEGIN, out);
        state = ST_THINK;
    }
    for (int i = 0; !skip_gen && i < max_tokens; i++) {
        if (next_token < 0 || (uint64_t)next_token >= tok->vocab_size) {
            slot->broken = 1;
            fprintf(info, "\nInvalid generated token: %d (context must be reset)\n", next_token);
            return -1;
        }
        slot->pending = next_token;          /* sampled, not yet fed: lets the next turn carry on from here */
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
        if (ferror(out) || g_slots_abort)       /* client disconnected or server stopping: context stays valid */
            return -1;
        if (atomic_load(&slot->stop)) {         /* clean stop between two tokens */
            stopped_req = 1;
            break;
        }
        if (i + 1 < max_tokens) {
            next_token = step_forward(sm, m, next_token, 0, &wait_seconds);
            if (next_token == MODEL_FORWARD_ERROR) {
                if (g_slots_abort)
                    return -1;
                slot->broken = 1;               /* the step may have half-updated the layers */
                fprintf(info, "\nForward pass failed during generation after %.3f seconds "
                              "(context must be reset)\n", omp_get_wtime() - generation_start);
                return -1;
            }
        }
    }
    double generation_seconds = skip_gen ? 0.0 : omp_get_wtime() - generation_start;
    fputc('\n', out);

    double prefill_tps = prefill_seconds > 0.0 ? (double)total / prefill_seconds : 0.0;
    /* The first output token is computed during prefill, so it is not a decode token. */
    size_t decode_tokens = generated_tokens > 0 ? generated_tokens - 1 : 0;
    double generation_tps = generation_seconds > 0.0 ? (double)decode_tokens / generation_seconds : 0.0;

    if (stopped_req)
        fprintf(info, "\nStopped by request%s.\n",
                skip_gen ? " before generation started (the prompt was rolled back)" : "");
    else if (!stopped)
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
            "  Turn wait:      %.3f s (queued behind other slots)\n"
            "  Context:        %zu of %zu tokens in window (position %zu, %zu slid out to SSD)\n"
            "  Pinned prompt:  %zu tokens (system prompt, never slid out)\n"
            "  Prefill mode:   %s\n"
            "  Expert loads:   prefill %llu of %llu requests, decode %llu of %llu requests\n",
            total, prefill_seconds, prefill_tps,
            generated_tokens, think_tokens, decode_tokens, generation_seconds, generation_tps, wait_seconds,
            kvw_len(m), ctx, m->position, kvw_spilled(m), slot->pinned, used_batch ? "batched" : "per-token",
            (unsigned long long)(load1 - load0), (unsigned long long)(req1 - req0),
            (unsigned long long)(sm->weights->stat_loads - load1),
            (unsigned long long)(sm->weights->stat_requests - req1));
    return 0;
}

static int slot_run_mode(SlotManager *sm, int id, const char *user_prompt, int mode, FILE *out, FILE *info)
{
    if (!slot_is_active(sm, id)) {
        fprintf(info, "Invalid or inactive slot\n");
        return -1;
    }
    InferenceSlot *slot = &sm->slots[id];
    int expected = 0;
    if (!atomic_compare_exchange_strong(&slot->running, &expected, 1)) {
        fprintf(info, "Slot is busy with another request\n");
        return -1;
    }
    atomic_store(&slot->stop, 0);
    int rc = run_impl(sm, slot, user_prompt, mode, out, info);
    atomic_store(&slot->running, 0);
    return rc;
}

int slot_run(SlotManager *sm, int id, const char *user_prompt, FILE *out, FILE *info)
{
    return slot_run_mode(sm, id, user_prompt, RUN_FRESH, out, info);
}

int slot_continue(SlotManager *sm, int id, const char *user_prompt, FILE *out, FILE *info)
{
    return slot_run_mode(sm, id, user_prompt, RUN_CONTINUE, out, info);
}
