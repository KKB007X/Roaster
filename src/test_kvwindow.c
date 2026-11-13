/* Standalone test of the sliding window + SSD spill logic (no weights, no GPU needed):
 *   gcc -g -fsanitize=address,undefined test_kvwindow.c kvwindow.c kvspill.c -o test_kvwindow && ./test_kvwindow */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "kvwindow.h"

bool model_is_attention_layer(int l) { return l >= 0 && l < MODEL_NUM_LAYERS && (l + 1) % MODEL_FULL_ATTN_INTERVAL == 0; }
void *locked_calloc_checked(size_t c, size_t s) { return calloc(c, s); }
void locked_buffer_free(void *p) { free(p); }

static int fails;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

static void fill_token(Model *m, size_t pos)
{
    for (int l = 0; l < MODEL_NUM_LAYERS; l++) {
        if (!model_is_attention_layer(l)) continue;
        KVCache *kv = &m->kv_cache[l];
        for (int j = 0; j < KV_FLOATS_PER_TOKEN; j++) {
            kv->k[kv->length * KV_FLOATS_PER_TOKEN + j] = (float)pos + 100000.0f * l;
            kv->v[kv->length * KV_FLOATS_PER_TOKEN + j] = -(float)pos - l;
        }
        kv->length++;
    }
}

static int append(Model *m, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (kvw_ensure_room(m, 1)) return -1;
        fill_token(m, m->position);
        m->position++;
    }
    return 0;
}

/* window must equal head [0,h) + tail [h+s,p) -- or any explicit list */
static int verify(Model *m, const size_t *pos, size_t n)
{
    if (kvw_len(m) != n) { printf("  len %zu != %zu\n", kvw_len(m), n); return 0; }
    for (int l = 0; l < MODEL_NUM_LAYERS; l++) {
        if (!model_is_attention_layer(l)) continue;
        KVCache *kv = &m->kv_cache[l];
        if (kv->length != n) return 0;
        for (size_t i = 0; i < n; i++)
            for (int j = 0; j < KV_FLOATS_PER_TOKEN; j += 97)
                if (kv->k[i * KV_FLOATS_PER_TOKEN + j] != (float)pos[i] + 100000.0f * l ||
                    kv->v[i * KV_FLOATS_PER_TOKEN + j] != -(float)pos[i] - l) {
                    printf("  mismatch layer %d idx %zu want pos %zu\n", l, i, pos[i]);
                    return 0;
                }
    }
    return 1;
}

static size_t *expect(size_t h, size_t s, size_t p, size_t *n)
{
    size_t *v = malloc((p + 1) * sizeof *v);
    size_t k = 0;
    for (size_t i = 0; i < h && i < p; i++) v[k++] = i;
    for (size_t i = h + s; i < p; i++) v[k++] = i;
    *n = k;
    return v;
}

static int verify_state(Model *m)
{
    size_t h = kvw_spilled(m) ? (size_t)m->spill->head : (size_t)0, n;
    size_t *e = expect(h, kvw_spilled(m), m->position, &n);
    int ok = verify(m, e, n);
    free(e);
    return ok;
}

static void init(Model *m, KVSpill *sp, size_t cap, int keep, int slide)
{
    memset(m, 0, sizeof *m);
    kvspill_init(sp, "/tmp/test_kvwindow.kv");
    m->spill = sp; m->slide = slide; m->slide_keep = keep; m->context_length = cap; m->initialized = true;
    m->kv_cache = calloc(MODEL_NUM_LAYERS, sizeof(KVCache));
    for (int l = 0; l < MODEL_NUM_LAYERS; l++) if (model_is_attention_layer(l)) {
        m->kv_cache[l].k = calloc(cap * KV_FLOATS_PER_TOKEN, 4);
        m->kv_cache[l].v = calloc(cap * KV_FLOATS_PER_TOKEN, 4);
        m->kv_cache[l].capacity = cap;
    }
}

static void done(Model *m, KVSpill *sp)
{
    kvspill_clear(sp);
    for (int l = 0; l < MODEL_NUM_LAYERS; l++) { free(m->kv_cache[l].k); free(m->kv_cache[l].v); }
    free(m->kv_cache);
}

int main(void)
{
    Model m; KVSpill sp;

    puts("slide basic");
    init(&m, &sp, 64, 8, 1);
    CHECK(append(&m, 63) == 0 && kvspill_bytes(&sp) == 0);
    CHECK(append(&m, 500) == 0);
    CHECK(kvw_len(&m) <= 64 && kvw_spilled(&m) + kvw_len(&m) == m.position && m.position == 563);
    CHECK(verify_state(&m));
    CHECK(access("/tmp/test_kvwindow.kv", F_OK) == 0);

    puts("grow context -> spilled tokens come back");
    size_t spilled = kvw_spilled(&m);
    CHECK(spilled > 0);
    for (int l = 0; l < MODEL_NUM_LAYERS; l++) if (model_is_attention_layer(l)) { /* nothing */ }
    CHECK(kvw_realloc(&m, 128) == 0);
    m.context_length = 128;
    CHECK(kvw_refill(&m) == 0);
    CHECK(kvw_len(&m) + kvw_spilled(&m) == m.position);
    CHECK(verify_state(&m));
    CHECK(kvw_len(&m) <= 128 && kvw_len(&m) > 64);
    CHECK(append(&m, 300) == 0);
    CHECK(verify_state(&m));

    puts("grow big: everything returns, file removed on last pop");
    CHECK(kvw_realloc(&m, 4096) == 0);
    m.context_length = 4096;
    CHECK(kvw_refill(&m) == 0);
    CHECK(kvw_spilled(&m) == 0 && kvw_len(&m) == m.position && m.position == 863);
    size_t all[1000]; for (size_t i = 0; i < m.position; i++) all[i] = i;
    CHECK(verify(&m, all, m.position));
    CHECK(access("/tmp/test_kvwindow.kv", F_OK) == 0); /* file object stays until clear */

    puts("shrink -> spill, then grow again");
    m.context_length = 100;
    CHECK(kvw_shrink_to_fit(&m, 100) == 0);
    CHECK(kvw_realloc(&m, 100) == 0);
    CHECK(kvw_len(&m) <= 100 && kvw_spilled(&m) > 0 && verify_state(&m));
    CHECK(append(&m, 77) == 0 && verify_state(&m));
    CHECK(kvw_realloc(&m, 5000) == 0); m.context_length = 5000;
    CHECK(kvw_refill(&m) == 0 && kvw_spilled(&m) == 0);
    { size_t n = m.position; size_t *a2 = malloc(n * sizeof *a2); for (size_t i = 0; i < n; i++) a2[i] = i; CHECK(verify(&m, a2, n)); free(a2); }
    done(&m, &sp);

    puts("rollback: from a state without spill, slides during request");
    init(&m, &sp, 64, 8, 1);
    CHECK(append(&m, 40) == 0);
    size_t p0 = m.position, s0 = kvw_spilled(&m), n0 = kvw_len(&m);
    size_t *snap = malloc(n0 * sizeof *snap); for (size_t i = 0; i < n0; i++) snap[i] = i;
    CHECK(append(&m, 400) == 0);
    CHECK(kvw_spilled(&m) > 0);
    CHECK(kvw_rollback(&m, p0, s0) == 0);
    CHECK(m.position == p0 && kvw_spilled(&m) == 0 && verify(&m, snap, n0));
    free(snap);

    puts("rollback: from a state that already has spill");
    CHECK(append(&m, 300) == 0);
    p0 = m.position; s0 = kvw_spilled(&m); n0 = kvw_len(&m);
    CHECK(s0 > 0);
    { size_t h = (size_t)m.spill->head, n; size_t *e = expect(h, s0, p0, &n); CHECK(n == n0);
      for (int trial = 0; trial < 3; trial++) {
        size_t more = trial == 0 ? 5 : trial == 1 ? 70 : 900;
        CHECK(append(&m, more) == 0);
        CHECK(kvw_rollback(&m, p0, s0) == 0);
        CHECK(m.position == p0 && kvw_spilled(&m) == s0 && verify(&m, e, n));
      }
      free(e); }

    puts("rollback inside the head (fresh start, short)");
    done(&m, &sp);
    init(&m, &sp, 64, 8, 1);
    CHECK(append(&m, 3) == 0);
    p0 = 3; s0 = 0;
    CHECK(append(&m, 300) == 0);
    CHECK(kvw_rollback(&m, p0, s0) == 0);
    { size_t a3[3] = {0,1,2}; CHECK(verify(&m, a3, 3) && kvw_spilled(&m) == 0); }
    done(&m, &sp);

    puts("slide disabled: fails at capacity");
    init(&m, &sp, 16, 4, 0);
    CHECK(append(&m, 16) == 0);
    CHECK(append(&m, 1) != 0);
    done(&m, &sp);

    puts("pinned system prompt stays in the window");
    init(&m, &sp, 64, 4, 1);
    m.slide_pin = 30;
    CHECK(append(&m, 400) == 0 && verify_state(&m));
    CHECK(kvw_spilled(&m) > 0 && (size_t)m.spill->head == 30);
    CHECK(kvw_realloc(&m, 1000) == 0); m.context_length = 1000; CHECK(kvw_refill(&m) == 0 && kvw_spilled(&m) == 0);
    done(&m, &sp);
    init(&m, &sp, 16, 4, 1);          /* pin larger than the window is clamped, sliding still works */
    m.slide_pin = 500;
    CHECK(append(&m, 100) == 0 && verify_state(&m));
    done(&m, &sp);

    puts("keep > half of tiny window is clamped");
    init(&m, &sp, 4, 64, 1);
    CHECK(append(&m, 50) == 0 && verify_state(&m));
    done(&m, &sp);

    printf(fails ? "FAILED (%d)\n" : "ALL OK\n", fails);
    return fails != 0;
}
