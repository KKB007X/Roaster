#include "kvwindow.h"
#include <stdio.h>
#include <string.h>

#define MAX_ATT 64
#define TOK KV_FLOATS_PER_TOKEN

static int att_layers(const Model *m, int *idx)
{
    int n = 0;
    if (!m->kv_cache)
        return 0;
    for (int l = 0; l < MODEL_NUM_LAYERS && n < MAX_ATT; l++)
        if (model_is_attention_layer(l) && m->kv_cache[l].k)
            idx[n++] = l;
    return n;
}

size_t kvw_len(const Model *m)
{
    if (!m->kv_cache)
        return 0;
    for (int l = 0; l < MODEL_NUM_LAYERS; l++)
        if (model_is_attention_layer(l) && m->kv_cache[l].k)
            return m->kv_cache[l].length;
    return 0;
}

size_t kvw_spilled(const Model *m)
{
    return m->spill ? (size_t)m->spill->count : 0;
}

static size_t head_for(const Model *m, size_t cap)
{
    if (m->spill && m->spill->count > 0)
        return (size_t)m->spill->head;           /* fixed while anything is spilled */
    size_t h = m->slide_keep > 0 ? (size_t)m->slide_keep : 0;
    if (h > cap / 2)
        h = cap / 2;
    /* A pinned system prompt may take more of the window, but always leave room to slide. */
    size_t pin = m->slide_pin, room = cap / 8 > 1 ? cap / 8 : 1;
    if (pin > cap - room)
        pin = cap > room ? cap - room : 0;
    if (pin > h)
        h = pin;
    size_t len = kvw_len(m);
    return h > len ? len : h;
}

size_t kvw_head(const Model *m)
{
    return head_for(m, m->context_length);
}

size_t kvw_max_chunk(const Model *m)
{
    size_t cap = m->context_length, len = kvw_len(m);
    if (m->slide && m->spill)
        return cap - kvw_head(m);
    return cap > len ? cap - len : 0;
}

static int evict_at(Model *m, size_t h, size_t n)
{
    if (n == 0)
        return 0;
    if (!m->spill)
        return -1;
    int idx[MAX_ATT];
    int na = att_layers(m, idx);
    size_t len = kvw_len(m);
    if (na == 0 || h > len || n > len - h)
        return -1;
    const float *kp[MAX_ATT], *vp[MAX_ATT];
    for (int a = 0; a < na; a++) {
        kp[a] = m->kv_cache[idx[a]].k + h * TOK;
        vp[a] = m->kv_cache[idx[a]].v + h * TOK;
    }
    if (kvspill_append(m->spill, na, n, kp, vp, h) != 0)
        return -1;
    const size_t tail = len - h - n;
    for (int a = 0; a < na; a++) {
        KVCache *kv = &m->kv_cache[idx[a]];
        memmove(kv->k + h * TOK, kv->k + (h + n) * TOK, tail * TOK * sizeof(float));
        memmove(kv->v + h * TOK, kv->v + (h + n) * TOK, tail * TOK * sizeof(float));
        kv->length = len - n;
    }
    return 0;
}

int kvw_evict(Model *m, size_t n)
{
    return evict_at(m, kvw_head(m), n);
}

int kvw_ensure_room(Model *m, size_t need)
{
    const size_t cap = m->context_length, len = kvw_len(m);
    if (len + need <= cap)
        return 0;
    if (!m->slide || !m->spill)
        return -1;
    const size_t h = kvw_head(m);
    if (need > cap - h)
        return -1;
    const size_t over = len + need - cap;
    const size_t block = (cap - h) / 4;          /* slide in blocks, not one token at a time */
    size_t n = over > block ? over : block;
    if (n > len - h)
        n = len - h;
    if (n < over)
        return -1;
    return evict_at(m, h, n);
}

int kvw_restore(Model *m, size_t n)
{
    if (n == 0)
        return 0;
    if (!m->spill || n > m->spill->count)
        return -1;
    int idx[MAX_ATT];
    int na = att_layers(m, idx);
    const size_t len = kvw_len(m), h = (size_t)m->spill->head;
    if (na == 0 || h > len || len + n > m->kv_cache[idx[0]].capacity)
        return -1;
    const size_t tail = len - h;
    float *kd[MAX_ATT], *vd[MAX_ATT];
    for (int a = 0; a < na; a++) {
        KVCache *kv = &m->kv_cache[idx[a]];
        memmove(kv->k + (h + n) * TOK, kv->k + h * TOK, tail * TOK * sizeof(float));
        memmove(kv->v + (h + n) * TOK, kv->v + h * TOK, tail * TOK * sizeof(float));
        kd[a] = kv->k + h * TOK;
        vd[a] = kv->v + h * TOK;
    }
    if (kvspill_pop(m->spill, na, n, kd, vd) != 0) {
        for (int a = 0; a < na; a++) {            /* undo the shift */
            KVCache *kv = &m->kv_cache[idx[a]];
            memmove(kv->k + h * TOK, kv->k + (h + n) * TOK, tail * TOK * sizeof(float));
            memmove(kv->v + h * TOK, kv->v + (h + n) * TOK, tail * TOK * sizeof(float));
        }
        return -1;
    }
    for (int a = 0; a < na; a++)
        m->kv_cache[idx[a]].length = len + n;
    return 0;
}

/*
 * Absolute positions: head [0,h), spilled [h,h+s), window tail [h+s,p).
 * Target (state at the snapshot): every position < p0 except the ones spilled then, [h,h+s0).
 */
int kvw_rollback(Model *m, size_t p0, size_t s0)
{
    const size_t p = m->position, s = kvw_spilled(m);
    int idx[MAX_ATT];
    int na = att_layers(m, idx);
    if (na == 0 || p0 > p || s < s0)
        return -1;
    const size_t cap = m->kv_cache[idx[0]].capacity;
    if (s == 0) {                                 /* never slid: just drop the newer tokens */
        if (p0 > cap)
            return -1;
        for (int a = 0; a < na; a++)
            m->kv_cache[idx[a]].length = p0;
        m->position = p0;
        return 0;
    }
    const size_t h = (size_t)m->spill->head, sp_end = h + s;
    const size_t tk = p0 > sp_end ? p0 - sp_end : 0;          /* tail tokens to keep */
    const size_t hk = p0 < h ? p0 : h;                        /* head tokens to keep */
    const size_t lim = p0 > h ? p0 : h;
    const size_t dc = sp_end > lim ? sp_end - lim : 0;        /* spilled tokens newer than p0 */
    const size_t top = p0 < sp_end ? p0 : sp_end;
    const size_t r = top > h + s0 ? top - (h + s0) : 0;       /* spilled tokens to bring back */
    if (dc + r > s || hk + r + tk > cap)
        return -1;
    if (dc && kvspill_pop(m->spill, na, dc, NULL, NULL) != 0)
        return -1;
    if (hk < h) {                                 /* p0 lies inside the head: no spill existed then */
        for (int a = 0; a < na; a++)
            m->kv_cache[idx[a]].length = hk;
        m->position = p0;
        return 0;
    }
    float *kd[MAX_ATT], *vd[MAX_ATT];
    for (int a = 0; a < na; a++) {
        KVCache *kv = &m->kv_cache[idx[a]];
        if (r && tk) {
            memmove(kv->k + (h + r) * TOK, kv->k + h * TOK, tk * TOK * sizeof(float));
            memmove(kv->v + (h + r) * TOK, kv->v + h * TOK, tk * TOK * sizeof(float));
        }
        kd[a] = kv->k + h * TOK;
        vd[a] = kv->v + h * TOK;
    }
    if (r && kvspill_pop(m->spill, na, r, kd, vd) != 0)
        return -1;
    for (int a = 0; a < na; a++)
        m->kv_cache[idx[a]].length = hk + r + tk;
    m->position = p0;
    return 0;
}

int kvw_shrink_to_fit(Model *m, size_t new_ctx)
{
    const size_t len = kvw_len(m);
    if (len <= new_ctx)
        return 0;
    const size_t h = head_for(m, new_ctx);
    if (h >= new_ctx)
        return -1;
    const size_t need = len - new_ctx;
    size_t n = need + (new_ctx - h) / 4;
    if (n > len - h)
        n = len - h;
    if (n < need)
        return -1;
    return evict_at(m, h, n);
}

int kvw_realloc(Model *m, size_t new_ctx)
{
    int idx[MAX_ATT];
    int na = att_layers(m, idx);
    const size_t len = kvw_len(m);
    if (na == 0 || new_ctx == 0 || len > new_ctx || new_ctx > ((size_t)-1) / TOK / sizeof(float))
        return -1;
    float *nk[MAX_ATT] = {0}, *nv[MAX_ATT] = {0};
    for (int a = 0; a < na; a++) {
        nk[a] = locked_calloc_checked(new_ctx * TOK, sizeof(float));
        nv[a] = locked_calloc_checked(new_ctx * TOK, sizeof(float));
        if (!nk[a] || !nv[a]) {
            for (int b = 0; b <= a; b++) {
                locked_buffer_free(nk[b]);
                locked_buffer_free(nv[b]);
            }
            return -1;
        }
    }
    for (int a = 0; a < na; a++) {
        KVCache *kv = &m->kv_cache[idx[a]];
        memcpy(nk[a], kv->k, len * TOK * sizeof(float));
        memcpy(nv[a], kv->v, len * TOK * sizeof(float));
        locked_buffer_free(kv->k);
        locked_buffer_free(kv->v);
        kv->k = nk[a];
        kv->v = nv[a];
        kv->capacity = new_ctx;
    }
    return 0;
}

/* After the window grew: bring spilled tokens back, leaving one slide block of headroom if not
 * everything fits (so the next token does not immediately push them out again). */
int kvw_refill(Model *m)
{
    const size_t s = kvw_spilled(m), cap = m->context_length, len = kvw_len(m);
    if (!s || len >= cap)
        return 0;
    const size_t avail = cap - len;
    size_t r = s;
    if (r > avail) {
        size_t block = cap > (size_t)m->spill->head ? (cap - (size_t)m->spill->head) / 4 : 0;
        r = avail > block ? avail - block : 0;
    }
    return r ? kvw_restore(m, r) : 0;
}
