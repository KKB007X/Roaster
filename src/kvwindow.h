#ifndef KVWINDOW_H
#define KVWINDOW_H

#include "model.h"

#define KVW_DEFAULT_KEEP 64

/* Page-aligned, mlocked zeroed buffers (defined in model.c). */
void *locked_calloc_checked(size_t count, size_t size);
void  locked_buffer_free(void *data);

size_t kvw_len(const Model *m);        /* tokens in the attention window */
size_t kvw_spilled(const Model *m);    /* tokens on SSD */
size_t kvw_head(const Model *m);       /* tokens that are never evicted */
/* Largest number of tokens that can be appended in one go. */
size_t kvw_max_chunk(const Model *m);

/* Make room for `need` more tokens, sliding the window if enabled. 0 = ok. */
int kvw_ensure_room(Model *m, size_t need);
/* Move the n oldest window tokens after the head to SSD. */
int kvw_evict(Model *m, size_t n);
/* Bring the n most recently spilled tokens back into the window (right after the head). */
int kvw_restore(Model *m, size_t n);
/* Return the window to the state it had when position was p0 and s0 tokens were spilled
 * (undoes appends and slides). Sets m->position = p0. */
int kvw_rollback(Model *m, size_t p0, size_t s0);

/* Used by model_set_context(). */
int kvw_shrink_to_fit(Model *m, size_t new_ctx);
int kvw_realloc(Model *m, size_t new_ctx);
int kvw_refill(Model *m);

#endif
