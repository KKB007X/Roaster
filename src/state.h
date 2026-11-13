#ifndef STATE_H
#define STATE_H

#include <stdio.h>
#include <stddef.h>
#include "model.h"

/*
 * Save / restore / snapshot of one slot's *context*: the KV window of the full-attention layers,
 * the recurrent (Gated DeltaNet) state and the position. Model weights are never touched.
 * Tokens that slid out of the window live in the slot's SSD spill file (see kvspill.h) and are
 * not part of the state file.
 */

/* Write / read the context of an initialized model. 0 = ok, -1 = error (I/O, size or format mismatch).
 * model_state_load() also accepts a model with a different context length, as long as the saved
 * window fits into it. */
int model_state_save(const Model *m, FILE *f);
int model_state_load(Model *m, FILE *f);

/* Approximate RAM held by this slot's context (KV cache + recurrent state + work buffers). */
size_t model_ram_bytes(const Model *m);

/* In-RAM snapshot used to roll a slot back when a request is interrupted while the prompt is
 * still being processed. Costs about 65 MB for the lifetime of the snapshot. */
typedef struct ModelSnapshot ModelSnapshot;
ModelSnapshot *model_snapshot_take(const Model *m);               /* NULL if out of memory */
/* Rolls KV window (including anything that slid to SSD meanwhile), recurrent state and position
 * back. 0 = ok, -1 = failed (the context is then inconsistent: reset it). */
int            model_snapshot_restore(Model *m, const ModelSnapshot *s);
void           model_snapshot_free(ModelSnapshot *s);

#endif
