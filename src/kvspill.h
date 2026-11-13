#ifndef KVSPILL_H
#define KVSPILL_H

#include <stddef.h>
#include <stdint.h>

/*
 * SSD store for KV-cache entries that slid out of a slot's attention window.
 *
 * Tokens leave the window in order, so the spilled tokens are always one contiguous range of
 * absolute positions: [head, head + count). The store is a stack: kvspill_append() pushes the
 * newest evicted tokens, kvspill_pop() takes the most recent ones back (to refill a window that
 * grew, or to roll a request back). Nothing is held in RAM except a small chunk index.
 *
 * File layout: one chunk per append. A chunk holds, for every attention layer a = 0..layers-1,
 * a K block followed by a V block of n_alloc * KV_FLOATS_PER_TOKEN floats.
 */

#define KV_FLOATS_PER_TOKEN 512      /* 2 KV heads x 256 */

typedef struct {
    uint64_t off;                    /* byte offset of the chunk in the file */
    uint32_t n_alloc;                /* tokens the chunk was written with (block stride) */
    uint32_t n_valid;                /* leading tokens still in use */
} SpillChunk;

typedef struct KVSpill {
    char path[320];
    int fd;                          /* -1 until the first append */
    SpillChunk *chunks;
    size_t nchunks, cap_chunks;
    uint64_t count;                  /* tokens currently spilled */
    uint64_t head;                   /* tokens kept at the start of the window while count > 0 */
    uint64_t file_end;
} KVSpill;

void     kvspill_init(KVSpill *sp, const char *path);
/* Push n tokens. k[a] / v[a] point at n contiguous tokens of attention layer a. `head` is
 * recorded when the store is empty. 0 = ok, -1 = I/O error (nothing was recorded). */
int      kvspill_append(KVSpill *sp, int layers, size_t n, const float *const *k,
                        const float *const *v, uint64_t head);
/* Remove the n most recent tokens. With kdst/vdst they are copied out, oldest first, into
 * kdst[a] / vdst[a] (n contiguous tokens each); with NULL they are just dropped. */
int      kvspill_pop(KVSpill *sp, int layers, size_t n, float *const *kdst, float *const *vdst);
/* Copy a file (data, no metadata). 0 = ok. */
int      kvspill_copy_file(const char *src, const char *dst);
/* Write the spill to a pair of files: the data file and its chunk index. 0 = ok, -1 = error. */
int      kvspill_save(const KVSpill *sp, int layers, const char *data_path, const char *index_path);
/* Replace the contents of sp (its own path is reused for the data) with a saved pair. The files
 * are validated first; on any error sp is left empty. 0 = ok, -1 = error. */
int      kvspill_load(KVSpill *sp, int layers, const char *data_path, const char *index_path);
/* Close and delete the file and forget everything (the path stays set for reuse). */
void     kvspill_clear(KVSpill *sp);
uint64_t kvspill_bytes(const KVSpill *sp);

#endif
