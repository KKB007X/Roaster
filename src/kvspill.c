#define _GNU_SOURCE
#include "kvspill.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#define TOKB (KV_FLOATS_PER_TOKEN * sizeof(float))

static int pwrite_all(int fd, const void *buf, size_t n, uint64_t off)
{
    const char *p = buf;
    while (n > 0) {
        ssize_t w = pwrite(fd, p, n, (off_t)off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += w; off += (uint64_t)w; n -= (size_t)w;
    }
    return 0;
}

static int pread_all(int fd, void *buf, size_t n, uint64_t off)
{
    char *p = buf;
    while (n > 0) {
        ssize_t r = pread(fd, p, n, (off_t)off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) return -1;
        p += r; off += (uint64_t)r; n -= (size_t)r;
    }
    return 0;
}

void kvspill_init(KVSpill *sp, const char *path)
{
    memset(sp, 0, sizeof *sp);
    sp->fd = -1;
    snprintf(sp->path, sizeof sp->path, "%s", path ? path : "");
}

uint64_t kvspill_bytes(const KVSpill *sp)
{
    return sp ? sp->file_end : 0;
}

void kvspill_clear(KVSpill *sp)
{
    if (!sp) return;
    if (sp->fd >= 0) {
        close(sp->fd);
        sp->fd = -1;
        if (sp->path[0]) unlink(sp->path);
    }
    free(sp->chunks);
    sp->chunks = NULL;
    sp->nchunks = sp->cap_chunks = 0;
    sp->count = sp->head = sp->file_end = 0;
}

static int spill_open(KVSpill *sp)
{
    if (sp->fd >= 0) return 0;
    if (!sp->path[0]) return -1;
    sp->fd = open(sp->path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    sp->file_end = 0;
    return sp->fd >= 0 ? 0 : -1;
}

int kvspill_append(KVSpill *sp, int layers, size_t n, const float *const *k,
                   const float *const *v, uint64_t head)
{
    if (n == 0) return 0;
    if (!sp || layers <= 0 || n > UINT32_MAX) return -1;
    if (spill_open(sp) != 0) {
        fprintf(stderr, "Roaster: cannot open spill file %s: %s\n", sp->path, strerror(errno));
        return -1;
    }
    if (sp->nchunks == sp->cap_chunks) {
        size_t nc = sp->cap_chunks ? sp->cap_chunks * 2 : 16;
        SpillChunk *t = realloc(sp->chunks, nc * sizeof *t);
        if (!t) return -1;
        sp->chunks = t;
        sp->cap_chunks = nc;
    }
    const size_t blk = n * TOKB;
    const uint64_t off = sp->file_end;
    for (int a = 0; a < layers; a++) {
        if (pwrite_all(sp->fd, k[a], blk, off + (uint64_t)(2 * a) * blk) != 0 ||
            pwrite_all(sp->fd, v[a], blk, off + (uint64_t)(2 * a + 1) * blk) != 0) {
            fprintf(stderr, "Roaster: spill write failed (%s): %s\n", sp->path, strerror(errno));
            if (ftruncate(sp->fd, (off_t)off) != 0) { /* best effort */ }
            return -1;
        }
    }
    sp->chunks[sp->nchunks++] = (SpillChunk){ off, (uint32_t)n, (uint32_t)n };
    sp->file_end = off + (uint64_t)2 * (uint64_t)layers * blk;
    if (sp->count == 0) sp->head = head;
    sp->count += n;
    return 0;
}

int kvspill_pop(KVSpill *sp, int layers, size_t n, float *const *kdst, float *const *vdst)
{
    if (n == 0) return 0;
    if (!sp || n > sp->count || sp->fd < 0) return -1;

    /* Phase 1: copy the data out (state untouched, so an I/O error changes nothing). */
    if (kdst && vdst) {
        size_t left = n, ci = sp->nchunks;
        while (left) {
            const SpillChunk *c = &sp->chunks[--ci];
            size_t r = left < c->n_valid ? left : c->n_valid;
            for (int a = 0; a < layers; a++) {
                uint64_t ko = c->off + ((uint64_t)(2 * a) * c->n_alloc + (c->n_valid - r)) * TOKB;
                uint64_t vo = c->off + ((uint64_t)(2 * a + 1) * c->n_alloc + (c->n_valid - r)) * TOKB;
                if (pread_all(sp->fd, kdst[a] + (left - r) * KV_FLOATS_PER_TOKEN, r * TOKB, ko) != 0 ||
                    pread_all(sp->fd, vdst[a] + (left - r) * KV_FLOATS_PER_TOKEN, r * TOKB, vo) != 0)
                    return -1;
            }
            left -= r;
        }
    }
    /* Phase 2: forget the tokens; give back file space of emptied chunks. */
    size_t left = n;
    while (left) {
        SpillChunk *c = &sp->chunks[sp->nchunks - 1];
        size_t r = left < c->n_valid ? left : c->n_valid;
        c->n_valid -= (uint32_t)r;
        sp->count -= r;
        left -= r;
        if (c->n_valid == 0) {
            uint64_t off = c->off;
            sp->nchunks--;
            sp->file_end = off;
            if (ftruncate(sp->fd, (off_t)off) != 0) { /* best effort */ }
        }
    }
    if (sp->count == 0) sp->head = 0;
    return 0;
}

/* ------------------------------------------------------------------ save / load */

int kvspill_copy_file(const char *src, const char *dst)
{
    int in = open(src, O_RDONLY | O_CLOEXEC);
    if (in < 0)
        return -1;
    int out = open(dst, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (out < 0) {
        close(in);
        return -1;
    }
    int rc = 0;
    char *buf = malloc(1 << 20);              /* per call: saves on different slots may overlap */
    if (!buf) {
        close(in);
        close(out);
        unlink(dst);
        return -1;
    }
    for (;;) {
        ssize_t r = read(in, buf, 1 << 20);
        if (r < 0) {
            if (errno == EINTR) continue;
            rc = -1;
            break;
        }
        if (r == 0)
            break;
        for (ssize_t done = 0; done < r;) {
            ssize_t w = write(out, buf + done, (size_t)(r - done));
            if (w < 0) {
                if (errno == EINTR) continue;
                rc = -1;
                goto fin;
            }
            done += w;
        }
    }
fin:
    free(buf);
    close(in);
    if (close(out) != 0)
        rc = -1;
    if (rc != 0)
        unlink(dst);
    return rc;
}

#define IDX_MAGIC   0x58444b52u        /* "RKDX" */
#define IDX_VERSION 1u

typedef struct {
    uint32_t magic, version;
    uint32_t layers, reserved;
    uint64_t count, head, file_end, nchunks;
} SpillIndexHeader;

int kvspill_save(const KVSpill *sp, int layers, const char *data_path, const char *index_path)
{
    if (!sp || sp->count == 0 || sp->fd < 0 || layers <= 0)
        return -1;
    if (kvspill_copy_file(sp->path, data_path) != 0)
        return -1;
    FILE *f = fopen(index_path, "wb");
    if (!f) {
        unlink(data_path);
        return -1;
    }
    SpillIndexHeader h = { IDX_MAGIC, IDX_VERSION, (uint32_t)layers, 0,
                           sp->count, sp->head, sp->file_end, sp->nchunks };
    int ok = fwrite(&h, sizeof h, 1, f) == 1 &&
             (sp->nchunks == 0 || fwrite(sp->chunks, sizeof *sp->chunks, sp->nchunks, f) == sp->nchunks);
    if (fclose(f) != 0)
        ok = 0;
    if (!ok) {
        unlink(data_path);
        unlink(index_path);
        return -1;
    }
    return 0;
}

int kvspill_load(KVSpill *sp, int layers, const char *data_path, const char *index_path)
{
    kvspill_clear(sp);
    FILE *f = fopen(index_path, "rb");
    if (!f)
        return -1;
    SpillIndexHeader h;
    SpillChunk *chunks = NULL;
    struct stat st;
    int ok = fread(&h, sizeof h, 1, f) == 1 && h.magic == IDX_MAGIC && h.version == IDX_VERSION &&
             h.layers == (uint32_t)layers && h.count > 0 && h.nchunks > 0 && h.nchunks < (1u << 28) &&
             stat(data_path, &st) == 0 && (uint64_t)st.st_size >= h.file_end;
    if (ok) {
        chunks = malloc(h.nchunks * sizeof *chunks);
        ok = chunks && fread(chunks, sizeof *chunks, h.nchunks, f) == h.nchunks;
    }
    fclose(f);
    if (ok) {                                  /* chunk table must describe the data file */
        uint64_t total = 0, prev_end = 0;
        for (uint64_t i = 0; ok && i < h.nchunks; i++) {
            const SpillChunk *c = &chunks[i];
            uint64_t size = 2ull * (uint64_t)layers * c->n_alloc * (uint64_t)KV_FLOATS_PER_TOKEN * sizeof(float);
            ok = c->n_valid > 0 && c->n_valid <= c->n_alloc && c->off >= prev_end && c->off + size <= h.file_end;
            prev_end = c->off + size;
            total += c->n_valid;
        }
        ok = ok && total == h.count;
    }
    if (!ok) {
        free(chunks);
        return -1;
    }
    if (!sp->path[0] || kvspill_copy_file(data_path, sp->path) != 0) {
        free(chunks);
        return -1;
    }
    sp->fd = open(sp->path, O_RDWR | O_CLOEXEC);
    if (sp->fd < 0) {
        unlink(sp->path);
        free(chunks);
        return -1;
    }
    sp->chunks = chunks;
    sp->nchunks = sp->cap_chunks = (size_t)h.nchunks;
    sp->count = h.count;
    sp->head = h.head;
    sp->file_end = h.file_end;
    if (ftruncate(sp->fd, (off_t)h.file_end) != 0) { /* trailing bytes are harmless */ }
    return 0;
}
