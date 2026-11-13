/*
 * OpenAI-compatible HTTP front-end for roaster.
 *
 *   GET  /v1/models
 *   GET  /health
 *   POST /v1/chat/completions   (stream and non-stream)
 *   /v1/sessions...             session management, incl. PATCH /v1/sessions/{id} to change
 *                               context window, routing, sampling and sliding at any time
 *
 * Every request gets its own slot on the roaster server (connection = slot).
 * Reasoning text is returned as "reasoning_content" (DeepSeek/vLLM convention).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdarg.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ctype.h>
#include "api.h"
#include "protocol.h"

#define MAX_BODY (4u << 20)
#define SESSION_NAME_MAX 64

static volatile sig_atomic_t g_stop;

/* ============================================================ growable buffer */

typedef struct { char *p; size_t n, cap; } Buf;

static int buf_grow(Buf *b, size_t add)
{
    if (b->p && b->n + add + 1 <= b->cap)
        return 0;
    size_t nc = b->cap ? b->cap : 256;
    while (nc < b->n + add + 1)
        nc *= 2;
    char *t = realloc(b->p, nc);
    if (!t)
        return -1;
    b->p = t;
    b->cap = nc;
    b->p[b->n] = '\0';
    return 0;
}

static void buf_put(Buf *b, const void *s, size_t n)
{
    if (buf_grow(b, n))
        return;
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = '\0';
}

static void buf_puts(Buf *b, const char *s) { buf_put(b, s, strlen(s)); }

static void buf_printf(Buf *b, const char *fmt, ...)
{
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n > 0 && buf_grow(b, (size_t)n) == 0) {
        vsnprintf(b->p + b->n, (size_t)n + 1, fmt, ap2);
        b->n += (size_t)n;
    }
    va_end(ap2);
}

/* Length of the valid UTF-8 sequence at s (1..4), or 0 if invalid/truncated. */
static int utf8_len(const char *s, size_t n)
{
    const unsigned char *u = (const unsigned char *)s;
    int need;
    uint32_t cp;
    if (n == 0) return 0;
    if (u[0] >= 0xC2 && u[0] <= 0xDF)      { need = 2; cp = u[0] & 0x1F; }
    else if ((u[0] & 0xF0) == 0xE0)        { need = 3; cp = u[0] & 0x0F; }
    else if (u[0] >= 0xF0 && u[0] <= 0xF4) { need = 4; cp = u[0] & 0x07; }
    else return 0;
    if (n < (size_t)need) return 0;
    for (int k = 1; k < need; k++) {
        if ((u[k] & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (u[k] & 0x3F);
    }
    if (need == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) return 0;
    if (need == 4 && (cp < 0x10000 || cp > 0x10FFFF)) return 0;
    return need;
}

/* Number of leading bytes that do not end in a truncated UTF-8 sequence. */
static size_t utf8_complete_prefix(const unsigned char *s, size_t n)
{
    size_t lo = n > 3 ? n - 3 : 0;
    for (size_t k = n; k > lo; k--) {
        unsigned char c = s[k - 1];
        if ((c & 0xC0) == 0x80)
            continue;
        size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return (k - 1 + need > n) ? k - 1 : n;
    }
    return n;
}

/* Append a JSON string literal (invalid UTF-8 becomes U+FFFD). */
static void buf_json(Buf *b, const char *s, size_t n)
{
    buf_put(b, "\"", 1);
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            switch (c) {
            case '"':  buf_put(b, "\\\"", 2); break;
            case '\\': buf_put(b, "\\\\", 2); break;
            case '\n': buf_put(b, "\\n", 2);  break;
            case '\r': buf_put(b, "\\r", 2);  break;
            case '\t': buf_put(b, "\\t", 2);  break;
            default:
                if (c < 0x20) buf_printf(b, "\\u%04x", c);
                else buf_put(b, s + i, 1);
            }
            i++;
        } else {
            int l = utf8_len(s + i, n - i);
            if (l > 0) { buf_put(b, s + i, (size_t)l); i += (size_t)l; }
            else       { buf_put(b, "\xEF\xBF\xBD", 3); i++; }
        }
    }
    buf_put(b, "\"", 1);
}

static void buf_put_utf8(Buf *b, uint32_t cp)
{
    char o[4];
    if (cp < 0x80)         { o[0] = (char)cp; buf_put(b, o, 1); }
    else if (cp < 0x800)   { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); buf_put(b, o, 2); }
    else if (cp < 0x10000) { o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                             o[2] = (char)(0x80 | (cp & 0x3F)); buf_put(b, o, 3); }
    else                   { o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
                             o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F)); buf_put(b, o, 4); }
}

/* ============================================================ minimal JSON parser */

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct JVal {
    JType t;
    double num;
    int b;
    char *s;
    size_t slen;
    struct JVal *v;   /* array items / object values */
    char **k;         /* object keys */
    size_t n;
} JVal;

static void jfree(JVal *j)
{
    if (!j) return;
    if (j->t == J_STR)
        free(j->s);
    if (j->t == J_ARR || j->t == J_OBJ) {
        for (size_t i = 0; i < j->n; i++) {
            jfree(&j->v[i]);
            if (j->k) free(j->k[i]);
        }
        free(j->v);
        free(j->k);
    }
    memset(j, 0, sizeof(*j));
}

typedef struct { const char *p, *end; } JP;

static void jws(JP *p)
{
    while (p->p < p->end && (*p->p == ' ' || *p->p == '\t' || *p->p == '\n' || *p->p == '\r'))
        p->p++;
}

static int jhex4(JP *p, uint32_t *out)
{
    if (p->end - p->p < 4) return 0;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = *p->p++;
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}

static int jstr(JP *p, char **out, size_t *olen)
{
    Buf b = {0};
    p->p++;                                   /* opening quote */
    if (buf_grow(&b, 0)) return -1;
    while (p->p < p->end) {
        char c = *p->p++;
        if (c == '"') {
            *out = b.p;
            *olen = b.n;
            return 0;
        }
        if (c != '\\') { buf_put(&b, &c, 1); continue; }
        if (p->p >= p->end) break;
        char e = *p->p++;
        switch (e) {
        case '"': case '\\': case '/': buf_put(&b, &e, 1); break;
        case 'b': buf_put(&b, "\b", 1); break;
        case 'f': buf_put(&b, "\f", 1); break;
        case 'n': buf_put(&b, "\n", 1); break;
        case 'r': buf_put(&b, "\r", 1); break;
        case 't': buf_put(&b, "\t", 1); break;
        case 'u': {
            uint32_t cp;
            if (!jhex4(p, &cp)) goto bad;
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                uint32_t lo;
                if (p->end - p->p >= 6 && p->p[0] == '\\' && p->p[1] == 'u') {
                    p->p += 2;
                    if (!jhex4(p, &lo)) goto bad;
                    cp = (lo >= 0xDC00 && lo <= 0xDFFF)
                         ? 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00) : 0xFFFD;
                } else cp = 0xFFFD;
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) cp = 0xFFFD;
            buf_put_utf8(&b, cp);
            break;
        }
        default: goto bad;
        }
    }
bad:
    free(b.p);
    return -1;
}

static int jval_in(JP *p, JVal *out, int depth);

/* On failure `out` is freed and zeroed, so callers never leak partial trees. */
static int jval(JP *p, JVal *out, int depth)
{
    memset(out, 0, sizeof(*out));
    int rc = jval_in(p, out, depth);
    if (rc) jfree(out);
    return rc;
}

static int jval_in(JP *p, JVal *out, int depth)
{
    if (depth > 64) return -1;
    jws(p);
    if (p->p >= p->end) return -1;
    char c = *p->p;

    if (c == '"') {
        out->t = J_STR;
        return jstr(p, &out->s, &out->slen);
    }
    if (c == '{' || c == '[') {
        int obj = c == '{';
        char close = obj ? '}' : ']';
        out->t = obj ? J_OBJ : J_ARR;
        p->p++;
        jws(p);
        if (p->p < p->end && *p->p == close) { p->p++; return 0; }
        for (;;) {
            char *key = NULL;
            size_t kl;
            jws(p);
            if (obj) {
                if (p->p >= p->end || *p->p != '"') return -1;
                if (jstr(p, &key, &kl)) return -1;
                jws(p);
                if (p->p >= p->end || *p->p != ':') { free(key); return -1; }
                p->p++;
            }
            JVal child;
            if (jval(p, &child, depth + 1)) { free(key); return -1; }
            JVal *nv = realloc(out->v, (out->n + 1) * sizeof(JVal));
            if (nv) out->v = nv;
            char **nk = out->k;
            if (obj) {
                nk = realloc(out->k, (out->n + 1) * sizeof(char *));
                if (nk) out->k = nk;
            }
            if (!nv || (obj && !nk)) { free(key); jfree(&child); return -1; }
            out->v[out->n] = child;
            if (obj) out->k[out->n] = key;
            out->n++;
            jws(p);
            if (p->p >= p->end) return -1;
            if (*p->p == ',') { p->p++; continue; }
            if (*p->p == close) { p->p++; return 0; }
            return -1;
        }
    }
    if (p->end - p->p >= 4 && !strncmp(p->p, "true", 4))  { out->t = J_BOOL; out->b = 1; p->p += 4; return 0; }
    if (p->end - p->p >= 5 && !strncmp(p->p, "false", 5)) { out->t = J_BOOL; out->b = 0; p->p += 5; return 0; }
    if (p->end - p->p >= 4 && !strncmp(p->p, "null", 4))  { out->t = J_NULL; p->p += 4; return 0; }
    if (c == '-' || (c >= '0' && c <= '9')) {
        char *e;
        double d = strtod(p->p, &e);              /* body buffer is NUL-terminated */
        if (e == p->p || e > p->end) return -1;
        out->t = J_NUM;
        out->num = d;
        p->p = e;
        return 0;
    }
    return -1;
}

static const JVal *jget(const JVal *o, const char *key)
{
    if (!o || o->t != J_OBJ) return NULL;
    for (size_t i = 0; i < o->n; i++)
        if (!strcmp(o->k[i], key)) return &o->v[i];
    return NULL;
}

static int jnum_int(const JVal *x, int *out)
{
    if (!x || x->t != J_NUM) return 0;
    double d = x->num;
    if (d > 2e9) d = 2e9;
    if (d < -2e9) d = -2e9;
    *out = (int)d;
    return 1;
}

/* ============================================================ settings (JSON <-> SlotConfig) */

enum { CFG_SLOT = 1, CFG_SAMPLING = 2 };

static int jflag(const JVal *x, int *out)
{
    if (!x) return 0;
    if (x->t == J_BOOL) { *out = x->b; return 1; }
    if (x->t == J_NUM)  { *out = x->num != 0; return 1; }
    return 0;
}

/* Copy every setting present in the JSON object into cf; absent keys are left alone.
 * Slot-level: context_length, cache_route, route_j/m/alpha/p, slide, keep.
 * Sampling:   temperature, top_p, top_k, max_tokens, think. */
static void apply_settings(const JVal *root, SlotConfig *cf, int which)
{
    const JVal *x;
    int iv;
    if (which & CFG_SLOT) {
        if ((x = jget(root, "context_length")) && x->t == J_NUM)
            cf->context_length = x->num < 1 ? 0 : (size_t)(x->num > 1e12 ? 1e12 : x->num);
        if (jflag(jget(root, "cache_route"), &iv)) cf->cache_route = iv;
        if (jnum_int(jget(root, "route_j"), &iv)) cf->route_j = iv;
        if (jnum_int(jget(root, "route_m"), &iv)) cf->route_m = iv;
        if ((x = jget(root, "route_alpha")) && x->t == J_NUM) cf->route_alpha = (float)x->num;
        if ((x = jget(root, "route_p")) && x->t == J_NUM) cf->route_p = (float)x->num;
        if (jflag(jget(root, "slide"), &iv)) cf->slide = iv;
        if (jnum_int(jget(root, "keep"), &iv) || jnum_int(jget(root, "slide_keep"), &iv)) cf->keep = iv;
    }
    if (which & CFG_SAMPLING) {
        if ((x = jget(root, "temperature")) && x->t == J_NUM) cf->temperature = (float)x->num;
        if ((x = jget(root, "top_p")) && x->t == J_NUM) cf->top_p = (float)x->num;
        if (jnum_int(jget(root, "top_k"), &iv)) cf->top_k = iv;
        if (jnum_int(jget(root, "max_tokens"), &iv)) cf->max_tokens = iv;
        if (jnum_int(jget(root, "max_completion_tokens"), &iv)) cf->max_tokens = iv;
        if (jflag(jget(root, "think"), &iv)) cf->think = iv;
        if ((x = jget(jget(root, "chat_template_kwargs"), "enable_thinking")) && x->t == J_BOOL) cf->think = x->b;
    }
}

static int slot_level_differs(const SlotConfig *a, const SlotConfig *b)
{
    return a->context_length != b->context_length || a->cache_route != b->cache_route ||
           a->route_j != b->route_j || a->route_m != b->route_m || a->route_alpha != b->route_alpha ||
           a->route_p != b->route_p || a->slide != b->slide || a->keep != b->keep;
}

static int sampling_differs(const SlotConfig *a, const SlotConfig *b)
{
    return a->temperature != b->temperature || a->top_p != b->top_p || a->top_k != b->top_k ||
           a->max_tokens != b->max_tokens || a->think != b->think;
}

static void config_line(char *buf, size_t cap, const char *verb, const SlotConfig *c)
{
    snprintf(buf, cap, "%s %zu %d %d %d %.9g %.9g %.9g %.9g %d %d %d %d %d", verb,
             c->context_length, c->cache_route, c->route_j, c->route_m, c->route_alpha, c->route_p,
             c->temperature, c->top_p, c->top_k, c->max_tokens, c->think, c->slide, c->keep);
}

/* ============================================================ HTTP helpers */

static int send_all(int fd, const void *data, size_t n)
{
    const char *p = data;
    while (n > 0) {
        ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static const char *reason(int status)
{
    switch (status) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    default:  return "Internal Server Error";
    }
}

static void send_json(int fd, int status, const char *body, size_t n)
{
    char head[320];
    int h = snprintf(head, sizeof head,
                     "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
                     "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                     status, reason(status), n);
    if (send_all(fd, head, (size_t)h) == 0 && n)
        send_all(fd, body, n);
}

static void send_error(int fd, int status, const char *type, const char *msg)
{
    Buf b = {0};
    buf_puts(&b, "{\"error\":{\"message\":");
    buf_json(&b, msg, strlen(msg));
    buf_printf(&b, ",\"type\":\"%s\",\"param\":null,\"code\":null}}", type);
    send_json(fd, status, b.p, b.n);
    free(b.p);
}

typedef struct {
    char method[16], path[256], auth[512], session[SESSION_NAME_MAX];
    size_t content_length;
    int expect100, chunked;
} Req;

/* 0 = ok, -1 = connection problem (no reply), otherwise an HTTP status to answer with. */
static int read_request(int fd, Req *rq, Buf *body)
{
    Buf h = {0};
    char tmp[4096];
    char *end;
    memset(rq, 0, sizeof *rq);

    while (!(h.p && (end = strstr(h.p, "\r\n\r\n")))) {
        if (h.n > 65536) { free(h.p); return 431; }
        ssize_t n = read(fd, tmp, sizeof tmp);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { free(h.p); return -1; }
        buf_put(&h, tmp, (size_t)n);
    }
    size_t hlen = (size_t)(end - h.p) + 4;
    *end = '\0';

    char *save = NULL;
    char *line = strtok_r(h.p, "\r\n", &save);
    if (!line || sscanf(line, "%15s %255s", rq->method, rq->path) != 2) { free(h.p); return 400; }
    while ((line = strtok_r(NULL, "\r\n", &save))) {
        if (!strncasecmp(line, "content-length:", 15)) {
            rq->content_length = (size_t)strtoull(line + 15, NULL, 10);
        } else if (!strncasecmp(line, "authorization:", 14)) {
            const char *v = line + 14;
            while (*v == ' ' || *v == '\t') v++;
            snprintf(rq->auth, sizeof rq->auth, "%s", v);
        } else if (!strncasecmp(line, "x-session-id:", 13)) {
            const char *v = line + 13;
            while (*v == ' ' || *v == '\t') v++;
            snprintf(rq->session, sizeof rq->session, "%s", v);
        } else if (!strncasecmp(line, "expect:", 7)) {
            if (strcasestr(line + 7, "100-continue")) rq->expect100 = 1;
        } else if (!strncasecmp(line, "transfer-encoding:", 18)) {
            if (strcasestr(line + 18, "chunked")) rq->chunked = 1;
        }
    }
    if (rq->chunked) { free(h.p); return 411; }
    if (rq->content_length > MAX_BODY) { free(h.p); return 413; }

    size_t have = h.n - hlen;
    if (have > rq->content_length) have = rq->content_length;
    buf_put(body, h.p + hlen, have);
    free(h.p);
    if (rq->expect100 && body->n < rq->content_length)
        send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    while (body->n < rq->content_length) {
        ssize_t n = read(fd, tmp, sizeof tmp);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        size_t take = (size_t)n;
        if (body->n + take > rq->content_length) take = rq->content_length - body->n;
        buf_put(body, tmp, take);
    }
    buf_grow(body, 0);
    return 0;
}

/* ============================================================ roaster server link */

static int up_connect(const char *path)
{
    struct sockaddr_un a = { .sun_family = AF_UNIX };
    if (strlen(path) >= sizeof a.sun_path) return -1;
    strcpy(a.sun_path, path);
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    if (connect(fd, (struct sockaddr *)&a, sizeof a) != 0) { close(fd); return -1; }
    return fd;
}

static int read_line(int fd, char *buf, size_t cap)
{
    size_t n = 0;
    for (;;) {                                /* always consume the whole line */
        char c;
        ssize_t r = read(fd, &c, 1);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        if (c == '\n') break;
        if (n + 1 < cap) buf[n++] = c;
    }
    buf[n] = '\0';
    return 0;
}

/* ============================================================ chat completions */

typedef struct {
    int fd, stream, started, client_ok, include_usage;
    int mode;                 /* 0 = answer, 1 = reasoning */
    int special;              /* inside a literal special token: dropped */
    char id[48], model[160];
    long created;
    Buf pend, content, reasoning;
} Run;

static unsigned long g_counter;

static void send_chunk(Run *r, const char *data, size_t n)
{
    if (!r->client_ok || n == 0) return;
    Buf c = {0};
    buf_printf(&c, "%zx\r\n", n);
    buf_put(&c, data, n);
    buf_put(&c, "\r\n", 2);
    if (!c.p || send_all(r->fd, c.p, c.n) != 0)
        r->client_ok = 0;
    free(c.p);
}

static void chunk_head(Buf *b, const Run *r)
{
    buf_printf(b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":",
               r->id, r->created);
    buf_json(b, r->model, strlen(r->model));
    buf_put(b, ",", 1);
}

static int start_stream(Run *r)
{
    if (r->started) return r->client_ok;
    r->started = 1;
    static const char head[] =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n";
    if (send_all(r->fd, head, sizeof(head) - 1) != 0) { r->client_ok = 0; return 0; }
    Buf b = {0};
    chunk_head(&b, r);
    buf_puts(&b, "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n");
    send_chunk(r, b.p, b.n);
    free(b.p);
    return r->client_ok;
}

static void emit(Run *r, int reasoning, const char *s, size_t n)
{
    if (!n) return;
    if (!r->stream) {
        Buf *dst = reasoning ? &r->reasoning : &r->content;
        buf_put(dst, s, n);
        return;
    }
    if (!start_stream(r)) return;
    Buf b = {0};
    chunk_head(&b, r);
    buf_printf(&b, "\"choices\":[{\"index\":0,\"delta\":{\"%s\":", reasoning ? "reasoning_content" : "content");
    buf_json(&b, s, n);
    buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    send_chunk(r, b.p, b.n);
    free(b.p);
}

/* Emit pending text. Unless forced, hold back trailing newlines and half-received UTF-8. */
static void flush_pend(Run *r, int force)
{
    size_t n = r->pend.n, keep = n;
    if (!n) return;
    const unsigned char *s = (const unsigned char *)r->pend.p;
    if (!force) {
        while (keep > 0 && s[keep - 1] == '\n') keep--;
        keep = utf8_complete_prefix(s, keep);
    }
    if (!keep) return;
    emit(r, r->mode == 1, r->pend.p, keep);
    memmove(r->pend.p, r->pend.p + keep, n - keep);
    r->pend.n = n - keep;
    r->pend.p[r->pend.n] = '\0';
}

static void msg_text(const JVal *content, Buf *out)
{
    if (!content) return;
    if (content->t == J_STR) {
        for (size_t i = 0; i < content->slen; i++)
            if (content->s[i]) buf_put(out, content->s + i, 1);
    } else if (content->t == J_ARR) {
        for (size_t i = 0; i < content->n; i++) {
            const JVal *part = &content->v[i];
            const JVal *type = jget(part, "type"), *text = jget(part, "text");
            if (type && type->t == J_STR && strcmp(type->s, "text") && strcmp(type->s, "input_text"))
                continue;                          /* images etc. are ignored */
            if (text && text->t == J_STR) buf_put(out, text->s, text->slen);
        }
    }
}

/* The server always wraps the prompt in a single user turn, so history is flattened into text. */
/* All system/developer messages of the request, joined. They are not part of the flattened turn:
 * the server wraps them in a real system turn and pins them in the attention window. */
static void collect_system(const JVal *msgs, Buf *out)
{
    for (size_t i = 0; i < msgs->n; i++) {
        const JVal *role = jget(&msgs->v[i], "role");
        const char *rs = role && role->t == J_STR ? role->s : "user";
        if (strcmp(rs, "system") && strcmp(rs, "developer")) continue;
        Buf txt = {0};
        msg_text(jget(&msgs->v[i], "content"), &txt);
        if (txt.n) {
            if (out->n) buf_puts(out, "\n\n");
            buf_put(out, txt.p, txt.n);
        }
        free(txt.p);
    }
}

static void build_prompt(const JVal *msgs, size_t start, Buf *out)
{
    Buf sys = {0}, conv = {0}, first = {0};
    int turns = 0, first_is_user = 0;
    for (size_t i = start; i < msgs->n; i++) {
        const JVal *m = &msgs->v[i];
        const JVal *role = jget(m, "role");
        const char *rs = role && role->t == J_STR ? role->s : "user";
        Buf txt = {0};
        msg_text(jget(m, "content"), &txt);
        if (!strcmp(rs, "system") || !strcmp(rs, "developer")) {
            /* handled by collect_system() */
        } else {
            const char *label = !strcmp(rs, "assistant") ? "Assistant"
                              : (!strcmp(rs, "tool") || !strcmp(rs, "function")) ? "Tool" : "User";
            if (turns == 0) {
                buf_put(&first, txt.p ? txt.p : "", txt.n);
                first_is_user = !strcmp(label, "User");
            }
            if (conv.n) buf_puts(&conv, "\n\n");
            buf_printf(&conv, "%s: ", label);
            buf_put(&conv, txt.p ? txt.p : "", txt.n);
            turns++;
        }
        free(txt.p);
    }
    if (sys.n) {
        buf_put(out, sys.p, sys.n);
        if (turns) buf_puts(out, "\n\n");
    }
    if (turns == 1 && first_is_user) buf_put(out, first.p ? first.p : "", first.n);
    else if (conv.n) buf_put(out, conv.p, conv.n);
    free(sys.p); free(conv.p); free(first.p);
}

static int info_int(const char *info, const char *key)
{
    const char *p = strstr(info, key);
    return p ? atoi(p + strlen(key)) : 0;
}

/* ============================================================ sessions
 *
 * A session is one context that lives on the roaster server as long as the API keeps its
 * connection open. Requests that name the same session (field "session", header X-Session-Id,
 * default "default") continue the same context until it is reset, unloaded or deleted.
 * One request at a time can use a session; the others get HTTP 409.
 */

#define DEFAULT_SESSION "default"

typedef struct Session {
    char name[SESSION_NAME_MAX];
    int fd;                   /* connection to `roaster serve`; closing it destroys the slot */
    int slot;
    int busy;                 /* a request or an admin call is using fd */
    struct Session *next;
} Session;

static pthread_mutex_t g_sess_lock = PTHREAD_MUTEX_INITIALIZER;
static Session *g_sessions;
static unsigned long g_sess_counter;

static int valid_session_name(const char *s)
{
    size_t n = strlen(s);
    if (n == 0 || n >= SESSION_NAME_MAX) return 0;
    for (; *s; s++)
        if (!isalnum((unsigned char)*s) && *s != '-' && *s != '_' && *s != '.') return 0;
    return 1;
}

static Session *session_find_locked(const char *name)
{
    for (Session *s = g_sessions; s; s = s->next)
        if (!strcmp(s->name, name)) return s;
    return NULL;
}

/* One request / one-line answer on the session's connection. */
static int up_cmd(int fd, const char *line, char *reply, size_t cap)
{
    if (send_all(fd, line, strlen(line)) != 0 || send_all(fd, "\n", 1) != 0) return -1;
    return read_line(fd, reply, cap);
}

/* Ask the server to stop the running request of a slot (works from a separate connection). */
static int api_stop_slot(const ApiOptions *opt, int slot)
{
    int fd = up_connect(opt->socket_path);
    if (fd < 0) return 0;
    char line[48], reply[64];
    snprintf(line, sizeof line, "STOP %d", slot);
    int hit = up_cmd(fd, line, reply, sizeof reply) == 0 && !strncmp(reply, "OK 1", 4);
    close(fd);
    return hit;
}

/* Create the slot for a new session. Returns 0 or an HTTP status (message in msg). */
static int session_open(const ApiOptions *opt, const SlotConfig *cf, Session *s, char *msg, size_t cap)
{
    int fd = up_connect(opt->socket_path);
    if (fd < 0) {
        snprintf(msg, cap, "cannot connect to the roaster server (is `roaster serve` running?)");
        return 502;
    }
    char line[512], reply[512];
    config_line(line, sizeof line, "CREATE", cf);
    if (up_cmd(fd, line, reply, sizeof reply) != 0) {
        close(fd);
        snprintf(msg, cap, "lost connection to the roaster server");
        return 502;
    }
    if (strncmp(reply, "OK ", 3) != 0) {
        const char *m = !strncmp(reply, "ERR ", 4) ? reply + 4 : reply;
        int busy = strstr(m, "could not create slot") != NULL;
        snprintf(msg, cap, "%s", m);
        close(fd);
        return busy ? 503 : 400;
    }
    s->fd = fd;
    s->slot = atoi(reply + 3);
    return 0;
}

static void session_remove(Session *s)
{
    pthread_mutex_lock(&g_sess_lock);
    for (Session **l = &g_sessions; *l; l = &(*l)->next)
        if (*l == s) { *l = s->next; break; }
    pthread_mutex_unlock(&g_sess_lock);
    if (s->fd >= 0) close(s->fd);          /* the server destroys the slot (and its saved context) */
    free(s);
}

/* Find (or create) a session and mark it busy. Returns 0 or an HTTP status (message in msg). */
static int session_acquire(const ApiOptions *opt, const char *name, const SlotConfig *cf,
                           int create, int must_be_new, Session **out, char *msg, size_t cap)
{
    pthread_mutex_lock(&g_sess_lock);
    Session *s = session_find_locked(name);
    if (s) {
        int st = 0;
        if (must_be_new) { snprintf(msg, cap, "session '%s' already exists", name); st = 409; }
        else if (s->busy) { snprintf(msg, cap, "session '%s' is busy with another request", name); st = 409; }
        else s->busy = 1;
        pthread_mutex_unlock(&g_sess_lock);
        if (!st) *out = s;
        return st;
    }
    if (!create) {
        pthread_mutex_unlock(&g_sess_lock);
        snprintf(msg, cap, "no such session: '%s'", name);
        return 404;
    }
    s = calloc(1, sizeof *s);
    if (!s) {
        pthread_mutex_unlock(&g_sess_lock);
        snprintf(msg, cap, "out of memory");
        return 500;
    }
    snprintf(s->name, sizeof s->name, "%s", name);
    s->fd = -1;
    s->slot = -1;
    s->busy = 1;                           /* placeholder: others see it as busy while the slot is created */
    s->next = g_sessions;
    g_sessions = s;
    pthread_mutex_unlock(&g_sess_lock);
    int st = session_open(opt, cf, s, msg, cap);
    if (st) {
        session_remove(s);
        return st;
    }
    *out = s;
    return 0;
}

/* dead = the connection broke (or the session is being deleted): drop it instead of reusing it. */
static void session_release(Session *s, int dead)
{
    if (dead) { session_remove(s); return; }
    pthread_mutex_lock(&g_sess_lock);
    s->busy = 0;
    pthread_mutex_unlock(&g_sess_lock);
}

typedef struct {
    int loaded, broken;
    long long position, context, ram, state, window, spilled, spill_bytes, pinned;
    SlotConfig cfg;                           /* current settings of the slot */
} SessInfo;

static long long kv_num(const char *line, const char *key)
{
    char pat[32];
    snprintf(pat, sizeof pat, "%s=", key);
    for (const char *p = line; (p = strstr(p, pat)); p++)
        if (p == line || p[-1] == ' ') return atoll(p + strlen(pat));
    return -1;
}

static double kv_dbl(const char *line, const char *key)
{
    char pat[32];
    snprintf(pat, sizeof pat, "%s=", key);
    for (const char *p = line; (p = strstr(p, pat)); p++)
        if (p == line || p[-1] == ' ') return atof(p + strlen(pat));
    return 0.0;
}

static int session_query(Session *s, SessInfo *si)
{
    char reply[1024];
    if (up_cmd(s->fd, "INFO", reply, sizeof reply) != 0 || strncmp(reply, "OK", 2) != 0) return -1;
    si->loaded = kv_num(reply, "loaded") == 1;
    si->broken = kv_num(reply, "broken") == 1;
    si->position = kv_num(reply, "position");
    si->context = kv_num(reply, "context");
    si->ram = kv_num(reply, "ram");
    si->state = kv_num(reply, "state");
    si->window = kv_num(reply, "window");
    si->spilled = kv_num(reply, "spilled");
    si->spill_bytes = kv_num(reply, "spill_bytes");
    si->pinned = kv_num(reply, "pinned");
    SlotConfig *c = &si->cfg;
    c->context_length = si->context > 0 ? (size_t)si->context : 0;
    c->cache_route = (int)kv_num(reply, "cache_route");
    c->route_j = (int)kv_num(reply, "route_j");
    c->route_m = (int)kv_num(reply, "route_m");
    c->route_alpha = (float)kv_dbl(reply, "route_alpha");
    c->route_p = (float)kv_dbl(reply, "route_p");
    c->temperature = (float)kv_dbl(reply, "temp");
    c->top_p = (float)kv_dbl(reply, "top_p");
    c->top_k = (int)kv_num(reply, "top_k");
    c->max_tokens = (int)kv_num(reply, "max_tokens");
    c->think = (int)kv_num(reply, "think");
    c->slide = (int)kv_num(reply, "slide");
    c->keep = (int)kv_num(reply, "keep");
    return 0;
}

static void session_json(Buf *b, const char *name, const Session *s, const SessInfo *si)
{
    buf_puts(b, "{\"id\":");
    buf_json(b, name, strlen(name));
    buf_puts(b, ",\"object\":\"session\"");
    if (!s || !si) {
        buf_puts(b, ",\"busy\":true}");
        return;
    }
    const SlotConfig *c = &si->cfg;
    buf_printf(b, ",\"slot\":%d,\"busy\":false,\"loaded\":%s,\"broken\":%s,\"position\":%lld,"
                  "\"context_length\":%lld,\"window_tokens\":%lld,\"spilled_tokens\":%lld,\"pinned_tokens\":%lld,"
                  "\"ram_bytes\":%lld,\"state_bytes\":%lld,\"spill_bytes\":%lld,"
                  "\"config\":{\"context_length\":%lld,\"cache_route\":%s,\"route_j\":%d,\"route_m\":%d,"
                  "\"route_alpha\":%g,\"route_p\":%g,\"temperature\":%g,\"top_p\":%g,\"top_k\":%d,"
                  "\"max_tokens\":%d,\"think\":%s,\"slide\":%s,\"keep\":%d}}",
               s->slot, si->loaded ? "true" : "false", si->broken ? "true" : "false",
               si->position, si->context, si->window, si->spilled, si->pinned, si->ram, si->state, si->spill_bytes,
               si->context, c->cache_route ? "true" : "false", c->route_j, c->route_m,
               (double)c->route_alpha, (double)c->route_p, (double)c->temperature, (double)c->top_p,
               c->top_k, c->max_tokens, c->think ? "true" : "false", c->slide ? "true" : "false", c->keep);
}

/* ============================================================ chat completions (continued) */

static void put_escaped_cmd(Buf *cmd, const char *verb, const char *text, size_t n)
{
    buf_puts(cmd, verb);
    for (size_t i = 0; i < n; i++) {
        char c = text[i];
        if (c == '\n') buf_put(cmd, "\\n", 2);
        else if (c == '\r') buf_put(cmd, "\\r", 2);
        else if (c == '\\') buf_put(cmd, "\\\\", 2);
        else if (c) buf_put(cmd, &c, 1);
    }
    buf_put(cmd, "\n", 1);
}

static void handle_chat(int fd, const ApiOptions *opt, const Req *rq, Buf *body)
{
    JVal root;
    Buf prompt = {0}, cmd = {0}, info = {0}, sysb = {0};
    Run r;
    Session *sess = NULL;
    int dead = 0, stop_sent = 0, dirty = 0;
    SlotConfig base;                          /* persistent settings of the session */
    char msg[256], line[512], reply[512];
    memset(&base, 0, sizeof base);
    memset(&r, 0, sizeof r);
    r.fd = fd;
    r.client_ok = 1;
    r.created = (long)time(NULL);
    snprintf(r.id, sizeof r.id, "chatcmpl-%lx%lx", (unsigned long)r.created,
             __sync_fetch_and_add(&g_counter, 1));

    JP jp = { body->p, body->p + body->n };
    int bad = jval(&jp, &root, 0);
    if (!bad) { jws(&jp); bad = jp.p != jp.end; }
    if (bad || root.t != J_OBJ) {
        jfree(&root);
        send_error(fd, 400, "invalid_request_error", "request body is not valid JSON");
        return;
    }

    /* ---- parameters ---- */
    /* Used only if this request creates the session; slot-level keys (context_length, routing,
     * slide, keep) set here, sampling keys are applied per request below. */
    SlotConfig cf = opt->defaults;
    const JVal *x;
    int iv;
    apply_settings(&root, &cf, CFG_SLOT);
    if (jnum_int(jget(&root, "n"), &iv) && iv != 1) {
        send_error(fd, 400, "invalid_request_error", "only n=1 is supported");
        goto out;
    }
    if ((x = jget(&root, "stream")) && x->t == J_BOOL) r.stream = x->b;
    if ((x = jget(jget(&root, "stream_options"), "include_usage")) && x->t == J_BOOL) r.include_usage = x->b;
    x = jget(&root, "model");
    snprintf(r.model, sizeof r.model, "%s", x && x->t == J_STR && x->s[0] ? x->s : opt->model_name);

    const char *sname = rq->session[0] ? rq->session : DEFAULT_SESSION;
    if ((x = jget(&root, "session")) && x->t == J_STR && x->s[0]) sname = x->s;
    if (!valid_session_name(sname)) {
        send_error(fd, 400, "invalid_request_error",
                   "invalid session name (1-63 characters: letters, digits, '-', '_', '.')");
        goto out;
    }
    int do_reset = (x = jget(&root, "reset")) && x->t == J_BOOL && x->b;

    const char *err = slot_config_check(&cf);
    if (err) {
        send_error(fd, 400, "invalid_request_error", err);
        goto out;
    }

    const JVal *msgs = jget(&root, "messages");
    if (!msgs || msgs->t != J_ARR || msgs->n == 0) {
        send_error(fd, 400, "invalid_request_error", "'messages' must be a non-empty array");
        goto out;
    }

    /* ---- session ---- */
    int st = session_acquire(opt, sname, &cf, 1, 0, &sess, msg, sizeof msg);
    if (st) {
        send_error(fd, st, st >= 500 ? "server_error" : "invalid_request_error", msg);
        goto out;
    }
    if (do_reset && (up_cmd(sess->fd, "RESET", reply, sizeof reply) != 0 || strncmp(reply, "OK", 2) != 0)) {
        dead = (reply[0] == '\0');
        send_error(fd, dead ? 502 : 409, dead ? "server_error" : "invalid_request_error",
                   dead ? "lost connection to the roaster server" : reply);
        goto out;
    }
    SessInfo si;
    if (session_query(sess, &si) != 0) {
        dead = 1;
        send_error(fd, 502, "server_error", "lost connection to the roaster server");
        goto out;
    }
    if (!si.loaded) {
        snprintf(msg, sizeof msg, "session '%s' is unloaded; POST /v1/sessions/%s/load first", sname, sname);
        send_error(fd, 409, "invalid_request_error", msg);
        goto out;
    }
    if (si.broken) {
        snprintf(msg, sizeof msg, "session '%s' was interrupted in an unrecoverable way; POST /v1/sessions/%s/reset", sname, sname);
        send_error(fd, 409, "invalid_request_error", msg);
        goto out;
    }

    /* ---- prompt ----
     * An empty context gets the whole message list (flattened). A context that already holds the
     * conversation only gets what is new: the messages after the last assistant message. */
    size_t start = 0;
    if (si.position > 0) {
        for (size_t i = 0; i < msgs->n; i++) {
            const JVal *role = jget(&msgs->v[i], "role");
            if (role && role->t == J_STR && !strcmp(role->s, "assistant")) start = i + 1;
        }
        if (start >= msgs->n) {
            send_error(fd, 400, "invalid_request_error",
                       "the context already holds the conversation: send a message after the last assistant message (or \"reset\": true)");
            goto out;
        }
    }
    build_prompt(msgs, start, &prompt);
    collect_system(msgs, &sysb);
    if (sysb.n > ROASTER_MAX_PROMPT) {
        send_error(fd, 400, "invalid_request_error", "system prompt too long");
        goto out;
    }
    if (prompt.n == 0) {
        send_error(fd, 400, "invalid_request_error", "messages contain no text");
        goto out;
    }
    if (prompt.n > ROASTER_MAX_PROMPT) {
        snprintf(msg, sizeof msg, "prompt too long (%zu bytes, max %d)", prompt.n, ROASTER_MAX_PROMPT);
        send_error(fd, 400, "invalid_request_error", msg);
        goto out;
    }

    /* ---- system prompt: applies when the conversation starts (empty context) and is pinned ---- */
    if (si.position == 0 && sysb.n > 0) {
        Buf sc = {0};
        put_escaped_cmd(&sc, "SYS ", sysb.p, sysb.n);
        if (sc.n) sc.p[--sc.n] = '\0';                     /* up_cmd adds the newline */
        int rc = up_cmd(sess->fd, sc.p, reply, sizeof reply);
        free(sc.p);
        if (rc != 0) {
            dead = 1;
            send_error(fd, 502, "server_error", "lost connection to the roaster server");
            goto out;
        }
        if (strncmp(reply, "OK", 2) != 0) {
            send_error(fd, 400, "invalid_request_error", !strncmp(reply, "ERR ", 4) ? reply + 4 : reply);
            goto out;
        }
    }

    /* ---- dynamic settings ----
     * Slot-level keys in the body (context_length, cache_route, route_*, slide, keep) change the
     * session for good; sampling keys (temperature, top_p, top_k, max_tokens, think) apply to this
     * request only and are put back afterwards. */
    base = si.cfg;
    SlotConfig want = base;
    apply_settings(&root, &want, CFG_SLOT | CFG_SAMPLING);
    err = slot_config_check(&want);
    if (err) {
        send_error(fd, 400, "invalid_request_error", err);
        goto out;
    }
    if (slot_level_differs(&want, &base)) {
        SlotConfig nc = base;
        nc.context_length = want.context_length; nc.cache_route = want.cache_route;
        nc.route_j = want.route_j; nc.route_m = want.route_m; nc.route_alpha = want.route_alpha;
        nc.route_p = want.route_p; nc.slide = want.slide; nc.keep = want.keep;
        config_line(line, sizeof line, "CONFIG", &nc);
        if (up_cmd(sess->fd, line, reply, sizeof reply) != 0) {
            dead = 1;
            send_error(fd, 502, "server_error", "lost connection to the roaster server");
            goto out;
        }
        if (strncmp(reply, "OK", 2) != 0) {
            const char *e = !strncmp(reply, "ERR ", 4) ? reply + 4 : reply;
            int code = strstr(e, "could not") ? 500 : (strstr(e, "busy") || strstr(e, "unloaded")) ? 409 : 400;
            send_error(fd, code, code == 500 ? "server_error" : "invalid_request_error", e);
            goto out;
        }
        base = nc;
    }
    if (sampling_differs(&want, &base)) {
        snprintf(line, sizeof line, "SET %.9g %.9g %d %d %d", want.temperature, want.top_p, want.top_k,
                 want.max_tokens, want.think);
        if (up_cmd(sess->fd, line, reply, sizeof reply) != 0) {
            dead = 1;
            send_error(fd, 502, "server_error", "lost connection to the roaster server");
            goto out;
        }
        if (strncmp(reply, "OK", 2) != 0) {
            send_error(fd, 400, "invalid_request_error", !strncmp(reply, "ERR ", 4) ? reply + 4 : reply);
            goto out;
        }
        dirty = 1;
    }
    put_escaped_cmd(&cmd, "CONT ", prompt.p, prompt.n);
    if (send_all(sess->fd, cmd.p, cmd.n) != 0) {
        dead = 1;
        send_error(fd, 502, "server_error", "lost connection to the roaster server");
        goto out;
    }

    /* ---- stream: answer bytes \0 info \0 (always read to the end so the session stays usable) ---- */
    int phase = 0;
    unsigned char ibuf[4096];
    while (phase < 2) {
        ssize_t got = read(sess->fd, ibuf, sizeof ibuf);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { dead = 1; break; }
        for (ssize_t i = 0; i < got && phase < 2; i++) {
            unsigned char c = ibuf[i];
            if (phase == 1) {
                if (c == 0) phase = 2;
                else buf_put(&info, &c, 1);
                continue;
            }
            switch (c) {
            case 0:
                if (r.pend.n && r.pend.p[r.pend.n - 1] == '\n') r.pend.n--;   /* server's final newline */
                flush_pend(&r, 1);
                phase = 1;
                break;
            case STREAM_THINK_BEGIN: flush_pend(&r, 1); r.mode = 1; break;
            case STREAM_THINK_END:   flush_pend(&r, 1); r.mode = 0; break;
            case STREAM_TOOL_BEGIN:  buf_puts(&r.pend, "<tool_call>");  break;
            case STREAM_TOOL_END:    buf_puts(&r.pend, "</tool_call>"); break;
            case STREAM_SPECIAL_BEGIN: r.special = 1; break;
            case STREAM_SPECIAL_END:   r.special = 0; break;
            default:
                if (!r.special) buf_put(&r.pend, &c, 1);
            }
        }
        flush_pend(&r, 0);
        if (!r.client_ok && !stop_sent) {      /* HTTP client went away: stop generating, keep the context */
            stop_sent = 1;
            api_stop_slot(opt, sess->slot);
        }
    }
    if (dead) {
        if (r.client_ok) {
            if (!r.started) send_error(fd, 502, "server_error", "lost connection to the roaster server");
            else { send_chunk(&r, "", 0); send_all(fd, "0\r\n\r\n", 5); }
        }
        goto out;
    }
    if (!r.client_ok)
        goto out;

    const char *itext = info.p ? info.p : "";
    while (*itext == '\n' || *itext == ' ') itext++;
    int ok = strstr(itext, "Performance:") != NULL;
    if (!ok) {
        const char *m = *itext ? itext : "the roaster server closed the connection";
        int ctx = strstr(m, "too long") || strstr(m, "exceeds slot context");
        int conflict = strstr(m, "unloaded") || strstr(m, "inconsistent") || strstr(m, "busy");
        int code = ctx ? 400 : conflict ? 409 : 500;
        if (!r.started) {
            send_error(fd, code, code == 500 ? "server_error" : "invalid_request_error", m);
        } else {
            Buf e = {0};
            buf_puts(&e, "data: {\"error\":{\"message\":");
            buf_json(&e, m, strlen(m));
            buf_puts(&e, ",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n");
            send_chunk(&r, e.p, e.n);
            send_chunk(&r, "", 0);
            send_all(fd, "0\r\n\r\n", 5);
            free(e.p);
        }
        goto out;
    }

    int by_request = strstr(itext, "Stopped by request") != NULL;
    const char *finish = strstr(itext, "Stopped at max-tokens") ? "length" : "stop";
    int ptok = info_int(itext, "Prompt tokens:");
    int ctok = info_int(itext, "Output tokens:");

    if (!r.stream) {
        Buf o = {0};
        buf_printf(&o, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", r.id, r.created);
        buf_json(&o, r.model, strlen(r.model));
        buf_puts(&o, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        buf_json(&o, r.content.p, r.content.n);
        if (r.reasoning.n) {
            buf_puts(&o, ",\"reasoning_content\":");
            buf_json(&o, r.reasoning.p, r.reasoning.n);
        }
        buf_printf(&o, "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},\"session\":",
                   finish, ptok, ctok, ptok + ctok);
        buf_json(&o, sname, strlen(sname));
        buf_printf(&o, ",\"stopped_by_request\":%s}", by_request ? "true" : "false");
        send_json(fd, 200, o.p, o.n);
        free(o.p);
    } else {
        start_stream(&r);
        Buf b = {0};
        chunk_head(&b, &r);
        buf_printf(&b, "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],\"stopped_by_request\":%s}\n\n",
                   finish, by_request ? "true" : "false");
        if (r.include_usage) {
            chunk_head(&b, &r);
            buf_printf(&b, "\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n",
                       ptok, ctok, ptok + ctok);
        }
        buf_puts(&b, "data: [DONE]\n\n");
        send_chunk(&r, b.p, b.n);
        if (r.client_ok)
            send_all(fd, "0\r\n\r\n", 5);
        free(b.p);
    }

out:
    if (sess && dirty && !dead) {             /* per-request sampling does not stick */
        snprintf(line, sizeof line, "SET %.9g %.9g %d %d %d", base.temperature, base.top_p, base.top_k,
                 base.max_tokens, base.think);
        if (up_cmd(sess->fd, line, reply, sizeof reply) != 0) dead = 1;
    }
    if (sess) session_release(sess, dead);
    jfree(&root);
    free(prompt.p); free(cmd.p); free(info.p); free(sysb.p);
    free(r.pend.p); free(r.content.p); free(r.reasoning.p);
}

/* ============================================================ session endpoints
 *
 *   GET    /v1/sessions                list
 *   POST   /v1/sessions                create   {"id":"x","context_length":8192}  (both optional)
 *   GET    /v1/sessions/{id}           status
 *   DELETE /v1/sessions/{id}           destroy (frees everything, including a saved context)
 *   POST   /v1/sessions/{id}/stop      finish the running request after the current token
 *   POST   /v1/sessions/{id}/reset     empty the context
 *   POST   /v1/sessions/{id}/unload    free the context's RAM (saved to disk; {"discard":true} drops it)
 *   POST   /v1/sessions/{id}/load      bring the context back into RAM
 *   POST   /v1/sessions/{id}/save      {"path":"/dir"}  store the whole context in a directory
 *   POST   /v1/sessions/{id}/restore   {"path":"/dir"}  replace the context with a stored one
 *   PATCH  /v1/sessions/{id}           change settings at any time (same as POST .../config):
 *   POST   /v1/sessions/{id}/config      {"context_length":16384,"cache_route":true,"route_j":2,
 *                                         "route_m":12,"route_alpha":1,"route_p":0,"slide":true,
 *                                         "keep":64,"temperature":0.7,"top_p":0.9,"top_k":20,
 *                                         "max_tokens":1024,"think":false,
 *                                         "system":"..."}  (any subset)
 *                                      "system" is used (and pinned) when the conversation next starts;
 *                                      reset the session to apply it to an existing context.
 *                                      Enlarging context_length pulls context that slid out to SSD
 *                                      back into the window; shrinking pushes the oldest out.
 *                                      Everything that slid out is kept in the state dir (SSD) and
 *                                      deleted when the session is deleted. 409 while a request runs.
 */

/* extra: optional JSON members (starting with a comma) appended to the session object. */
static void send_session_ex(int fd, Session *s, const char *name, const char *extra)
{
    SessInfo si;
    Buf b = {0};
    if (session_query(s, &si) != 0) {
        send_error(fd, 502, "server_error", "lost connection to the roaster server");
        session_release(s, 1);
        return;
    }
    session_json(&b, name, s, &si);
    if (extra && b.n && b.p[b.n - 1] == '}') {
        b.n--;
        buf_puts(&b, extra);
        buf_put(&b, "}", 1);
    }
    send_json(fd, 200, b.p, b.n);
    free(b.p);
    session_release(s, 0);
}

static void send_session(int fd, Session *s, const char *name)
{
    send_session_ex(fd, s, name, NULL);
}

/* HTTP status for an error text from the server's SAVE / RESTORE. */
static int path_error_status(const char *e)
{
    if (strstr(e, "no saved context")) return 404;
    if (strstr(e, "invalid")) return 400;
    if (strstr(e, "busy") || strstr(e, "unloaded") || strstr(e, "inconsistent") || strstr(e, "empty")) return 409;
    return 500;
}

/* POST /v1/sessions/{id}/save | /restore  {"path": "<directory>"} */
static void session_path_cmd(int fd, Session *s, const char *name, const JVal *root, int save)
{
    const JVal *p = jget(root, "path");
    char reply[1024];
    if (!p || p->t != J_STR || p->slen == 0 || p->slen > 4000 || memchr(p->s, '\n', p->slen)) {
        send_error(fd, 400, "invalid_request_error", "'path' (a directory, one line) is required");
        session_release(s, 0);
        return;
    }
    Buf cmd = {0};
    buf_puts(&cmd, save ? "SAVE " : "RESTORE ");
    buf_put(&cmd, p->s, p->slen);
    int rc = up_cmd(s->fd, cmd.p, reply, sizeof reply);
    free(cmd.p);
    if (rc != 0) {
        send_error(fd, 502, "server_error", "lost connection to the roaster server");
        session_release(s, 1);
        return;
    }
    if (strncmp(reply, "OK", 2) != 0) {
        const char *e = !strncmp(reply, "ERR ", 4) ? reply + 4 : reply;
        int code = path_error_status(e);
        send_error(fd, code, code >= 500 ? "server_error" : "invalid_request_error", e);
        session_release(s, 0);
        return;
    }
    if (save) {
        Buf x = {0};
        buf_puts(&x, ",\"saved_to\":");
        buf_json(&x, p->s, p->slen);
        buf_printf(&x, ",\"saved_bytes\":%lld", kv_num(reply, "bytes"));
        send_session_ex(fd, s, name, x.p);
        free(x.p);
    } else {
        send_session(fd, s, name);
    }
}

/* Apply a settings object to an acquired (idle) session, answer with its status, release it. */
static void session_apply_config(int fd, Session *s, const char *name, const JVal *root)
{
    SessInfo si;
    char line[512], reply[512];
    if (session_query(s, &si) != 0) {
        send_error(fd, 502, "server_error", "lost connection to the roaster server");
        session_release(s, 1);
        return;
    }
    SlotConfig nc = si.cfg;
    apply_settings(root, &nc, CFG_SLOT | CFG_SAMPLING);
    const JVal *sysv = jget(root, "system");
    if (sysv && sysv->t == J_STR && sysv->slen > ROASTER_MAX_PROMPT) {
        send_error(fd, 400, "invalid_request_error", "system prompt too long");
        session_release(s, 0);
        return;
    }
    const char *err = slot_config_check(&nc);
    if (err) {
        send_error(fd, 400, "invalid_request_error", err);
        session_release(s, 0);
        return;
    }
    config_line(line, sizeof line, "CONFIG", &nc);
    if (up_cmd(s->fd, line, reply, sizeof reply) != 0) {
        send_error(fd, 502, "server_error", "lost connection to the roaster server");
        session_release(s, 1);
        return;
    }
    if (strncmp(reply, "OK", 2) != 0) {
        const char *e = !strncmp(reply, "ERR ", 4) ? reply + 4 : reply;
        int code = strstr(e, "could not") ? 500 : (strstr(e, "busy") || strstr(e, "unloaded")) ? 409 : 400;
        send_error(fd, code, code == 500 ? "server_error" : "invalid_request_error", e);
        session_release(s, 0);
        return;
    }
    if (sysv && sysv->t == J_STR) {      /* system prompt for the next fresh conversation */
        Buf sc = {0};
        put_escaped_cmd(&sc, "SYS ", sysv->s, sysv->slen);
        if (sc.n) sc.p[--sc.n] = '\0';
        int rc = up_cmd(s->fd, sc.p, reply, sizeof reply);
        free(sc.p);
        if (rc != 0) {
            send_error(fd, 502, "server_error", "lost connection to the roaster server");
            session_release(s, 1);
            return;
        }
        if (strncmp(reply, "OK", 2) != 0) {
            send_error(fd, 400, "invalid_request_error", !strncmp(reply, "ERR ", 4) ? reply + 4 : reply);
            session_release(s, 0);
            return;
        }
    }
    send_session(fd, s, name);
}

static void handle_sessions(int fd, const ApiOptions *opt, const Req *rq, Buf *body, const char *rest)
{
    char name[SESSION_NAME_MAX] = "", action[16] = "", msg[256], reply[512];
    const char *m = rq->method;
    if (*rest == '/') {
        rest++;
        const char *sl = strchr(rest, '/');
        size_t n = sl ? (size_t)(sl - rest) : strlen(rest);
        if (n >= SESSION_NAME_MAX) { send_error(fd, 404, "invalid_request_error", "unknown endpoint"); return; }
        memcpy(name, rest, n);
        name[n] = '\0';
        if (sl) snprintf(action, sizeof action, "%s", sl + 1);
    }

    JVal root;
    memset(&root, 0, sizeof root);
    if (body->n) {
        JP jp = { body->p, body->p + body->n };
        int bad = jval(&jp, &root, 0);
        if (!bad) { jws(&jp); bad = jp.p != jp.end; }
        if (bad || root.t != J_OBJ) {
            jfree(&root);
            send_error(fd, 400, "invalid_request_error", "request body is not valid JSON");
            return;
        }
    }
    Session *s = NULL;
    Buf b = {0};
    int st;

    if (!name[0]) {
        if (!strcmp(m, "GET")) {
            char names[128][SESSION_NAME_MAX];
            int count = 0;
            pthread_mutex_lock(&g_sess_lock);
            for (Session *it = g_sessions; it && count < 128; it = it->next)
                snprintf(names[count++], SESSION_NAME_MAX, "%s", it->name);
            pthread_mutex_unlock(&g_sess_lock);
            buf_puts(&b, "{\"object\":\"list\",\"data\":[");
            for (int i = 0; i < count; i++) {
                SessInfo si;
                if (i) buf_put(&b, ",", 1);
                st = session_acquire(opt, names[i], NULL, 0, 0, &s, msg, sizeof msg);
                if (st == 0 && session_query(s, &si) == 0) {
                    session_json(&b, names[i], s, &si);
                    session_release(s, 0);
                } else {
                    if (st == 0) session_release(s, 1);
                    session_json(&b, names[i], NULL, NULL);
                }
            }
            buf_puts(&b, "]}");
            send_json(fd, 200, b.p, b.n);
        } else if (!strcmp(m, "POST")) {
            SlotConfig cf = opt->defaults;
            const JVal *x = jget(&root, "id");
            char id[SESSION_NAME_MAX];
            if (x && x->t == J_STR && x->s[0]) snprintf(id, sizeof id, "%s", x->s);
            else snprintf(id, sizeof id, "s%lu", __sync_add_and_fetch(&g_sess_counter, 1));
            apply_settings(&root, &cf, CFG_SLOT | CFG_SAMPLING);   /* any setting can be given at creation */
            const char *err = slot_config_check(&cf);
            if (!valid_session_name(id))
                send_error(fd, 400, "invalid_request_error", "invalid session id (1-63 characters: letters, digits, '-', '_', '.')");
            else if (err)
                send_error(fd, 400, "invalid_request_error", err);
            else if ((st = session_acquire(opt, id, &cf, 1, 1, &s, msg, sizeof msg)) != 0)
                send_error(fd, st, st >= 500 ? "server_error" : "invalid_request_error", msg);
            else
                send_session(fd, s, id);
        } else {
            send_error(fd, 405, "invalid_request_error", "use GET or POST");
        }
        goto done;
    }

    if (!action[0]) {
        if (!strcmp(m, "GET") || !strcmp(m, "DELETE") || !strcmp(m, "PATCH")) {
            st = session_acquire(opt, name, NULL, 0, 0, &s, msg, sizeof msg);
            if (st) {
                send_error(fd, st, "invalid_request_error", msg);
            } else if (!strcmp(m, "GET")) {
                send_session(fd, s, name);
            } else if (!strcmp(m, "PATCH")) {
                session_apply_config(fd, s, name, &root);
            } else {
                session_release(s, 1);          /* closes the connection: the server frees the slot */
                buf_puts(&b, "{\"id\":");
                buf_json(&b, name, strlen(name));
                buf_puts(&b, ",\"object\":\"session\",\"deleted\":true}");
                send_json(fd, 200, b.p, b.n);
            }
        } else {
            send_error(fd, 405, "invalid_request_error", "use GET, PATCH or DELETE");
        }
        goto done;
    }

    if (strcmp(m, "POST") != 0) {
        send_error(fd, 405, "invalid_request_error", "use POST");
        goto done;
    }
    if (!strcmp(action, "stop")) {
        int slot = -1, busy = 0;
        pthread_mutex_lock(&g_sess_lock);
        Session *it = session_find_locked(name);
        if (it) { slot = it->slot; busy = it->busy; }
        pthread_mutex_unlock(&g_sess_lock);
        if (!it) {
            snprintf(msg, sizeof msg, "no such session: '%s'", name);
            send_error(fd, 404, "invalid_request_error", msg);
        } else {
            int hit = busy && slot >= 0 && api_stop_slot(opt, slot);
            buf_puts(&b, "{\"id\":");
            buf_json(&b, name, strlen(name));
            buf_printf(&b, ",\"object\":\"session\",\"stopped\":%s}", hit ? "true" : "false");
            send_json(fd, 200, b.p, b.n);
        }
        goto done;
    }
    if (!strcmp(action, "save") || !strcmp(action, "restore")) {
        st = session_acquire(opt, name, NULL, 0, 0, &s, msg, sizeof msg);
        if (st) send_error(fd, st, "invalid_request_error", msg);
        else session_path_cmd(fd, s, name, &root, !strcmp(action, "save"));
        goto done;
    }
    if (!strcmp(action, "config")) {
        st = session_acquire(opt, name, NULL, 0, 0, &s, msg, sizeof msg);
        if (st) send_error(fd, st, "invalid_request_error", msg);
        else session_apply_config(fd, s, name, &root);
        goto done;
    }
    const char *command = NULL;
    if (!strcmp(action, "reset")) command = "RESET";
    else if (!strcmp(action, "load")) command = "LOAD";
    else if (!strcmp(action, "unload")) {
        const JVal *d = jget(&root, "discard");
        command = d && d->t == J_BOOL && d->b ? "UNLOAD DISCARD" : "UNLOAD";
    } else {
        send_error(fd, 404, "invalid_request_error", "unknown endpoint");
        goto done;
    }
    st = session_acquire(opt, name, NULL, 0, 0, &s, msg, sizeof msg);
    if (st) {
        send_error(fd, st, "invalid_request_error", msg);
    } else if (up_cmd(s->fd, command, reply, sizeof reply) != 0) {
        send_error(fd, 502, "server_error", "lost connection to the roaster server");
        session_release(s, 1);
    } else if (strncmp(reply, "OK", 2) != 0) {
        const char *e = !strncmp(reply, "ERR ", 4) ? reply + 4 : reply;
        send_error(fd, strstr(e, "could not") ? 500 : 409,
                   strstr(e, "could not") ? "server_error" : "invalid_request_error", e);
        session_release(s, 0);
    } else {
        send_session(fd, s, name);
    }
done:
    free(b.p);
    jfree(&root);
}

/* ============================================================ connections */

typedef struct { int fd; const ApiOptions *opt; } Conn;

static void handle_models(int fd, const ApiOptions *opt)
{
    Buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
    buf_json(&b, opt->model_name, strlen(opt->model_name));
    buf_puts(&b, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"roaster\"}]}");
    send_json(fd, 200, b.p, b.n);
    free(b.p);
}

static void *conn_main(void *arg)
{
    Conn *c = arg;
    int fd = c->fd;
    const ApiOptions *opt = c->opt;
    free(c);

    struct timeval tv = { .tv_sec = 30 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    Req rq;
    Buf body = {0};
    int st = read_request(fd, &rq, &body);
    if (st > 0) {
        send_error(fd, st, "invalid_request_error", reason(st));
        goto done;
    }
    if (st < 0)
        goto done;

    char *q = strchr(rq.path, '?');
    if (q) *q = '\0';

    if (!strcmp(rq.method, "OPTIONS")) {
        static const char r[] =
            "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, PATCH, DELETE, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Authorization, Content-Type, X-Session-Id\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send_all(fd, r, sizeof(r) - 1);
        goto done;
    }
    if (!strcmp(rq.path, "/health")) {
        const char *ok = "{\"status\":\"ok\"}";
        send_json(fd, 200, ok, strlen(ok));
        goto done;
    }
    if (opt->api_key && *opt->api_key) {
        char want[600];
        snprintf(want, sizeof want, "Bearer %s", opt->api_key);
        if (strcmp(rq.auth, want) != 0) {
            send_error(fd, 401, "invalid_request_error", "invalid or missing API key");
            goto done;
        }
    }
    if (!strcmp(rq.path, "/v1/models") || !strcmp(rq.path, "/models")) {
        handle_models(fd, opt);
    } else if (!strcmp(rq.path, "/v1/chat/completions") || !strcmp(rq.path, "/chat/completions")) {
        if (strcmp(rq.method, "POST") != 0) send_error(fd, 405, "invalid_request_error", "use POST");
        else handle_chat(fd, opt, &rq, &body);
    } else if (!strncmp(rq.path, "/v1/sessions", 12) && (rq.path[12] == '\0' || rq.path[12] == '/')) {
        handle_sessions(fd, opt, &rq, &body, rq.path + 12);
    } else {
        send_error(fd, 404, "invalid_request_error", "unknown endpoint");
    }
done:
    free(body.p);
    close(fd);
    return NULL;
}

static void on_signal(int sig) { (void)sig; g_stop = 1; }

int api_run(const ApiOptions *opt)
{
    const char *host = !strcmp(opt->host, "localhost") ? "127.0.0.1" : opt->host;
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons((uint16_t)opt->port) };
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) {
        fprintf(stderr, "Invalid --host (IPv4 address expected): %s\n", opt->host);
        return -1;
    }
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return -1; }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(srv, (struct sockaddr *)&a, sizeof a) != 0 || listen(srv, 64) != 0) {
        perror("bind/listen");
        close(srv);
        return -1;
    }
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa = { .sa_handler = on_signal };     /* no SA_RESTART: accept() must return */
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    printf("OpenAI-compatible API on http://%s:%d/v1  (roaster socket: %s, think default: %s)\n",
           opt->host, opt->port, opt->socket_path, opt->defaults.think ? "on" : "off");
    fflush(stdout);

    while (!g_stop) {
        int fd = accept(srv, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        Conn *c = malloc(sizeof *c);
        if (!c) { close(fd); continue; }
        c->fd = fd;
        c->opt = opt;
        pthread_t t;
        if (pthread_create(&t, NULL, conn_main, c) != 0) { close(fd); free(c); continue; }
        pthread_detach(t);
    }
    printf("\nShutting down...\n");
    close(srv);
    return 0;
}