#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "server.h"
#include "protocol.h"

typedef struct Client {
    int fd;
    int done;
    pthread_t thread;
    struct Client *next;
} Client;

static SlotManager *g_sm;
/* Slot scheduling (round robin per token) is handled inside slot_run(). */
static pthread_mutex_t g_clients_lock = PTHREAD_MUTEX_INITIALIZER;
static Client *g_clients;
static volatile sig_atomic_t g_stop;

static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
    g_slots_abort = 1;
}

static void reply_err(FILE *out, const char *msg)
{
    fprintf(out, "ERR %s\n", msg);
    fflush(out);
}

static void *client_main(void *arg)
{
    Client *c = arg;
    FILE *in = fdopen(dup(c->fd), "r");
    FILE *out = fdopen(dup(c->fd), "w");
    int id = -1;
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;

    while (in && out && (len = getline(&line, &cap, in)) >= 0) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';

        if (strncmp(line, "CREATE ", 7) == 0) {
            if (id >= 0) { reply_err(out, "slot already created"); continue; }
            SlotConfig cf;
            int n = sscanf(line + 7, "%zu %d %d %d %f %f %f %f %d %d %d",
                           &cf.context_length, &cf.cache_route, &cf.route_j, &cf.route_m,
                           &cf.route_alpha, &cf.route_p, &cf.temperature, &cf.top_p,
                           &cf.top_k, &cf.max_tokens, &cf.think);
            if (n != 11) { reply_err(out, "malformed CREATE"); continue; }
            const char *err = slot_config_check(&cf);
            if (err) { reply_err(out, err); continue; }
            id = slot_create(g_sm, &cf);
            if (id < 0) { reply_err(out, "could not create slot (no free slot or out of memory; see server log)"); continue; }
            fprintf(out, "OK %d\n", id);
            fflush(out);
        } else if (strncmp(line, "RUN ", 4) == 0) {
            if (id < 0) { reply_err(out, "no slot"); continue; }
            if (len - 4 > ROASTER_MAX_PROMPT) { reply_err(out, "prompt too long"); continue; }
            char *info = NULL;
            size_t info_len = 0;
            FILE *inf = open_memstream(&info, &info_len);
            slot_run(g_sm, id, line + 4, out, inf);
            fclose(inf);
            fputc('\0', out);
            fwrite(info, 1, info_len, out);
            fputc('\0', out);
            fflush(out);
            free(info);
            if (ferror(out))
                break;
        } else {
            reply_err(out, "unknown command");
        }
    }

    if (id >= 0) {
        slot_destroy(g_sm, id);
    }
    free(line);

    pthread_mutex_lock(&g_clients_lock);
    c->done = 1;
    pthread_mutex_unlock(&g_clients_lock);
    if (in) fclose(in);
    if (out) fclose(out);
    close(c->fd);
    return NULL;
}

/* Join and free clients that have finished (all = also wait for the rest). */
static void reap_clients(int all)
{
    pthread_mutex_lock(&g_clients_lock);
    Client **link = &g_clients;
    while (*link) {
        Client *c = *link;
        if (all && !c->done)
            shutdown(c->fd, SHUT_RDWR);        /* wakes a blocked read */
        if (all || c->done) {
            pthread_mutex_unlock(&g_clients_lock);
            pthread_join(c->thread, NULL);
            pthread_mutex_lock(&g_clients_lock);
            *link = c->next;
            free(c);
            link = &g_clients;                 /* list may have changed while unlocked */
            continue;
        }
        link = &c->next;
    }
    pthread_mutex_unlock(&g_clients_lock);
}

int server_run(SlotManager *sm, const char *socket_path)
{
    g_sm = sm;
    struct sockaddr_un addr = { .sun_family = AF_UNIX };
    if (strlen(socket_path) >= sizeof(addr.sun_path)) {
        fprintf(stderr, "Socket path too long: %s\n", socket_path);
        return -1;
    }
    strcpy(addr.sun_path, socket_path);

    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return -1; }
    unlink(socket_path);                       /* stale socket from a previous run */
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(srv, 16) != 0) {
        perror(socket_path);
        close(srv);
        return -1;
    }

    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa = { .sa_handler = on_signal };   /* no SA_RESTART: accept() must return */
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    printf("Listening on %s (Ctrl+C to stop)\n", socket_path);
    fflush(stdout);

    while (!g_stop) {
        int fd = accept(srv, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        reap_clients(0);
        Client *c = calloc(1, sizeof(*c));
        if (!c) { close(fd); continue; }
        c->fd = fd;
        pthread_mutex_lock(&g_clients_lock);
        c->next = g_clients;
        g_clients = c;
        pthread_mutex_unlock(&g_clients_lock);
        if (pthread_create(&c->thread, NULL, client_main, c) != 0) {
            pthread_mutex_lock(&g_clients_lock);
            g_clients = c->next;
            pthread_mutex_unlock(&g_clients_lock);
            close(fd);
            free(c);
        }
    }

    printf("\nShutting down...\n");
    g_slots_abort = 1;
    close(srv);
    unlink(socket_path);
    reap_clients(1);
    return 0;
}