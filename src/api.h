#ifndef API_H
#define API_H
#include "slots.h"

typedef struct {
    const char *host;          /* bind address, e.g. 127.0.0.1 */
    int port;
    const char *socket_path;   /* roaster serve socket */
    const char *model_name;    /* reported by /v1/models */
    const char *api_key;       /* NULL = no auth */
    SlotConfig defaults;       /* per-request slot defaults (overridable in the JSON body) */
} ApiOptions;

/* Serve an OpenAI-compatible HTTP API until SIGINT/SIGTERM. */
int api_run(const ApiOptions *opt);

#endif