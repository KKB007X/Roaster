/*
 * Roaster - local MoE inference.
 *
 *   roaster serve --model <file.gguf> [--max-slots N] [--expert-cache N] [--threads N] [--socket PATH]
 *                 [--state-dir DIR]
 *       Loads the model and waits for chat clients. Ctrl+C stops it.
 *
 *   roaster chat [--socket PATH] [slot options]
 *       Opens a chat on its own slot. The slot is destroyed when the chat ends (exit, Ctrl+D, Ctrl+C).
 *
 *   roaster api [--socket PATH] [--host H] [--port N] [--api-key K] [--model-name NAME] [slot options]
 *       OpenAI-compatible HTTP API in front of a running `roaster serve`.
 *
 * Server-wide settings are options of `serve`; per-slot settings (context, routing, sampling,
 * max tokens) are options of `chat` / `api`. Slot handling lives in slots.c, networking in
 * server.c/client.c/api.c.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "tokenizer.h"
#include "weights.h"
#include "model.h"
#include "slots.h"
#include "server.h"
#include "client.h"
#include "api.h"
#include "protocol.h"

#define DEFAULT_MAX_SLOTS     8
#define DEFAULT_EXPERT_CACHE  16
#define DEFAULT_API_HOST      "127.0.0.1"
#define DEFAULT_API_PORT      8080
#define DEFAULT_MODEL_NAME    "roaster"

static void usage(const char *prog)
{
    printf(
        "Usage:\n"
        "  %s serve --model <file.gguf> [--max-slots N] [--expert-cache N] [--threads N] [--socket PATH]\n"
        "           [--state-dir DIR] [--save-root DIR]\n"
        "  %s chat  [--socket PATH] [--context N] [--cache-route 0|1] [--route-j N] [--route-m N]\n"
        "           [--route-alpha X] [--route-p X] [--temp X] [--top-p X] [--top-k N] [--max-tokens N]\n"
        "           [--think 0|1] [--slide 0|1] [--keep N] [--system TEXT | --system-file PATH]\n"
        "           [--color auto|on|off]\n"
        "  %s api   [--socket PATH] [--host ADDR] [--port N] [--api-key KEY] [--model-name NAME]\n"
        "           [slot options as for chat; --think defaults to 0 here]\n"
        "\n"
        "serve: --max-slots (8)  --expert-cache per layer (16)  --threads, 0 = all (0)\n"
        "       --state-dir (%s): where unloaded contexts and slid-out context are stored (use an SSD)\n"
        "       --save-root: confine directories used by the API's save/restore calls to this folder\n"
        "chat:  defaults are context %d, cache-route %d, route-j/m %d/%d, alpha %.1f, p %.1f,\n"
        "       temp %.2f, top-p %.2f, top-k %d, max-tokens %d, think %d, slide %d, keep %d\n"
        "       (slide: when the context window is full, old KV entries slide out to --state-dir on SSD\n"
        "        and come back if the window is enlarged; keep: first tokens that are never slid out)\n"
        "api:   defaults to %s:%d; the API key may also be set with ROASTER_API_KEY\n"
        "All:   --socket defaults to %s\n",
        prog, prog, prog, DEFAULT_STATE_DIR, DEFAULT_CONTEXT, DEFAULT_CACHE_ROUTE, DEFAULT_ROUTE_J, DEFAULT_ROUTE_M,
        DEFAULT_ROUTE_ALPHA, DEFAULT_ROUTE_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K,
        DEFAULT_MAX_TOKENS, DEFAULT_THINK, DEFAULT_SLIDE, DEFAULT_KEEP, DEFAULT_API_HOST, DEFAULT_API_PORT, ROASTER_DEFAULT_SOCKET);
}

/* The tokenizer lives in the GGUF metadata, read straight from the model file. */
static int load_tokenizer(const char *path, Tokenizer *tokenizer)
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        perror("Opening model for tokenizer");
        return -1;
    }
    uint32_t magic, version;
    uint64_t tensor_count, metadata_count;
    int ok = fread(&magic, sizeof(magic), 1, file) == 1 &&
             fread(&version, sizeof(version), 1, file) == 1 &&
             fread(&tensor_count, sizeof(tensor_count), 1, file) == 1 &&
             fread(&metadata_count, sizeof(metadata_count), 1, file) == 1 &&
             magic == 0x46554747;
    if (!ok) {
        fprintf(stderr, "Invalid or unreadable GGUF header\n");
        fclose(file);
        return -1;
    }
    int rc = tokenizer_load_metadata(file, metadata_count, tokenizer);
    fclose(file);
    if (rc != 0)
        fprintf(stderr, "Failed to load tokenizer metadata\n");
    return rc;
}

static int cmd_serve(int argc, char **argv)
{
    const char *model_path = NULL, *socket_path = ROASTER_DEFAULT_SOCKET, *state_dir = DEFAULT_STATE_DIR;
    const char *save_root = NULL;
    int max_slots = DEFAULT_MAX_SLOTS, expert_cache = DEFAULT_EXPERT_CACHE, threads = 0;

    for (int i = 2; i < argc; i++) {
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(argv[i], "--model") && v)             model_path = argv[++i];
        else if (!strcmp(argv[i], "--max-slots") && v)    max_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--expert-cache") && v) expert_cache = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && v)      threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--socket") && v)       socket_path = argv[++i];
        else if (!strcmp(argv[i], "--state-dir") && v)    state_dir = argv[++i];
        else if (!strcmp(argv[i], "--save-root") && v)    save_root = argv[++i];
        else { fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]); return 1; }
    }
    if (!model_path || max_slots <= 0 || expert_cache <= 0 || threads < 0) {
        fprintf(stderr, "Invalid server configuration or missing --model\n");
        return 1;
    }
    if (threads > 0)
        omp_set_num_threads(threads);

    Weights weights;
    Tokenizer tokenizer = {0};
    SlotManager slots;
    int rc = 1;

    printf("Loading model weights...\n");
    if (weights_load(&weights, model_path, expert_cache) != 0) {
        fprintf(stderr, "Failed to load model weights\n");
        return 1;
    }
    printf("Loading tokenizer...\n");
    if (load_tokenizer(model_path, &tokenizer) != 0)
        goto weights_cleanup;
    if (slots_init(&slots, &weights, &tokenizer, max_slots, state_dir) != 0) {
        fprintf(stderr, "Failed to allocate slot table\n");
        goto tokenizer_cleanup;
    }

    if (save_root)
        slots_set_save_root(&slots, save_root);
    printf("Roaster ready: %d slots, expert cache %d per layer, %d CPU threads, vocabulary %llu\n",
           max_slots, expert_cache, threads > 0 ? threads : omp_get_max_threads(),
           (unsigned long long)tokenizer.vocab_size);
    rc = server_run(&slots, socket_path) == 0 ? 0 : 1;

    slots_free(&slots);
tokenizer_cleanup:
    tokenizer_free(&tokenizer);
weights_cleanup:
    weights_free(&weights);
    puts("Roaster stopped.");
    return rc;
}

/* Shared per-slot options. Returns 1 = handled, 0 = not a slot option. */
static int parse_slot_option(SlotConfig *c, const char *a, const char *v)
{
    if (!strcmp(a, "--context"))         c->context_length = (size_t)strtoull(v, NULL, 10);
    else if (!strcmp(a, "--cache-route")) c->cache_route = atoi(v);
    else if (!strcmp(a, "--route-j"))     c->route_j = atoi(v);
    else if (!strcmp(a, "--route-m"))     c->route_m = atoi(v);
    else if (!strcmp(a, "--route-alpha")) c->route_alpha = strtof(v, NULL);
    else if (!strcmp(a, "--route-p"))     c->route_p = strtof(v, NULL);
    else if (!strcmp(a, "--temp"))        c->temperature = strtof(v, NULL);
    else if (!strcmp(a, "--top-p"))       c->top_p = strtof(v, NULL);
    else if (!strcmp(a, "--top-k"))       c->top_k = atoi(v);
    else if (!strcmp(a, "--max-tokens"))  c->max_tokens = atoi(v);
    else if (!strcmp(a, "--slide"))       c->slide = (!strcmp(v, "1") || !strcmp(v, "on")) ? 1 :
                                                     (!strcmp(v, "0") || !strcmp(v, "off")) ? 0 : -1;
    else if (!strcmp(a, "--keep"))        c->keep = atoi(v);
    else if (!strcmp(a, "--think"))
        c->think = (!strcmp(v, "1") || !strcmp(v, "on")) ? 1 :
                   (!strcmp(v, "0") || !strcmp(v, "off")) ? 0 : -1;
    else return 0;
    return 1;
}

static int cmd_chat(int argc, char **argv)
{
    const char *socket_path = ROASTER_DEFAULT_SOCKET;
    SlotConfig c;
    slot_config_defaults(&c);
    int color = -1;
    const char *system = NULL;
    char *system_buf = NULL;

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!v) { fprintf(stderr, "Missing value for %s\n", a); return 1; }
        i++;
        if (!strcmp(a, "--socket"))           socket_path = v;
        else if (!strcmp(a, "--system"))      system = v;
        else if (!strcmp(a, "--system-file")) {
            FILE *f = fopen(v, "rb");
            if (!f) { perror(v); return 1; }
            system_buf = calloc(1, ROASTER_MAX_PROMPT + 1);
            size_t n = system_buf ? fread(system_buf, 1, ROASTER_MAX_PROMPT, f) : 0;
            fclose(f);
            if (!system_buf || n == 0) { fprintf(stderr, "Cannot read system prompt from %s\n", v); return 1; }
            system = system_buf;
        }
        else if (parse_slot_option(&c, a, v)) { /* handled */ }
        else if (!strcmp(a, "--color")) {
            if (!strcmp(v, "on")) color = 1;
            else if (!strcmp(v, "off")) color = 0;
            else if (!strcmp(v, "auto")) color = -1;
            else { fprintf(stderr, "--color must be auto, on or off\n"); return 1; }
        }
        else { fprintf(stderr, "Unknown option: %s\n", a); return 1; }
    }
    const char *err = slot_config_check(&c);
    if (err) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    int rc = client_run(socket_path, &c, color, system);
    free(system_buf);
    return rc;
}

static int cmd_api(int argc, char **argv)
{
    ApiOptions o = {0};
    o.host = DEFAULT_API_HOST;
    o.port = DEFAULT_API_PORT;
    o.socket_path = ROASTER_DEFAULT_SOCKET;
    o.model_name = DEFAULT_MODEL_NAME;
    o.api_key = getenv("ROASTER_API_KEY");
    slot_config_defaults(&o.defaults);
    o.defaults.think = 0;      /* reasoning is opt-in for API use (--think 1, or "think": true per request) */

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!v) { fprintf(stderr, "Missing value for %s\n", a); return 1; }
        i++;
        if (!strcmp(a, "--socket"))          o.socket_path = v;
        else if (!strcmp(a, "--host"))       o.host = v;
        else if (!strcmp(a, "--port"))       o.port = atoi(v);
        else if (!strcmp(a, "--api-key"))    o.api_key = v;
        else if (!strcmp(a, "--model-name")) o.model_name = v;
        else if (parse_slot_option(&o.defaults, a, v)) { /* handled */ }
        else { fprintf(stderr, "Unknown option: %s\n", a); return 1; }
    }
    if (o.port <= 0 || o.port > 65535) {
        fprintf(stderr, "--port must be between 1 and 65535\n");
        return 1;
    }
    const char *err = slot_config_check(&o.defaults);
    if (err) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    return api_run(&o) == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "serve"))
        return cmd_serve(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "chat"))
        return cmd_chat(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "api"))
        return cmd_api(argc, argv);
    usage(argv[0]);
    return argc >= 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) ? 0 : 1;
}
