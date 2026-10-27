/*
 * Roaster - local MoE inference.
 *
 *   roaster serve --model <file.gguf> [--max-slots N] [--expert-cache N] [--threads N] [--socket PATH]
 *       Loads the model and waits for chat clients. Ctrl+C stops it.
 *
 *   roaster chat [--socket PATH] [slot options]
 *       Opens a chat on its own slot. The slot is destroyed when the chat ends (exit, Ctrl+D, Ctrl+C).
 *
 * Server-wide settings are options of `serve`; per-slot settings (context, routing, sampling,
 * max tokens) are options of `chat`. Slot handling lives in slots.c, networking in server.c/client.c.
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
#include "protocol.h"

#define DEFAULT_MAX_SLOTS     8
#define DEFAULT_EXPERT_CACHE  16

static void usage(const char *prog)
{
    printf(
        "Usage:\n"
        "  %s serve --model <file.gguf> [--max-slots N] [--expert-cache N] [--threads N] [--socket PATH]\n"
        "  %s chat  [--socket PATH] [--context N] [--cache-route 0|1] [--route-j N] [--route-m N]\n"
        "           [--route-alpha X] [--route-p X] [--temp X] [--top-p X] [--top-k N] [--max-tokens N]\n"
        "           [--think 0|1] [--color auto|on|off]\n"
        "\n"
        "serve: --max-slots (8)  --expert-cache per layer (16)  --threads, 0 = all (0)\n"
        "chat:  defaults are context %d, cache-route %d, route-j/m %d/%d, alpha %.1f, p %.1f,\n"
        "       temp %.2f, top-p %.2f, top-k %d, max-tokens %d, think %d\n"
        "Both:  --socket defaults to %s\n",
        prog, prog, DEFAULT_CONTEXT, DEFAULT_CACHE_ROUTE, DEFAULT_ROUTE_J, DEFAULT_ROUTE_M,
        DEFAULT_ROUTE_ALPHA, DEFAULT_ROUTE_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K,
        DEFAULT_MAX_TOKENS, DEFAULT_THINK, ROASTER_DEFAULT_SOCKET);
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
    const char *model_path = NULL, *socket_path = ROASTER_DEFAULT_SOCKET;
    int max_slots = DEFAULT_MAX_SLOTS, expert_cache = DEFAULT_EXPERT_CACHE, threads = 0;

    for (int i = 2; i < argc; i++) {
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(argv[i], "--model") && v)             model_path = argv[++i];
        else if (!strcmp(argv[i], "--max-slots") && v)    max_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--expert-cache") && v) expert_cache = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && v)      threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--socket") && v)       socket_path = argv[++i];
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
    if (slots_init(&slots, &weights, &tokenizer, max_slots) != 0) {
        fprintf(stderr, "Failed to allocate slot table\n");
        goto tokenizer_cleanup;
    }

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

static int cmd_chat(int argc, char **argv)
{
    const char *socket_path = ROASTER_DEFAULT_SOCKET;
    SlotConfig c;
    slot_config_defaults(&c);
    int color = -1;

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!v) { fprintf(stderr, "Missing value for %s\n", a); return 1; }
        i++;
        if (!strcmp(a, "--socket"))           socket_path = v;
        else if (!strcmp(a, "--context"))     c.context_length = (size_t)strtoull(v, NULL, 10);
        else if (!strcmp(a, "--cache-route")) c.cache_route = atoi(v);
        else if (!strcmp(a, "--route-j"))     c.route_j = atoi(v);
        else if (!strcmp(a, "--route-m"))     c.route_m = atoi(v);
        else if (!strcmp(a, "--route-alpha")) c.route_alpha = strtof(v, NULL);
        else if (!strcmp(a, "--route-p"))     c.route_p = strtof(v, NULL);
        else if (!strcmp(a, "--temp"))        c.temperature = strtof(v, NULL);
        else if (!strcmp(a, "--top-p"))       c.top_p = strtof(v, NULL);
        else if (!strcmp(a, "--top-k"))       c.top_k = atoi(v);
        else if (!strcmp(a, "--max-tokens"))  c.max_tokens = atoi(v);
        else if (!strcmp(a, "--think"))
            c.think = (!strcmp(v, "1") || !strcmp(v, "on")) ? 1 :
                      (!strcmp(v, "0") || !strcmp(v, "off")) ? 0 : -1;
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
    return client_run(socket_path, &c, color);
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "serve"))
        return cmd_serve(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "chat"))
        return cmd_chat(argc, argv);
    usage(argv[0]);
    return argc >= 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) ? 0 : 1;
}