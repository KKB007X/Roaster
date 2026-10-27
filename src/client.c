#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "client.h"
#include "protocol.h"

enum { M_ANSWER, M_THINK, M_TOOL };

typedef struct {
    int color;
    int mode;      /* M_* */
    int special;   /* inside a literal special token */
    int last;      /* last character printed */
} Render;

/* answer: default | think: dim italic | tool call: yellow | other special tokens: magenta */
static void apply_style(const Render *r)
{
    if (!r->color)
        return;
    fputs("\x1b[0m", stdout);
    if (r->special)
        fputs("\x1b[35m", stdout);
    else if (r->mode == M_THINK)
        fputs("\x1b[2;3m", stdout);
    else if (r->mode == M_TOOL)
        fputs("\x1b[33m", stdout);
}

static void render_byte(Render *r, int ch)
{
    switch (ch) {
    case STREAM_THINK_BEGIN:
        r->mode = M_THINK;
        apply_style(r);
        fputs("<think>\n", stdout);
        r->last = '\n';
        break;
    case STREAM_THINK_END:
        if (r->last != '\n')
            putchar('\n');
        fputs("</think>", stdout);
        r->mode = M_ANSWER;
        apply_style(r);
        fputs("\n\n", stdout);
        r->last = '\n';
        break;
    case STREAM_TOOL_BEGIN:
        r->mode = M_TOOL;
        apply_style(r);
        fputs("<tool_call>", stdout);
        r->last = '>';
        break;
    case STREAM_TOOL_END:
        fputs("</tool_call>", stdout);
        r->mode = M_ANSWER;
        apply_style(r);
        r->last = '>';
        break;
    case STREAM_SPECIAL_BEGIN:
        r->special = 1;
        apply_style(r);
        break;
    case STREAM_SPECIAL_END:
        r->special = 0;
        apply_style(r);
        break;
    default:
        putchar(ch);
        r->last = ch;
    }
}

int client_run(const char *socket_path, const SlotConfig *c, int color)
{
    struct sockaddr_un addr = { .sun_family = AF_UNIX };
    if (strlen(socket_path) >= sizeof(addr.sun_path)) {
        fprintf(stderr, "Socket path too long: %s\n", socket_path);
        return 1;
    }
    strcpy(addr.sun_path, socket_path);
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0 || connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "Cannot connect to %s: is the server running?\n", socket_path);
        return 1;
    }
    FILE *in = fdopen(dup(fd), "r");
    FILE *out = fdopen(fd, "w");

    fprintf(out, "CREATE %zu %d %d %d %.9g %.9g %.9g %.9g %d %d %d\n",
            c->context_length, c->cache_route, c->route_j, c->route_m, c->route_alpha,
            c->route_p, c->temperature, c->top_p, c->top_k, c->max_tokens, c->think);
    fflush(out);

    char reply[512];
    if (!fgets(reply, sizeof(reply), in)) {
        fprintf(stderr, "Server closed the connection\n");
        return 1;
    }
    if (strncmp(reply, "OK ", 3) != 0) {
        fprintf(stderr, "Server: %s", reply);
        return 1;
    }
    int tty = isatty(STDIN_FILENO);
    if (color < 0)
        color = isatty(STDOUT_FILENO) && !getenv("NO_COLOR");
    /* Tokens must show up the moment they arrive, not when a line or buffer fills. */
    setvbuf(stdout, NULL, _IONBF, 0);
    fprintf(stderr, "Connected (slot %d, thinking %s). Type a prompt; 'exit' or Ctrl+D to leave.\n",
            atoi(reply + 3), c->think ? "on" : "off");

    char line[ROASTER_MAX_PROMPT + 2];
    for (;;) {
        if (tty) {
            fputs("> ", stdout);
            fflush(stdout);
        }
        if (!fgets(line, sizeof(line), stdin))
            break;
        line[strcspn(line, "\r\n")] = '\0';
        if (!line[0])
            continue;
        if (!strcmp(line, "exit") || !strcmp(line, "quit"))
            break;

        fprintf(out, "RUN %s\n", line);
        fflush(out);

        Render r = { .color = color, .mode = M_ANSWER, .special = 0, .last = '\n' };
        int ch;
        while ((ch = fgetc(in)) != EOF && ch != '\0')      /* answer, token by token */
            render_byte(&r, ch);
        if (color)
            fputs("\x1b[0m", stdout);
        if (ch == EOF) {
            fprintf(stderr, "\nServer closed the connection\n");
            return 1;
        }
        while ((ch = fgetc(in)) != EOF && ch != '\0')      /* performance / errors */
            fputc(ch, stderr);
    }
    fclose(out);     /* the server destroys the slot when the connection closes */
    fclose(in);
    return 0;
}