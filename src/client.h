#ifndef CLIENT_H
#define CLIENT_H
#include "slots.h"

/* system: optional system prompt (NULL = none); it is pinned in the attention window.
 * Connect, create a slot with `config`, and chat until EOF / "exit". Returns a process exit code.
 * color: 1 = always, 0 = never, -1 = auto (stdout is a tty and NO_COLOR is unset). */
int client_run(const char *socket_path, const SlotConfig *config, int color, const char *system);

#endif