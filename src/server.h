#ifndef SERVER_H
#define SERVER_H
#include "slots.h"

/* Listen on a Unix socket; every connection gets its own slot (destroyed on disconnect).
 * Returns when SIGINT/SIGTERM is received, after all clients are disconnected. */
int server_run(SlotManager *sm, const char *socket_path);

#endif
