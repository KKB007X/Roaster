#ifndef PROTOCOL_H
#define PROTOCOL_H

/*
 * Client <-> server protocol (Unix socket, line based).
 *
 *   client: CREATE <context> <cache_route> <j> <m> <alpha> <p> <temp> <top_p> <top_k> <max_tokens> <think> <slide> <keep>\n
 *           (the last two fields are optional: default slide on, keep 64)
 *   server: OK <slot id>\n   |   ERR <message>\n
 *
 * A client's slot lives exactly as long as its connection. Commands that run a prompt:
 *
 *   client: RUN  <prompt on one line>\n       fresh conversation (the old context is discarded)
 *   client: RUNE <prompt, \n \r \\ escaped>\n same, with escapes
 *   client: CONT <prompt, \n \r \\ escaped>\n continue: append a user turn to the existing context
 *   server: <answer stream> \0 <performance / error text> \0
 *
 * Context control (one-line answer: "OK ...\n" or "ERR <message>\n"):
 *
 *   SET <temp> <top_p> <top_k> <max_tokens> <think>   change sampling / limits of the slot
 *   SAVE <directory>                                  write the whole context (window, recurrent state,
 *                                                     SSD spill, system prompt) into a directory:
 *                                                     OK bytes=N. Works on loaded and unloaded slots.
 *   RESTORE <directory>                               replace the context with a saved one (the slot must
 *                                                     be loaded; the window grows to the saved size if
 *                                                     smaller). Errors: "no saved context found ...",
 *                                                     "invalid saved context ...", "invalid path ..."
 *   SYS <system prompt, \n \r \\ escaped>            system prompt for the next fresh conversation (RUN, or
 *                                                     CONT on an empty context). Empty text clears it.
 *                                                     It is wrapped in a system turn and pinned: it stays
 *                                                     in the attention window when the window slides.
 *   CONFIG <same 13 fields as CREATE>                 change EVERYTHING of an idle slot: context
 *                                                     window, routing, sampling, sliding. A larger
 *                                                     window pulls slid-out context back from SSD;
 *                                                     a smaller one pushes the oldest out to SSD.
 *                                                     (The window of an unloaded slot is fixed.)
 *   RESET                                             empty the context (also drops a saved one)
 *   UNLOAD [DISCARD]                                  free the context's RAM; it is saved to the
 *                                                     state dir unless DISCARD is given
 *   LOAD                                              bring the context back into RAM
 *   INFO                                              OK loaded=1 position=N context=N ram=N state=N broken=0 busy=0
 *                                                     window=N spilled=N spill_bytes=N slide=1 keep=N
 *                                                     plus every setting as key=value
 *   STOP <slot id>                                    from ANY connection: finish the running request
 *                                                     of that slot after the current token (OK 1 / OK 0)
 *
 * Answer stream: every generated token is written (and flushed) as soon as it is sampled.
 * Plain text is sent as-is. Control bytes 0x01..0x06 are in-band markers that tell the client
 * what kind of text follows, so it can colour it. The server strips these bytes from model text.
 */
#define ROASTER_DEFAULT_SOCKET "/tmp/roaster.sock"
#define ROASTER_MAX_PROMPT     16000

#define STREAM_THINK_BEGIN   '\x01'  /* <think> opened: reasoning text follows            */
#define STREAM_THINK_END     '\x02'  /* </think>: back to the normal answer               */
#define STREAM_TOOL_BEGIN    '\x03'  /* <tool_call> opened                                */
#define STREAM_TOOL_END      '\x04'  /* </tool_call> closed                               */
#define STREAM_SPECIAL_BEGIN '\x05'  /* any other special token; its literal text follows */
#define STREAM_SPECIAL_END   '\x06'

#endif
