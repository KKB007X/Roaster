#ifndef PROTOCOL_H
#define PROTOCOL_H

/*
 * Client <-> server protocol (Unix socket, line based).
 *
 *   client: CREATE <context> <cache_route> <j> <m> <alpha> <p> <temp> <top_p> <top_k> <max_tokens> <think>\n
 *   server: OK <slot id>\n   |   ERR <message>\n
 *   client: RUN <prompt on one line>\n
 *   server: <answer stream> \0 <performance / error text> \0
 *
 * A client's slot lives exactly as long as its connection.
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