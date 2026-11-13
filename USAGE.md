# Roaster usage guide

How to build, run and tune Roaster with Docker. For what Roaster is and how it performs, see the [README](README.md).

Roaster has three commands:

| Command | What it does |
|---|---|
| `serve` | Loads the model and waits for clients on a Unix socket. |
| `chat` | Terminal chat. Connects to `serve`. |
| `api` | OpenAI-compatible HTTP API. Connects to `serve`. |

`chat` and `api` are both thin clients of `serve`. Run `serve` first, then any mix of chats and API servers. They all share the same slots.

**Long conversations.** When a context fills up, Roaster does not stop. It slides the attention window, keeps your system prompt pinned, and parks the old part on your SSD. If you later enlarge the window, that part comes back. See [Context sliding](#context-sliding-and-the-ssd-spill). Every setting of a running session (window size, routing, sampling, sliding) can be changed on the fly. See [Changing settings on the fly](#changing-settings-on-the-fly). A whole context can also be stored in a directory and put back later, see [Saving and restoring a context](#saving-and-restoring-a-context).

---

## 1. Install

Choose one of the two installation methods below.

### Option A: Use the prebuilt Docker image (recommended)

Pull the Roaster v1.0 image from GitHub Container Registry:

```bash
docker pull ghcr.io/kkb007x/roaster:1.0
export ROASTER_IMAGE=ghcr.io/kkb007x/roaster:1.0
```

This does not require cloning the repository or building the source.

### Option B: Build from source

If you want to build Roaster yourself:

```bash
git clone https://github.com/KKB007X/Roaster.git
cd Roaster
docker build -t roaster:1.0 .
export ROASTER_IMAGE=roaster:1.0
```

The source list must include `kvspill.c` and `kvwindow.c` (the sliding window and the SSD store).

The remaining commands in this guide use `"$ROASTER_IMAGE"`, so they work with either installation method.

## 2. Find your GPU

Roaster needs to be given your GPU's render node, a file like `/dev/dri/renderD128`.

```bash
ls -l /dev/dri/by-path/
```

Each line links a PCI address to a `renderD…` node. To see which address is which GPU:

```bash
lspci | grep -Ei "vga|display|3d"
```

Match the PCI address (for example `0000:c5:00.0`) to its `…-render` entry. On a machine with a single GPU it is almost always `/dev/dri/renderD128`. With two GPUs (for example iGPU plus dedicated), the second is usually `renderD129`.

You can also check with `vulkaninfo --summary` on the host to confirm the GPU supports Vulkan.

Next, find the group that is allowed to use it:

```bash
ls -l /dev/dri/renderD128      # the group is shown, usually "render" or "video"
getent group render            # prints the group id
```

Notes:
- The container only sees the device you pass in, so passing just one node chooses that GPU.
- Roaster was developed on an AMD integrated GPU (Mesa RADV). Other Vulkan GPUs may work, as long as the image has a Vulkan driver for your GPU.
- With no GPU, drop the `--device` and `--group-add` lines. It still runs on CPU, but much slower.
- Only the `serve` container needs the GPU. `chat` and `api` do not.

## 3. Set your paths

> **Model:** use a Qwen3.6-35B-A3B GGUF in **Q4_K_M**. The benchmarks in the [README](README.md) were run with `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`. For results comparable to ours, use that exact file. Other quantizations and fine-tunes may load, but speed and memory use will differ.

Set these once in your terminal. Use your own values:

```bash
MODEL_DIR=/path/to/folder/with/the/gguf      # the folder, not the file
MODEL_FILE=your-model-file.gguf               # the file name inside that folder
GPU=/dev/dri/renderD128                        # from step 2
GPU_GROUP=$(stat -c %g "$GPU")                 # group id of that device
SOCK_DIR=/tmp/roaster                          # any folder; shared by server, chat and api
STATE_DIR=/path/on/your/ssd/roaster-state      # saved contexts and slid-out context; must be a real disk
mkdir -p "$SOCK_DIR" "$STATE_DIR"
```

> **Put `STATE_DIR` on an SSD, not on tmpfs.** Context that slides out of the window is written there. On many Linux systems `/tmp` is tmpfs, which is RAM. `serve` prints a warning at start-up if the state directory is on tmpfs.

## 4. Start the server

Leave this running in its own terminal:

```bash
docker run --rm -it --name roaster \
  --device="$GPU" \
  --group-add "$GPU_GROUP" \
  --memory=10g --memory-swap=10g \
  --ulimit memlock=-1:-1 \
  -e OMP_NUM_THREADS=4 -e OMP_DYNAMIC=FALSE -e OMP_WAIT_POLICY=passive \
  -v "$MODEL_DIR":/models:ro \
  -v "$SOCK_DIR":/run/roaster \
  -v "$STATE_DIR":/state \
  "$ROASTER_IMAGE" \
  serve --model "/models/$MODEL_FILE" \
        --expert-cache 64 --max-slots 4 --threads 4 \
        --socket /run/roaster/roaster.sock \
        --state-dir /state
```

Wait for `Listening on /run/roaster/roaster.sock`.

Adjust to your machine:
- `--memory`: your RAM budget for the container, see [Memory](#memory).
- `OMP_NUM_THREADS` and `--threads`: number of CPU threads. Keep them equal.
- `--expert-cache`: bigger is faster but uses more RAM.
- `--max-slots`: how many chats and/or API sessions can exist at the same time.
- `--state-dir`: where saved contexts and slid-out context live. Without the `-v` mount it is the container's own disk, which also works but is lost with the container.

## 5. Chat

In another terminal. You can open several, up to `--max-slots`:

```bash
docker run --rm -it -v "$SOCK_DIR":/run/roaster "$ROASTER_IMAGE" \
  chat --socket /run/roaster/roaster.sock \
       --cache-route 1 --temp 0.3 --max-tokens 512 --context 8192 --think 0 \
       --system "You are a concise assistant."
```

Type a prompt, press Enter, and the reply streams in. Type `exit` or press `Ctrl+D` to leave.

Good to know:
- **Each prompt starts a fresh conversation.** There is no history between prompts. Use the [API](#6-openai-compatible-api) for multi-turn conversations.
- `--system TEXT` or `--system-file PATH` sets a system prompt. It is wrapped in a system turn and pinned, and it is sent with every prompt because every prompt is a fresh conversation. Limit: 16000 characters.
- A prompt is one line, up to 16000 characters.
- Colours: thinking is dim italic, `<tool_call>` is yellow, other special tokens are magenta.
- Speed stats appear on stderr after each reply, including how many tokens are in the window, how many slid out to SSD, and how many are pinned.
- Several chats at once take turns one token each (round robin). Each one gets roughly 1/N of the speed.
- Scripting works without `-t`, with no prompt and no colour:

  ```bash
  echo "Explain mmap in two sentences" | docker run --rm -i -v "$SOCK_DIR":/run/roaster \
    "$ROASTER_IMAGE" chat --socket /run/roaster/roaster.sock --think 0 2>/dev/null
  ```

## 6. OpenAI-compatible API

`api` is an HTTP front-end that speaks the OpenAI chat-completions format, so existing OpenAI SDKs and tools can talk to Roaster.

```
your app (OpenAI SDK / curl) ──HTTP──▶ roaster api ──unix socket──▶ roaster serve
```

Start `serve` first (step 4), then in another terminal:

```bash
docker run --rm -it --name roaster-api \
  -p 127.0.0.1:8080:8080 \
  -v "$SOCK_DIR":/run/roaster \
  "$ROASTER_IMAGE" \
  api --socket /run/roaster/roaster.sock \
      --host 0.0.0.0 --port 8080 \
      --context 8192 --max-tokens 1024 --cache-route 1
```

Wait for `OpenAI-compatible API on http://0.0.0.0:8080/v1`.

Notes on the command:
- `--host 0.0.0.0` is the address **inside** the container. Who can reach it from outside is decided by `-p`. `127.0.0.1:8080:8080` allows only your own machine.
- To allow other machines, use `-p 8080:8080` **and** set an API key (below). There is no TLS, so put a reverse proxy in front for anything beyond a trusted network.
- The `api` container needs no GPU, no `--memory` and no `--ulimit`. Those belong to `serve`.
- The slot options of `api` (`--context`, `--cache-route`, `--slide`, …) are the **defaults for new sessions**. Any session can be changed later, see [Changing settings on the fly](#changing-settings-on-the-fly).

To require an API key, add `-e ROASTER_API_KEY=your-secret` to `docker run` (or pass `--api-key your-secret` to `api`). Clients must then send `Authorization: Bearer your-secret`. `/health` is always open.

### Try it

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"Answer in one sentence."},
       {"role":"user","content":"Hello!"}],"stream":true}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="x")  # any key if none is set
for ev in client.chat.completions.create(
        model="roaster", stream=True,
        messages=[{"role": "user", "content": "Write a haiku about coffee"}]):
    print(ev.choices[0].delta.content or "", end="")
```

### Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat completion, streaming (`"stream": true`) or not. |
| `GET /v1/models` | Lists one model, named by `--model-name`. |
| `GET /health` | Returns `{"status":"ok"}`. Needs no key. It only checks the API process, not the `serve` process. |
| `/v1/sessions/…` | Control of the sessions: create, inspect, change settings, stop, reset, unload, load, delete. See [Sessions](#sessions-the-context-is-kept-between-requests). |

### Request fields

| Field | Description |
|---|---|
| `messages` | Required. Non-empty array of `{role, content}`. `content` may be a string or an array of `text` parts. Images are ignored. `system` / `developer` messages become the pinned system prompt, see [System prompt](#system-prompt). |
| `stream` | `true` for Server-Sent Events. Default `false`. |
| `stream_options.include_usage` | `true` adds a final usage chunk when streaming. |
| `temperature`, `top_p` | Sampling, **for this request only**. Same ranges as the chat options. |
| `max_tokens`, `max_completion_tokens` | Max tokens in the reply, thinking included. This request only. |
| `top_k` | Extension: sample only from the K likeliest tokens. This request only. |
| `think` | Extension: `true` or `false`. Turns reasoning on or off for this request. |
| `chat_template_kwargs.enable_thinking` | Same as `think`, for clients that already send it. |
| `session` | Extension: name of the context to use (also accepted as the `X-Session-Id` header). Default `default`. |
| `reset` | Extension: `true` empties the session's context before this request. |
| `context_length`, `cache_route`, `route_j`, `route_m`, `route_alpha`, `route_p`, `slide`, `keep` | Extension: **change the session for good**, see [Changing settings on the fly](#changing-settings-on-the-fly). |
| `model` | Echoed back in the response. It does not select a model. |

Anything not listed is ignored.

**Per-request versus permanent.** Sampling fields (`temperature`, `top_p`, `top_k`, `max_tokens`, `think`) apply to that one request and are put back afterwards. Without them, a request uses the session's current settings. Slot-level fields (`context_length`, routing, `slide`, `keep`) are applied to the session permanently. For permanent sampling changes, use `PATCH` on the session.

### Reasoning

When thinking is on, the reasoning text comes back separately from the answer, in `reasoning_content` (the DeepSeek / vLLM convention). In streaming mode it arrives in `delta.reasoning_content`, followed by `delta.content` for the answer.

Thinking is **off by default** in `api` mode, because the model can spend the whole token budget thinking. Turn it on for every request with `--think 1`, or per request with `"think": true`. If it is on, give it room with a larger `--max-tokens`.

`finish_reason` is `"length"` when the reply was cut off by `max_tokens`, otherwise `"stop"`.

### System prompt

System (and `developer`) messages are collected and sent as a real system turn (`<|im_start|>system … <|im_end|>`) at the start of the conversation. The system prompt is **pinned**: it always stays in the attention window, even when the window slides.

- It is read when the session's context is **empty** (a new session, or after `reset`). System messages sent on later turns are ignored, because the conversation has already started.
- A session remembers its system prompt, also across resets. A request without a system message reuses it.
- To set or replace it without sending a chat request: `PATCH /v1/sessions/{id}` with `{"system": "…"}`. It applies when the conversation next starts, so `reset` the session to apply it to an existing context. `{"system": ""}` removes it.
- Limit: 16000 bytes. The pinned part can take up to 7/8 of the window. If the system prompt is longer than that, the rest of it can slide out.
- `GET /v1/sessions/{id}` shows `pinned_tokens`.

### Sessions: the context is kept between requests

**The context is reused by default.** Every request belongs to a *session*: the one named by `"session"` (or the `X-Session-Id` header), otherwise `default`. A session that does not exist yet is created on first use. Its context (the KV cache and recurrent state) stays on the server, so a follow-up request only processes the new message instead of re-reading the whole conversation. It stays until you reset, unload or delete it.

- **Clients that resend the full history work unchanged.** If the session already holds context, only the messages after the last assistant message are added. If the context is empty, the whole message list is used.
- **One request at a time per session.** A second request on a busy session gets HTTP 409. Use different sessions for parallel conversations.
- **A session holds one slot** (`--max-slots`) until it is deleted, even while unloaded. Raise `--max-slots` if you want many sessions.
- **The reasoning text stays in the context** too. Long thinking fills the window faster.
- **When the window is full** (default `slide` on) it slides instead of failing. See [Context sliding](#context-sliding-and-the-ssd-spill).
- **Sessions live as long as `roaster api` runs.** If you restart `api` they are gone. If you restart `serve`, the next request on each session returns 502 once and then starts a fresh session.

| Call | What it does |
|---|---|
| `GET /v1/sessions` | Lists sessions with their state. A session that is busy shows only `"busy": true`. |
| `POST /v1/sessions` | Creates a session. Body, all optional: `{"id": "work", "context_length": 16384, "cache_route": true, "keep": 128, …}`. Accepts every setting listed below. |
| `GET /v1/sessions/{id}` | State and settings, see below. |
| `PATCH /v1/sessions/{id}` | **Changes settings** of an idle session. Same as `POST /v1/sessions/{id}/config`. |
| `POST /v1/sessions/{id}/config` | Same as `PATCH`. |
| `POST /v1/sessions/{id}/stop` | **Ends the running generation properly** (see below). |
| `POST /v1/sessions/{id}/reset` | Empties the context. Also deletes the saved state and the SSD spill. |
| `POST /v1/sessions/{id}/save` | **Stores the whole context in a directory.** Body: `{"path": "/dir"}`. See [Saving and restoring a context](#saving-and-restoring-a-context). |
| `POST /v1/sessions/{id}/restore` | **Replaces the context with one stored in a directory.** Body: `{"path": "/dir"}`. |
| `POST /v1/sessions/{id}/unload` | **Frees the context's RAM** (see below). `{"discard": true}` throws the context away instead of saving it. |
| `POST /v1/sessions/{id}/load` | Brings the context back into RAM. An empty session gets an empty context. |
| `DELETE /v1/sessions/{id}` | Destroys the session, its slot, any saved file and the SSD spill. Returns 409 while it is busy. |

**What `GET /v1/sessions/{id}` returns**

| Field | Meaning |
|---|---|
| `loaded`, `broken`, `busy` | State of the session. |
| `position` | Tokens in the whole conversation. It keeps counting when the window slides. |
| `context_length` | Size of the attention window, in tokens. |
| `window_tokens` | Tokens currently in the window (RAM). |
| `spilled_tokens` | Tokens that slid out and are stored on SSD. |
| `pinned_tokens` | Tokens of the pinned system prompt. |
| `ram_bytes`, `state_bytes`, `spill_bytes` | RAM used by the context, size of the saved (unloaded) state file, size of the SSD spill file. |
| `config` | All current settings: `context_length`, `cache_route`, `route_j`, `route_m`, `route_alpha`, `route_p`, `temperature`, `top_p`, `top_k`, `max_tokens`, `think`, `slide`, `keep`. |

```bash
curl -X POST http://127.0.0.1:8080/v1/sessions/default/stop
curl -X POST http://127.0.0.1:8080/v1/sessions/default/unload
curl -X POST http://127.0.0.1:8080/v1/sessions/default/load
curl http://127.0.0.1:8080/v1/sessions/default
```

**Stop.** Generation ends after the current token. The partial answer stays in the context, the response has `"finish_reason": "stop"` and `"stopped_by_request": true`, and the next request carries on from that point. The same happens automatically when the HTTP client disconnects. If the stop arrives while the prompt is still being read, the prompt is rolled back and the context is exactly as it was before the request. This includes any sliding that happened meanwhile: tokens that slid out during that prompt are brought back from SSD.

**Unload and load.** Unload frees the memory of the session's context: the KV window, the recurrent state and the work buffers (about 170 MB at `--context 4096`). The model weights and the expert cache are not touched. The context is first written to the server's state directory (`--state-dir`, about 40 KB per window token plus 65 MB), and `load` restores it exactly. If the save fails, nothing is unloaded. While a session is unloaded, a chat request to it returns 409, and its context window cannot be changed. Saved state files are removed on load, reset, delete and clean shutdown. The SSD spill is not part of the state file: it stays where it is during unload and is only deleted on reset, discard, delete and shutdown.

### Changing settings on the fly

Every setting of a session can be changed after the session exists, with one call. Send any subset:

```bash
curl -X PATCH http://127.0.0.1:8080/v1/sessions/default \
  -H 'Content-Type: application/json' \
  -d '{"context_length": 16384, "cache_route": true, "route_j": 3, "temperature": 0.5}'
```

| Setting | Meaning |
|---|---|
| `context_length` | Attention window in tokens. Larger: slid-out context returns from SSD. Smaller: the oldest tokens go out to SSD. |
| `cache_route`, `route_j`, `route_m`, `route_alpha`, `route_p` | Cache-aware routing, see [Cache-aware routing](#slot-options-chat-and-api). `cache_route` takes `true`/`false`. |
| `slide`, `keep` | Context sliding on/off, and how many tokens at the start are never slid out. |
| `temperature`, `top_p`, `top_k`, `max_tokens`, `think` | Sampling and limits, as the new default of the session. |
| `system` | System prompt for the next conversation. See [System prompt](#system-prompt). |

The response is the session's new state (same as `GET`). Rules:

- **The session must be idle.** While a request is running you get HTTP 409. Wait or call `/stop` first.
- **Changing the window of an unloaded session is not possible** (HTTP 409). `load` it first. Other settings can be changed while unloaded.
- Invalid values get HTTP 400 with the reason, and nothing is changed.
- **Resizing needs memory.** For a moment the old and the new window exist side by side. If the resize cannot be done (out of memory, SSD error) you get HTTP 500 and the session keeps working, in the state it was in.
- Slot-level fields can also be put straight in a chat request body (`context_length`, routing, `slide`, `keep`). They change the session permanently. Sampling fields in a chat body only apply to that request.
- `roaster api --context …` and the other slot options only set the defaults for **new** sessions.

### Saving and restoring a context

`save` writes the complete context of a session into a directory. `restore` replaces the session's current context with one that was saved before, in the same session, another session, or after a restart of `serve`.

```bash
# store the context of the session "work" in a directory
curl -X POST http://127.0.0.1:8080/v1/sessions/work/save \
  -H 'Content-Type: application/json' -d '{"path": "/state/snapshots/project-a"}'

# later: replace the context of any session with it
curl -X POST http://127.0.0.1:8080/v1/sessions/work/restore \
  -H 'Content-Type: application/json' -d '{"path": "/state/snapshots/project-a"}'
```

`save` answers with the session state plus `saved_to` and `saved_bytes`. `restore` answers with the session state after the restore. The next chat request carries on exactly where the saved conversation stopped.

**What is stored.** The attention window, the recurrent state, the position, the part that slid out to SSD (`spill.kv` and `spill.idx`), the system prompt and the number of pinned tokens, plus a small metadata file. The directory is created if it does not exist, and an older snapshot in it is replaced. A save that is interrupted is never taken for a valid one, because the metadata file is written last.

| File | Content |
|---|---|
| `roaster-context.meta` | Version, model fingerprint, position, window, spilled and pinned token counts, system prompt. |
| `state.bin` | Window KV and recurrent state (about 40 KB per window token plus 65 MB). |
| `spill.kv`, `spill.idx` | The slid-out part (about 40 KB per token). Only present if something slid out. |

**Rules**
- **The path is a directory on the `serve` machine** (inside its container, when you use Docker). Mount a volume for it, for example `-v "$SNAP_DIR":/snapshots`, and use paths like `/snapshots/project-a`. With `serve --save-root DIR` every path is taken relative to `DIR` and `..` is refused. Use that whenever the API is reachable by anyone you do not fully trust, because `save` writes files wherever the server user may write.
- **The session must be idle.** A running request gives HTTP 409. `save` works on loaded and unloaded sessions. `restore` needs a loaded session (HTTP 409 otherwise).
- **Settings are kept.** Sampling, routing, `slide` and `keep` stay as they are. The system prompt comes with the saved context.
- **The window never gets smaller.** After a restore, the window is at least as large as the one that was saved. If the session's window is bigger, it stays, and everything that slid out is brought back from the snapshot while it fits.
- **Same model only.** A snapshot only restores with the model (and Roaster version) that made it. Otherwise: HTTP 400 `invalid saved context`.
- **The snapshot is a copy.** Resetting or deleting the session does not touch it. Delete the directory yourself when you no longer need it.
- **A failed restore leaves an empty context.** The snapshot is checked before anything is replaced, so this is rare (disk error, out of memory). The response says the previous context was discarded.

### Context sliding and the SSD spill

When the attention window is full, Roaster slides it instead of failing with "exceeds slot context".

- **What slides.** Only the key/value cache of the 10 full-attention layers. The recurrent layers carry a compressed memory of the whole conversation and need nothing.
- **What stays.** The first `keep` tokens (default 64) and the pinned system prompt are never slid out. If both are set, the larger one counts. `keep` is limited to half of the window.
- **How it slides.** In blocks of about a quarter of the window, not one token at a time. The window therefore sits between about 75% and 100% full.
- **Where the old part goes.** To a file in the state directory, `roaster-<pid>-slot<N>.kv`: about 40 KB per token, written to SSD, not kept in RAM. It stays on disk until the session is reset, discarded or deleted, or `serve` shuts down. It survives unload and load.
- **Getting it back.** Raise `context_length` (see above) and the slid-out tokens move back into the window, newest first. If the larger window is still not big enough for everything, the window is filled but about a quarter of it is left free, so the next token does not immediately push them out again.
- **Shrinking.** Lowering `context_length` below what is in the window pushes the oldest tokens out to SSD. Nothing is lost.
- **Quality.** The 10 attention layers cannot see slid-out tokens until they are brought back. The recurrent layers still carry a compressed memory of them, but long-range exact recall (quoting something from the slid-out part) is not guaranteed.
- **Switching it off.** `--slide 0` (or `"slide": false`) restores the old behaviour: when the window is full, the request fails with HTTP 400 "exceeds slot context", and prompt tokens plus `max_tokens` must fit. Shrinking the window still moves tokens to SSD even then.
- **Disk space.** Roughly 40 KB for every token that has slid out. A conversation of 1 million tokens in a 8192 window holds about 40 GB on SSD. Check that the state directory has room.

### Limits

- **Turns are appended as user turns.** Several new messages in one request are flattened into the user turn as plain text (`User: …`, `Assistant: …`). System messages are the exception: they become the pinned system prompt, and only when the context is empty.
- The whole prompt of one request is limited to 16000 bytes. Longer prompts get HTTP 400. The conversation as a whole is not limited, because of sliding.
- With `--slide 0`, prompt tokens plus `max_tokens` must fit in `--context`, or the request gets HTTP 400.
- Not supported: `n` above 1 (rejected), `stop` sequences, `tools` / function calling, `logprobs`, `seed`.
- The API was written to match the OpenAI format but has had little testing. Report anything an SDK does not accept.

### HTTP status codes

| Code | Meaning |
|---|---|
| 400 | Invalid JSON, invalid parameter or setting, empty or too long prompt or system prompt, context exceeded (only with `slide` off), an invalid `path`, or a snapshot that is damaged or from another model. |
| 401 | Missing or wrong API key. |
| 404 / 405 | Unknown endpoint, or wrong method. `restore`: no saved context in that directory. |
| 409 | The session is busy, is unloaded, or needs a reset. Also: settings cannot be changed while busy, and the window of an unloaded session cannot be changed. The message says which. |
| 413 | Request body larger than 4 MB. |
| 500 | The context window could not be resized (out of memory or SSD error), or `save` / `restore` hit a disk error. |
| 502 | `api` cannot reach `serve`. Is it running, and is the socket folder shared? |
| 503 | No free slot (every slot is held by a chat or a session). |

---

## Explaining the `docker run` flags

| Flag | What it does |
|---|---|
| `--device="$GPU"` | Gives the container your GPU. |
| `--group-add "$GPU_GROUP"` | Lets the container user open that GPU. |
| `--memory=10g --memory-swap=10g` | RAM cap with swap disabled. |
| `--ulimit memlock=-1:-1` | **Required** for `serve`. Roaster locks weights and caches in RAM. |
| `OMP_NUM_THREADS`, `OMP_DYNAMIC`, `OMP_WAIT_POLICY` | Fixed CPU thread count, and idle threads sleep instead of spinning. |
| `-v "$MODEL_DIR":/models:ro` | Your model folder, read-only. Put it on a fast SSD or NVMe. |
| `-v "$SOCK_DIR":/run/roaster` | The socket shared between the server and the chat / api containers. |
| `-v "$STATE_DIR":/state` | Saved contexts and slid-out context. Use a folder on an SSD, not tmpfs. |
| `-v "$SNAP_DIR":/snapshots` | Optional: where the API's `save` / `restore` calls keep their snapshots. |
| `-p 127.0.0.1:8080:8080` | Publishes the API port, reachable only from your machine. |

---

## Server options (`serve`)

| Option | Default | Description |
|---|---|---|
| `--model PATH` | *(required)* | GGUF path **inside the container** (for example `/models/file.gguf`). |
| `--max-slots N` | `8` | Maximum simultaneous chats and API sessions. Each uses one slot. |
| `--expert-cache N` | `16` | Experts kept in RAM per layer (1–256). More is faster but uses more RAM. |
| `--threads N` | `0` | CPU threads. `0` means all cores. |
| `--socket PATH` | `/tmp/roaster.sock` | Socket to listen on. |
| `--save-root DIR` | *(none)* | Confines the directories of the API's `save` / `restore` calls to `DIR` (paths become relative to it, `..` is refused). Without it, any path the server may write to is accepted. |
| `--state-dir DIR` | `/var/tmp/roaster-state` | Where unloaded contexts and slid-out context (the SSD spill) are stored. Use a real disk: a warning is printed if it is on tmpfs. Inside Docker, mount a volume and point `--state-dir` at it. |

Stop with `Ctrl+C`. Files for slots that are still open are deleted on a clean shutdown.

## Slot options (`chat` and `api`)

These are set per chat, or as defaults for every new API session. Each chat can differ. For `api`, sessions can be changed later (see [Changing settings on the fly](#changing-settings-on-the-fly)), and requests can override sampling, `--max-tokens` and `--think` in the JSON body.

**Basics**

| Option | Default | Range | Description |
|---|---|---|---|
| `--socket PATH` | `/tmp/roaster.sock` | | Must match the server. |
| `--context N` | `4096` | ≥ 1 | Size of the attention window in tokens. With sliding on (default) a conversation can be longer than this. With `--slide 0` prompt tokens plus `--max-tokens` must fit. |
| `--max-tokens N` | `2048` | ≥ 1 | Max tokens per reply, thinking included. |
| `--think 0\|1` | `1` for `chat`, `0` for `api` | `0`/`1` (or `off`/`on`) | `1` reasons in `<think>…</think>` first. `0` answers directly and is faster. |
| `--system TEXT` | *(none)* | | `chat` only. System prompt, pinned. |
| `--system-file PATH` | *(none)* | | `chat` only. Read the system prompt from a file (up to 16000 bytes). |
| `--color auto\|on\|off` | `auto` | | `chat` only. Output colours. `auto` is on for terminals, unless `NO_COLOR` is set. |

**Sliding**

| Option | Default | Range | Description |
|---|---|---|---|
| `--slide 0\|1` | `1` | `0`/`1` (or `off`/`on`) | When the window is full, slide it and move the old part to SSD. `0` fails instead. |
| `--keep N` | `64` | ≥ 0 | Tokens at the start of the conversation that are never slid out. Limited to half of the window. A pinned system prompt may need more and is handled separately. |

**Sampling**

| Option | Default | Range | Description |
|---|---|---|---|
| `--temp X` | `0.8` | ≥ 0 | Temperature. `0` is greedy and deterministic. |
| `--top-p X` | `0.95` | (0, 1] | Nucleus sampling. |
| `--top-k N` | `20` | 1–248320 | Sample only from the K likeliest tokens. |

**Cache-aware routing** (a speed/quality trade-off)

| Option | Default | Range | Description |
|---|---|---|---|
| `--cache-route 0\|1` | `0` | `0`/`1` | Prefer experts that are already in RAM. |
| `--route-j N` | `2` | 0–8 | Top experts that are never swapped out. |
| `--route-m N` | `12` | 8–256 | How far down the ranking to look for cached substitutes. |
| `--route-alpha X` | `1.0` | 0–1 | Weight multiplier for substituted experts. Lower trusts them less. |
| `--route-p X` | `0.0` | 0–1 | Limits the window by router probability mass. `0` is off. |

Normally each token uses the router's exact top-8 experts, and a cache miss costs SSD reads. With `--cache-route 1`, the first `--route-j` stay exact and the others prefer cached experts. It is faster, but may reduce quality a little. Start with `--cache-route 1` and defaults, and raise `--route-j` if the answers get worse.

## API-only options (`api`)

| Option | Default | Description |
|---|---|---|
| `--host ADDR` | `127.0.0.1` | IPv4 address to listen on. Use `0.0.0.0` inside Docker. |
| `--port N` | `8080` | HTTP port (1–65535). |
| `--api-key KEY` | *(none)* | Require `Authorization: Bearer KEY`. Can also be set with the `ROASTER_API_KEY` environment variable. |
| `--model-name NAME` | `roaster` | Name reported by `/v1/models` and used when a request has no `model`. |

Stop with `Ctrl+C`.

---

## Memory

Rough estimates (Q4_K):

- **Expert cache:** `--expert-cache × 40 layers × ~1.7 MB`. About 1.1 GB at 16 and about 4.5 GB at 64.
- **Each slot:** about 40 KB per window token, plus about 65 MB fixed. `--context 8192` is about 400 MB per slot. Only the window counts: slid-out tokens are on SSD.
- **Always resident:** the non-expert weights.

Chats and API sessions all use slots, so the worst case is `--max-slots` slots at once. An unloaded session uses no context RAM but still holds its slot, and its saved context takes disk space in the state directory. A request that arrives on a session with context also needs about 65 MB extra while its prompt is processed (the rollback snapshot). Growing a window needs the old and the new window at the same time for a moment. Set `--memory` above the total plus headroom. If the container gets killed, lower `--expert-cache`, `--max-slots` or `--context`.

**Disk (state directory):**

- Slid-out context: about 40 KB per token, until the session is reset, discarded or deleted.
- Unloaded context: about 40 KB per window token plus 65 MB, until the session is loaded again.
- Saved snapshots (`save`): about 40 KB per window token plus about 40 KB per slid-out token plus 65 MB, until you delete the directory.
- After a crash of `serve`, leftover `roaster-<pid>-slot<N>.state` and `.kv` files can be deleted by hand.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to …: is the server running?` | Start the server first. Both containers must mount the same socket folder and use the same `--socket` path. |
| API returns 502 | Same cause: `serve` is not running, or `api` has a different socket folder or path. |
| API returns 503, or `could not create slot` | All slots are busy. Raise `--max-slots` on `serve`, or wait for a chat or request to finish. |
| API returns 409 "busy" | That session is already answering. Wait, call `/stop`, or use another session. The same message appears when you try to change its settings while it runs. |
| API returns 409 "unloaded" | `POST /v1/sessions/{id}/load` first. |
| 409 "the context is unloaded; load it before changing the context window" | Load the session, change `context_length`, and unload again if you want. |
| `save` / `restore` returns 400 "invalid path" | The path is empty, contains `..`, or is too long. |
| `restore` returns 404 "no saved context" | That directory has no `roaster-context.meta`. Remember the path is inside the `serve` container: mount the folder there. |
| `restore` returns 400 "invalid saved context" | The snapshot is damaged, incomplete, or from another model or Roaster version. |
| `save` / `restore` returns 409 | The session is busy (wait or `/stop`), `restore` needs a loaded session, or `save` found an empty context. |
| API returns 409 "reset" | A failure left the context inconsistent. `POST /v1/sessions/{id}/reset`. |
| API returns 400 "exceeds slot context" | Only happens with sliding off. Switch it on (`"slide": true`), reset the session, or use a larger `context_length`. |
| API returns 400 "system prompt too long" | The system prompt is over 16000 bytes. Shorten it. |
| 500 "could not resize the context window" | Not enough RAM for the new window, an SSD error, or the pinned part is as large as the new window. Free memory or pick a larger value. The session is unchanged. |
| Warning "state dir … is on tmpfs" | Slid-out context would be kept in RAM. Start `serve` with `--state-dir` on a real disk. |
| `could not save the context` on unload | The state directory is full or not writable. Check `--state-dir`. |
| Slid-out context fills the disk | It takes about 40 KB per token. Reset or delete sessions you no longer need, or use `"slide": false`. |
| Old unload files not accepted after an update | The state file format changed. `reset` the session (or `unload` with `{"discard": true}`) and start the conversation again. |
| The model forgets things from early in a long conversation | They slid out of the attention window. Raise `context_length` (they come back from SSD), raise `keep`, or put what matters in the system prompt, which is pinned. |
| API returns 401 | The `Authorization: Bearer …` header does not match `--api-key` / `ROASTER_API_KEY`. |
| `curl` to the API gets "connection refused" | Inside Docker, run `api` with `--host 0.0.0.0` and publish the port with `-p`. With `--host 127.0.0.1` it is unreachable from outside the container. |
| API reply is empty, `finish_reason` is `length` | Thinking used the whole budget. Raise `max_tokens`, or send `"think": false`. |
| `mlock … failed` | Use `--ulimit memlock=-1:-1` and a large enough `--memory`. |
| `Vulkan: initialization failed; using CPU kernels` | The GPU isn't visible. Recheck `$GPU` and the group (step 2). It still works, but slowly. |
| `Permission denied` on the device | Use the right group: `stat -c %g "$GPU"`. |
| `Maximum number of slots reached` | Raise `--max-slots` or close a chat. |
| `Prompt plus requested generation exceeds slot context` | Only with `--slide 0`. Raise `--context`, lower `--max-tokens`, or enable sliding. |
| `Stopped at max-tokens while still thinking` | Raise `--max-tokens`, or use `--think 0`. |
| `Failed to load model weights` | Check the path inside the container, and that the GGUF is v3 of a supported model and quantization. |
