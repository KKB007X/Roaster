# Roaster usage guide

How to build, run and tune Roaster with Docker. For what Roaster is and how it performs, see the [README](README.md).

---

## 1. Build

```bash
git clone <repo-url> roaster
cd roaster
docker build -t roaster:dev .
```

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

## 3. Set your paths

Set these once in your terminal. Use your own values:

```bash
MODEL_DIR=/path/to/folder/with/the/gguf      # the folder, not the file
MODEL_FILE=your-model-file.gguf               # the file name inside that folder
GPU=/dev/dri/renderD128                        # from step 2
GPU_GROUP=$(stat -c %g "$GPU")                 # group id of that device
SOCK_DIR=/tmp/roaster                          # any folder; shared by server and chat
mkdir -p "$SOCK_DIR"
```

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
  roaster:dev \
  serve --model "/models/$MODEL_FILE" \
        --expert-cache 64 --max-slots 4 --threads 4 \
        --socket /run/roaster/roaster.sock
```

Wait for `Listening on /run/roaster/roaster.sock`.

Adjust to your machine:
- `--memory`: your RAM budget for the container, see [Memory](#memory).
- `OMP_NUM_THREADS` and `--threads`: number of CPU threads. Keep them equal.
- `--expert-cache`: bigger is faster but uses more RAM.

## 5. Chat

In another terminal. You can open several, up to `--max-slots`:

```bash
docker run --rm -it -v "$SOCK_DIR":/run/roaster roaster:dev \
  chat --socket /run/roaster/roaster.sock \
       --cache-route 1 --temp 0.3 --max-tokens 512 --context 8192 --think 0
```

Type a prompt, press Enter, and the reply streams in. Type `exit` or press `Ctrl+D` to leave.

Good to know:
- **Each prompt starts a fresh conversation.** There is no history between prompts.
- A prompt is one line, up to 16000 characters.
- Colours: thinking is dim italic, `<tool_call>` is yellow, other special tokens are magenta.
- Speed stats appear on stderr after each reply.
- Several chats at once take turns one token each (round robin). Each one gets roughly 1/N of the speed.
- Scripting works without `-t`, with no prompt and no colour:

  ```bash
  echo "Explain mmap in two sentences" | docker run --rm -i -v "$SOCK_DIR":/run/roaster \
    roaster:dev chat --socket /run/roaster/roaster.sock --think 0 2>/dev/null
  ```

---

## Explaining the `docker run` flags

| Flag | What it does |
|---|---|
| `--device="$GPU"` | Gives the container your GPU. |
| `--group-add "$GPU_GROUP"` | Lets the container user open that GPU. |
| `--memory=10g --memory-swap=10g` | RAM cap with swap disabled. |
| `--ulimit memlock=-1:-1` | **Required.** Roaster locks weights and caches in RAM. |
| `OMP_NUM_THREADS`, `OMP_DYNAMIC`, `OMP_WAIT_POLICY` | Fixed CPU thread count, and idle threads sleep instead of spinning. |
| `-v "$MODEL_DIR":/models:ro` | Your model folder, read-only. Put it on a fast SSD or NVMe. |
| `-v "$SOCK_DIR":/run/roaster` | The socket shared between the server and chat containers. |

---

## Server options (`serve`)

| Option | Default | Description |
|---|---|---|
| `--model PATH` | *(required)* | GGUF path **inside the container** (for example `/models/file.gguf`). |
| `--max-slots N` | `8` | Maximum simultaneous chats. Each chat uses one slot. |
| `--expert-cache N` | `16` | Experts kept in RAM per layer (1–256). More is faster but uses more RAM. |
| `--threads N` | `0` | CPU threads. `0` means all cores. |
| `--socket PATH` | `/tmp/roaster.sock` | Socket to listen on. |

Stop with `Ctrl+C`.

## Chat options (`chat`)

These are set per chat, so each chat can differ.

**Basics**

| Option | Default | Range | Description |
|---|---|---|---|
| `--socket PATH` | `/tmp/roaster.sock` | | Must match the server. |
| `--context N` | `4096` | ≥ 1 | Context length. Prompt tokens plus `--max-tokens` must fit. |
| `--max-tokens N` | `2048` | ≥ 1 | Max tokens per reply, thinking included. |
| `--think 0\|1` | `1` | `0`/`1` (or `off`/`on`) | `1` reasons in `<think>…</think>` first. `0` answers directly and is faster. |
| `--color auto\|on\|off` | `auto` | | Output colours. `auto` is on for terminals, unless `NO_COLOR` is set. |

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

---

## Memory

Rough estimates (Q4_K):

- **Expert cache:** `--expert-cache × 40 layers × ~1.7 MB`. About 1.1 GB at 16 and about 4.5 GB at 64.
- **Each slot:** about 40 KB per context token, plus about 65 MB fixed. `--context 8192` is about 400 MB per slot.
- **Always resident:** the non-expert weights.

Set `--memory` above the total plus headroom. If the container gets killed, lower `--expert-cache`, `--max-slots` or `--context`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to …: is the server running?` | Start the server first. Both containers must mount the same socket folder and use the same `--socket` path. |
| `mlock … failed` | Use `--ulimit memlock=-1:-1` and a large enough `--memory`. |
| `Vulkan: initialization failed; using CPU kernels` | The GPU isn't visible. Recheck `$GPU` and the group (step 2). It still works, but slowly. |
| `Permission denied` on the device | Use the right group: `stat -c %g "$GPU"`. |
| `Maximum number of slots reached` | Raise `--max-slots` or close a chat. |
| `Prompt plus requested generation exceeds slot context` | Raise `--context` or lower `--max-tokens`. |
| `Stopped at max-tokens while still thinking` | Raise `--max-tokens`, or use `--think 0`. |
| `Failed to load model weights` | Check the path inside the container, and that the GGUF is v3 of a supported model and quantization. |
