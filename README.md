# Roaster

Run the **Qwen3.6-35B-A3B** model (GGUF, Q4_K / Q6_K) locally on a GPU through Vulkan, with a CPU fallback. Only the experts a token actually needs are streamed from your SSD, so the model runs in far less RAM than the file size.

It has two parts:

- **server** (`serve`): loads the model once and waits for chats.
- **chat** (`chat`): connects to the server and streams the reply token by token. Thinking text, tool calls and other special tokens get different colours.

Everything runs in Docker. You bring your own `.gguf` file.

> Only the Qwen3.5/3.6 35B-A3B architecture is supported (sizes are compile-time constants). Supported quantizations: F32, F16, Q4_K, Q6_K.

**Get started:** build the Docker image and run your first chat with the **[usage guide](USAGE.md)**.

---

## Benchmarks: Roaster vs Colibri

Same machine (AMD Radeon 780M iGPU), same 10 GB RAM cap, Qwen3.6-35B-A3B at 4-bit. The comparison uses total throughput: input and output tokens divided by total response time.

```
Total throughput, tokens/s (higher is better)

Colibri  cap 64             █████████░░░░░░░░░░░   4.39*
Colibri  cap 96             ██████████░░░░░░░░░░   5.04*
Colibri  cap 96 + route     █████████████░░░░░░░   6.67*
Roaster  cap 64             ████████████████████  10.22
```

| | Colibri | Roaster | Roaster vs Colibri |
|---|---:|---:|---:|
| Throughput, cap 64 / 10 GB | 4.39 tok/s* | **10.22 tok/s** | **~2.3× faster** |
| Throughput vs Colibri's best (cap 96 + cache-route, 8 threads) | 6.67 tok/s* | **10.22 tok/s** (no cache-route, 4 threads) | **~1.5× faster** |
| Throughput, cap 8 (smallest cache) | 1.75 tok/s* | **4.07 tok/s** | **~2.3× faster** |
| RAM, cap 8 | 4.2 GB | **2.6 GB** | **~38% less** |
| Concurrent chats | one | **multiple slots, round robin** | |

\* Colibri's throughput was measured as *(input tokens + output tokens) ÷ total response time*, using the same method as Roaster. See [how fair is this comparison](#how-fair-is-this-comparison) for the remaining comparison limitations.

## Benchmark details

### Colibri results (as measured)

Qwen3.6-35B-A3B, `cap` is experts cached per layer. Throughput is (input tokens + output tokens) ÷ total response time.

| Weights | Config | Speed | RAM |
|---|---|---:|---:|
| first build | cap 8 | 1.75 | 4.2 GB |
| | cap 16 | 2.35 | 6.2 GB |
| | cap 24 | 2.80 | 7.2 GB |
| | cap 32 | 3.13 | 7.0 GB |
| | cap 64 (max) | 4.39 | 10 GB |
| | cap 64 + cache-route | 5.01 | |
| | cap 64 + no-think | 4.41 | |
| | cap 64 + both | 5.27 | |
| i4 gs64 | cap 64 + cache-route | 6.31 | 8.7 GB |
| | cap 80 + cache-route | 6.39 | 9.3 GB |
| | cap 96 + cache-route, 8 threads | 6.51 | 10 GB |
| | cap 96 + cache-route, 6 threads | 6.58 | 10 GB |
| | cap 96 + cache-route, 4 threads | 6.49 | 10 GB |
| | cap 96, no route | 5.04 | 10 GB |
| | cap 96 + cache-route, 8 threads, top-p 0.85 | 6.67 | 10 GB |

Colibri's cache-route gave roughly +14% (cap 64, first build) to +30% (cap 96, i4 gs64). Colibri reports the answers stay semantically almost identical.

### Roaster results

Q4_K_M GGUF, 4 threads, 1 slot, no cache-route, same 10 GB cap. Prefill and decode are timed separately.

| Build | Expert cache | Prompt | Prefill | Decode | RAM |
|---|---:|---|---:|---:|---:|
| early (Radeon 780M) | cap 8 | 32 tok | 4.23 tok/s | 4.07 tok/s | 2.6 GB |
| debug/profiling build | cap 64 | 43 tok | 8.12 tok/s | 9.01 tok/s | |
| current build | cap 64 | 16 tok | 9.73 tok/s | **10.22 tok/s** | |

### Several chats at once

Two chats, same settings, 512 tokens each. The chats take turns one token each, so the work is shared and total speed doesn't go up.

| Setup | Decode speed per chat | Total decode speed | Time queued |
|---|---:|---:|---:|
| 1 chat | 10.22 tok/s | 10.22 tok/s | 0 s |
| 2 chats | 4.67 and 4.68 tok/s | 9.35 tok/s | ~56 s each |

Round-robin sharing costs about 9% of total speed, probably because the chats use different experts and compete for the expert cache. For more chats, raise `--expert-cache` (see the [usage guide](USAGE.md)).

### How fair is this comparison

- **Throughput methodology.** Both engines were measured as (input tokens + output tokens) ÷ total response time. The comparison still uses different model weights and fine-tunes, and the results are single runs on one machine, so the ratios should be treated as indicative rather than a controlled comparison.
- **Different weights.** Colibri used its own i4 gs64 format and Roaster used a Q4_K_M GGUF of an uncensored fine-tune. Output quality has not been compared.
- **Cache-route for Roaster is not benchmarked yet.** Colibri's best results use it and Roaster's headline number doesn't, so Roaster has room to improve.
- **Short, single runs.** Each figure is one run on one machine. Different thread counts were used (Colibri 8, Roaster 4).
- **No prefill comparison yet,** because Colibri's prefill speed was never measured separately.

### Reproducing

To get comparable numbers, use the **same model file and quantization** as these benchmarks:

> `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` (Qwen3.6-35B-A3B, **Q4_K_M**)

A different fine-tune or quantization (Q6_K, Q8_0, IQ-quants, etc.) changes file size, SSD read volume, RAM use and speed, so results won't line up. Also use the same expert cache (`--expert-cache 64`), threads (`--threads 4`), 10 GB RAM cap and one slot.

Run the server and a chat as described in the [usage guide](USAGE.md). After each reply, Roaster prints:

```
Performance:
  Prompt tokens:  16
  Prefill speed:  9.73 tokens/s     <- prompt processing
  Decode tokens:  45
  Generation TPS: 10.22 tokens/s    <- the number used in this section
  Turn wait:      0.000 s           <- time spent queued behind other chats
```

Use the same prompt, `--temp 0` and the same `--max-tokens` when comparing runs.

### Tuning tip (AMD iGPU/APU)

The CPU and iGPU share one power budget. On the test machine, forcing the GPU to its high performance level and using the performance governor helped, and lowering the CPU clock gave even better results, probably because it left more power for the GPU. This is anecdotal and untested on other hardware.

```bash
# the card number may differ on your machine (check /sys/class/drm/)
sudo sh -c "echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level"
sudo cpupower frequency-set -g performance
cpupower frequency-info
```

These settings reset on reboot.
