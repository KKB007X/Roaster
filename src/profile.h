#ifndef PROFILE_H
#define PROFILE_H
/*
 * Lightweight per-phase profiler. Enable with:  ROASTER_PROFILE=1 ./roaster ...
 * Disabled cost: one predictable branch per probe.
 * A report is printed to stderr at the start of each new request (from
 * model_reset) and at exit. "prefill" counts every prompt token except the
 * last; "decode" counts the last prompt token and all generated tokens.
 * Phases marked with a leading '-' are measured INSIDE the phases above them,
 * so do not add them to the totals.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    PF_TOKEN, PF_EMBED, PF_ATTN, PF_DELTA_PROJ, PF_DELTA_CONV, PF_DELTA_STEP,
    PF_DELTA_OUT, PF_ROUTER, PF_EXP_PREFETCH, PF_EXP_MISS_SYNC, PF_EXP_COMPUTE,
    PF_SHARED, PF_LMHEAD, PF_SAMPLE,
    /* nested / sub-measurements */
    PF_VK_Q4_PREP, PF_VK_Q4_GPU, PF_VK_Q4_POST,
    PF_VK_Q6_PREP, PF_VK_Q6_GPU, PF_VK_Q6_POST,
    PF_VK_BATCH_PREP, PF_VK_BATCH_GPU, PF_VK_BATCH_POST, PF_VK_BATCH_JOBS,
    PF_COEXEC_CPU, PF_COEXEC_WAIT,
    PF_CPU_Q4, PF_CPU_Q6, PF_EXP_REQ, PF_EXP_MISS,
    PF_COUNT
};

extern int g_prof_on;
extern int g_prof_set;
extern double g_prof_t[2][PF_COUNT];
extern uint64_t g_prof_n[2][PF_COUNT];
extern uint64_t g_prof_bytes[2][PF_COUNT];

static inline double prof_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#define PROF_SET(want_logits) do { if (g_prof_on) g_prof_set = (want_logits) ? 1 : 0; } while (0)
#define PROF_T0(v) double v = g_prof_on ? prof_now() : 0.0
#define PROF_RESTART(v) do { if (g_prof_on) (v) = prof_now(); } while (0)
#define PROF_ADD(id, v, nbytes) do { if (g_prof_on) { \
    g_prof_t[g_prof_set][id] += prof_now() - (v); \
    g_prof_n[g_prof_set][id]++; \
    g_prof_bytes[g_prof_set][id] += (uint64_t)(nbytes); } } while (0)
#define PROF_ADD_T(id, t0, t1, nbytes) do { if (g_prof_on) { \
    g_prof_t[g_prof_set][id] += (t1) - (t0); \
    g_prof_n[g_prof_set][id]++; \
    g_prof_bytes[g_prof_set][id] += (uint64_t)(nbytes); } } while (0)
#define PROF_COUNTN(id, n, nbytes) do { if (g_prof_on) { \
    g_prof_n[g_prof_set][id] += (uint64_t)(n); \
    g_prof_bytes[g_prof_set][id] += (uint64_t)(nbytes); } } while (0)

int kernel_omp_team_size(void);
int model_omp_team_size(void);
void kernel_coexec_print(FILE *f);
void prof_init(void);
void prof_report_reset(void);

#ifdef PROFILE_IMPL
int g_prof_on = 0;
int g_prof_set = 0;
double g_prof_t[2][PF_COUNT];
uint64_t g_prof_n[2][PF_COUNT];
uint64_t g_prof_bytes[2][PF_COUNT];

static const char *const prof_names[PF_COUNT] = {
    "token total", "embedding", "full attention", "deltanet projections",
    "deltanet conv+l2norm", "deltanet state step", "deltanet norm+out proj",
    "moe router+select", "expert prefetch (parallel reads)",
    "expert sync miss loads", "expert compute (incl. sync loads)",
    "shared expert", "lm head", "sampler",
    "- vk q4: host prep+record", "- vk q4: submit+gpu+wait", "- vk q4: readback",
    "- vk q6: host prep+record", "- vk q6: submit+gpu+wait", "- vk q6: readback",
    "- vk batch: host prep+record", "- vk batch: submit+gpu+wait", "- vk batch: readback", "- vk batch: matvecs inside batches",
    "- coexec: CPU rows (overlapped)", "- coexec: CPU waited for GPU",
    "- cpu q4 fallback", "- cpu q6 fallback", "- expert requests", "- expert misses"
};

void prof_report_reset(void)
{
    if (!g_prof_on) return;
    if (g_prof_n[0][PF_TOKEN] || g_prof_n[1][PF_TOKEN]) {
        const char *wp = getenv("OMP_WAIT_POLICY");
        fprintf(stderr,
            "\n[openmp] team size inside kernels.c = %d, model.c = %d "
            "(0 means that file was compiled WITHOUT -fopenmp and its parallel loops run on 1 thread); "
            "OMP_NUM_THREADS=%s OMP_WAIT_POLICY=%s\n",
            kernel_omp_team_size(), model_omp_team_size(),
            getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "(unset)",
            wp ? wp : "(unset)");
    }
    {
        const char *hp = getenv("ROASTER_HUGEPAGES");
        if (hp && *hp && *hp != '0') {
            long kb = -1;
            FILE *f = fopen("/proc/self/smaps_rollup", "r");
            if (f) {
                char line[256];
                while (fgets(line, sizeof line, f))
                    if (sscanf(line, "AnonHugePages: %ld kB", &kb) == 1) break;
                fclose(f);
            }
            long thp_alloc = -1, thp_fallback = -1;
            FILE *vm = fopen("/proc/vmstat", "r");
            if (vm) {
                char line[256]; long v;
                while (fgets(line, sizeof line, vm)) {
                    if (sscanf(line, "thp_fault_alloc %ld", &v) == 1) thp_alloc = v;
                    else if (sscanf(line, "thp_fault_fallback %ld", &v) == 1) thp_fallback = v;
                }
                fclose(vm);
            }
            char defrag[96] = "?", mode[96] = "?";
            FILE *d = fopen("/sys/kernel/mm/transparent_hugepage/defrag", "r");
            if (d) { if (fgets(defrag, sizeof defrag, d)) defrag[strcspn(defrag, "\n")] = 0; fclose(d); }
            d = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
            if (d) { if (fgets(mode, sizeof mode, d)) mode[strcspn(mode, "\n")] = 0; fclose(d); }
            fprintf(stderr,
                "[hugepages] AnonHugePages in use now: %ld MB (includes the expert cache once warmed up)\n"
                "[hugepages] host-wide THP: enabled=%s defrag=%s thp_fault_alloc=%ld thp_fault_fallback=%ld "
                "(fallback >> alloc means memory is too fragmented to hand out 2 MB pages)\n",
                kb >= 0 ? kb / 1024 : -1, mode, defrag, thp_alloc, thp_fallback);
        }
    }
    kernel_coexec_print(stderr);
    for (int s = 0; s < 2; s++) {
        uint64_t tokens = g_prof_n[s][PF_TOKEN];
        if (!tokens) continue;
        double tot = g_prof_t[s][PF_TOKEN];
        fprintf(stderr,
            "\n=== ROASTER_PROFILE %s: %llu tokens, %.1f ms/token (%.2f tok/s) ===\n"
            "%-36s %9s %9s %6s %10s %10s %8s\n",
            s ? "decode" : "prefill", (unsigned long long)tokens,
            1e3 * tot / (double)tokens, (double)tokens / tot,
            "phase", "total s", "ms/token", "%tok", "calls/tok", "avg us", "GB/s");
        for (int i = 1; i < PF_COUNT; i++) {
            double t = g_prof_t[s][i];
            uint64_t n = g_prof_n[s][i], b = g_prof_bytes[s][i];
            if (!n) continue;
            fprintf(stderr, "%-36s %9.3f %9.2f %5.1f%% %10.1f ", prof_names[i], t,
                    1e3 * t / (double)tokens, 100.0 * t / tot, (double)n / (double)tokens);
            if (t > 0) fprintf(stderr, "%10.1f ", 1e6 * t / (double)n); else fprintf(stderr, "%10s ", "-");
            if (t > 0 && b) fprintf(stderr, "%8.2f", (double)b / t / 1e9);
            else if (b) fprintf(stderr, "%5.0f MB/tok", (double)b / (double)tokens / 1e6);
            fputc('\n', stderr);
        }
        {
            uint64_t cb = g_prof_bytes[s][PF_COEXEC_CPU], gb = g_prof_bytes[s][PF_VK_BATCH_GPU];
            if (cb)
                fprintf(stderr, "[coexec] %s: CPU computed %.1f%% of GPU-batched matvec bytes; CPU waited for the GPU %.1f ms/token, "
                        "(PF 'vk batch: submit+gpu+wait' time now includes the overlapped CPU work)\n",
                        s ? "decode" : "prefill", 100.0 * (double)cb / (double)(cb + gb),
                        1e3 * g_prof_t[s][PF_COEXEC_WAIT] / (double)tokens);
        }
        uint64_t req = g_prof_n[s][PF_EXP_REQ], miss = g_prof_n[s][PF_EXP_MISS];
        if (req)
            fprintf(stderr, "expert cache: %llu requests, %llu misses, hit rate %.1f%%, %.0f MB read/token\n",
                    (unsigned long long)req, (unsigned long long)miss,
                    100.0 * (1.0 - (double)miss / (double)req),
                    (double)g_prof_bytes[s][PF_EXP_MISS] / (double)tokens / 1e6);
    }
    memset(g_prof_t, 0, sizeof(g_prof_t));
    memset(g_prof_n, 0, sizeof(g_prof_n));
    memset(g_prof_bytes, 0, sizeof(g_prof_bytes));
}

void prof_init(void)
{
    static int done = 0;
    if (done) return;
    done = 1;
    const char *e = getenv("ROASTER_PROFILE");
    if (e && *e && *e != '0') {
        g_prof_on = 1;
        atexit(prof_report_reset);
    }
}
#endif /* PROFILE_IMPL */
#endif