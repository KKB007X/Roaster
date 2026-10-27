#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* Float kernels */
float kernel_dot_f32(const float *a, const float *b, size_t n);
void  kernel_matvec_f32(const float *matrix, const float *vector, float *output, size_t rows, size_t cols);

/* Quantized dot products and matrix-vector products (CPU reference / fallback) */
float kernel_dot_q4_k(const void *weights, const float *x, size_t n);
float kernel_dot_q6_k(const void *weights, const float *x, size_t n);
void  kernel_matvec_q4_k(const void *matrix, const float *vector, float *output, size_t rows, size_t cols);
void  kernel_matvec_q6_k(const void *matrix, const float *vector, float *output, size_t rows, size_t cols);

/* Normalization and activations */
void  kernel_rmsnorm(float *output, const float *input, const float *weight, size_t n, float epsilon);
void  kernel_l2norm(float *output, const float *input, size_t n, float epsilon);
float kernel_sigmoid(float x);
float kernel_silu(float x);
void  kernel_silu_array(float *x, size_t n);

/* Embedding row (Q4_K) and causal convolution */
void kernel_embedding_row_q4_k(const void *weights, size_t token_id, float *output, size_t hidden_size);
void kernel_conv1d_causal_f32(const float *input, const float *history, const float *kernel,
                              float *output, size_t channels, size_t kernel_size);

/*
 * Batched quantized matvec. Jobs are independent (a job must not read another job's output).
 * With Vulkan they run as ONE submit + ONE wait; otherwise they fall back to the CPU kernels.
 * Jobs sharing the same `vector` pointer and `cols` upload that vector once.
 */
typedef struct {
    int is_q6;             /* 0 = Q4_K, 1 = Q6_K */
    const void *matrix;
    const float *vector;
    float *output;
    size_t rows;
    size_t cols;
} KernelMatvecJob;

void kernel_matvec_quant_batch(const KernelMatvecJob *jobs, int n);

/* Split-phase variant: begin() submits all jobs and returns at once (1 = a GPU batch is in flight).
 * The caller may do independent CPU work, then MUST call kernel_matvec_quant_end(). `jobs` must
 * stay valid until end(). */
int  kernel_matvec_quant_begin(const KernelMatvecJob *jobs, int n);
void kernel_matvec_quant_end(void);

/* One MoE expert FFN: out = down( silu(gate x) * (up x) ), gate/up: inter x hidden, down: hidden x inter. */
typedef struct {
    const void *gate, *up, *down;
    int gate_q6, up_q6, down_q6;     /* 0 = Q4_K, 1 = Q6_K */
    float *out;                      /* `hidden` floats */
} KernelFfnExpert;

/* Runs n experts sharing the input x (gate/up -> SiLU*up -> down) in ONE GPU submit.
 * Returns 1 on success; 0 = unavailable and the caller must use the generic path.
 * inter and hidden must be multiples of 256 and n <= 9. */
int kernel_ffn_fused(const KernelFfnExpert *experts, int n, const float *x, size_t inter, size_t hidden);

/* Vulkan backend lifecycle; CPU kernels remain available as fallback. */
void   kernel_vk_shutdown(void);
void   kernel_vk_release_host_ptr(const void *ptr);
int    kernel_vk_register_region(const void *ptr, size_t size);

/* Driver-allocated, CPU-mapped, GPU-fast weight memory (NULL if unavailable -> use malloc). */
void  *kernel_vk_alloc_weights(size_t size, size_t *granted);
int    kernel_vk_owns_weights(const void *ptr);
int    kernel_vk_free_weights(void *ptr);   /* 1 if ptr was driver memory (nothing more to do) */

#endif