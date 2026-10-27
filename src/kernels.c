#define _POSIX_C_SOURCE 200809L
#include "kernels.h"
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdint.h>
#include <vulkan/vulkan.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define KERNELS_HAVE_AVX2 1
#endif

#define QK_K        256
#define Q4_K_SIZE   144
#define Q6_K_SIZE   210

/* ============================================================
 * GGML quantization structures
 * ============================================================ */

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_k;
typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} block_q6_k;
_Static_assert(
    sizeof(block_q4_k) == Q4_K_SIZE,
    "Q4_K block must be 144 bytes"
);

_Static_assert(
    sizeof(block_q6_k) == Q6_K_SIZE,
    "Q6_K block must be 210 bytes"
);

/* ============================================================
 * FP16 conversion
 * ============================================================ */

static float fp16_to_float_soft(uint16_t h) __attribute__((unused));
static float fp16_to_float(uint16_t h)
{
#if defined(__F16C__) && defined(KERNELS_HAVE_AVX2)
    return _cvtsh_ss(h);
#else
    return fp16_to_float_soft(h);
#endif
}

static float fp16_to_float_soft(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mantissa = (uint32_t)h & 0x03FFu;
    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            float value = (float)mantissa / 1024.0f;
            value *= 0x1p-14f;
            return sign ? -value : value;
        }

    } else if (exponent == 31) {
        result =
            sign |
            0x7F800000u |
            (mantissa << 13);
    } else {
        uint32_t exp32 = exponent - 15 + 127;
        result =
            sign |
            (exp32 << 23) |
            (mantissa << 13);
    }

    float value;
    memcpy(&value, &result, sizeof(value));
    return value;
}

/* ============================================================
 * F32 operations
 * ============================================================ */

float kernel_dot_f32(
    const float *a,
    const float *b,
    size_t n
)
{
#ifdef KERNELS_HAVE_AVX2
    size_t i = 0;
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
    }
    a0 = _mm256_add_ps(a0, a1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#endif
}

void kernel_matvec_f32(
    const float *matrix,
    const float *vector,
    float *output,
    size_t rows,
    size_t cols
)
{
    #pragma omp parallel for schedule(static) if (rows * cols >= (1u << 21))
    for (size_t row = 0; row < rows; row++) {
        const float *row_ptr = matrix + row * cols;
        output[row] = kernel_dot_f32(
            row_ptr,
            vector,
            cols
        );
    }
}

/* ============================================================
 * Q4_K helpers
 * ============================================================ */

static void q4_k_get_scales(
    const block_q4_k *block,
    int *scales,
    int *mins
)
{
    for (int j = 0; j < 8; j++) {
        if (j < 4) {
            scales[j] = block->scales[j] & 63;
            mins[j] = block->scales[j + 4] & 63;
        } else {
            scales[j] =
                (block->scales[j + 4] & 0x0F) |
                ((block->scales[j - 4] >> 6) << 4);
            mins[j] =
                (block->scales[j + 4] >> 4) |
                ((block->scales[j] >> 6) << 4);
        }
    }
}

static float dot_q4_k_block(
    const block_q4_k *block,
    const float *x
)
{
    float d = fp16_to_float(block->d);
    float dmin = fp16_to_float(block->dmin);
    int scales[8];
    int mins[8];
    q4_k_get_scales(
        block,
        scales,
        mins
    );
    float sum = 0.0f;
    int q_offset = 0;
    int x_offset = 0;
    for (int group = 0; group < 4; group++) {
        float scale0 =
            d * (float)scales[group * 2];
        float min0 =
            dmin * (float)mins[group * 2];
        float scale1 =
            d * (float)scales[group * 2 + 1];
        float min1 =
            dmin * (float)mins[group * 2 + 1];
        for (int i = 0; i < 32; i++) {
            int q = block->qs[q_offset + i] & 0x0F;
            sum +=
                (scale0 * q - min0) *
                x[x_offset + i];
        }

        for (int i = 0; i < 32; i++) {
            int q = block->qs[q_offset + i] >> 4;
            sum +=
                (scale1 * q - min1) *
                x[x_offset + 32 + i];
        }

        q_offset += 32;
        x_offset += 64;
    }

    return sum;
}


#ifdef KERNELS_HAVE_AVX2
static inline float hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
static inline __m256 u8x8_to_f32(__m128i b)
{
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b));
}
static inline __m256 i8x8_to_f32(__m128i b)
{
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
}

/*
 * (d*sc*q - dmin*m) * x  ==  d*sc * sum(q*x)  -  dmin*m * sum(x)
 * so the inner loop only needs q*x and x sums, both vectorizable.
 */
static float dot_q4_k_row_avx2(const block_q4_k *blocks, const float *x, size_t nb)
{
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    __m256 total = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        const block_q4_k *blk = &blocks[b];
        const float *xb = x + b * QK_K;
        float d = fp16_to_float(blk->d);
        float dmin = fp16_to_float(blk->dmin);
        int sc[8], mn[8];
        q4_k_get_scales(blk, sc, mn);
        for (int g = 0; g < 4; g++) {
            __m256i raw = _mm256_loadu_si256((const __m256i *)(blk->qs + 32 * g));
            __m256i lo = _mm256_and_si256(raw, m0f);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m0f);
            __m128i lo0 = _mm256_castsi256_si128(lo);
            __m128i lo1 = _mm256_extracti128_si256(lo, 1);
            __m128i hi0 = _mm256_castsi256_si128(hi);
            __m128i hi1 = _mm256_extracti128_si256(hi, 1);
            const float *xl = xb + 64 * g;
            const float *xh = xl + 32;
            __m256 xl0 = _mm256_loadu_ps(xl),      xl1 = _mm256_loadu_ps(xl + 8);
            __m256 xl2 = _mm256_loadu_ps(xl + 16), xl3 = _mm256_loadu_ps(xl + 24);
            __m256 xh0 = _mm256_loadu_ps(xh),      xh1 = _mm256_loadu_ps(xh + 8);
            __m256 xh2 = _mm256_loadu_ps(xh + 16), xh3 = _mm256_loadu_ps(xh + 24);
            __m256 sql = _mm256_mul_ps(u8x8_to_f32(lo0), xl0);
            sql = _mm256_fmadd_ps(u8x8_to_f32(_mm_srli_si128(lo0, 8)), xl1, sql);
            sql = _mm256_fmadd_ps(u8x8_to_f32(lo1), xl2, sql);
            sql = _mm256_fmadd_ps(u8x8_to_f32(_mm_srli_si128(lo1, 8)), xl3, sql);
            __m256 sqh = _mm256_mul_ps(u8x8_to_f32(hi0), xh0);
            sqh = _mm256_fmadd_ps(u8x8_to_f32(_mm_srli_si128(hi0, 8)), xh1, sqh);
            sqh = _mm256_fmadd_ps(u8x8_to_f32(hi1), xh2, sqh);
            sqh = _mm256_fmadd_ps(u8x8_to_f32(_mm_srli_si128(hi1, 8)), xh3, sqh);
            __m256 sxl = _mm256_add_ps(_mm256_add_ps(xl0, xl1), _mm256_add_ps(xl2, xl3));
            __m256 sxh = _mm256_add_ps(_mm256_add_ps(xh0, xh1), _mm256_add_ps(xh2, xh3));
            total = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)sc[2 * g]), sql, total);
            total = _mm256_fnmadd_ps(_mm256_set1_ps(dmin * (float)mn[2 * g]), sxl, total);
            total = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)sc[2 * g + 1]), sqh, total);
            total = _mm256_fnmadd_ps(_mm256_set1_ps(dmin * (float)mn[2 * g + 1]), sxh, total);
        }
    }
    return hsum256(total);
}

static inline __m256 q6_accum(__m256 total, __m256i q, const float *xp,
                              float d, int s0, int s1)
{
    __m128i a = _mm256_castsi256_si128(q);
    __m128i c = _mm256_extracti128_si256(q, 1);
    __m256 v0 = _mm256_mul_ps(i8x8_to_f32(a), _mm256_loadu_ps(xp));
    v0 = _mm256_fmadd_ps(i8x8_to_f32(_mm_srli_si128(a, 8)), _mm256_loadu_ps(xp + 8), v0);
    __m256 v1 = _mm256_mul_ps(i8x8_to_f32(c), _mm256_loadu_ps(xp + 16));
    v1 = _mm256_fmadd_ps(i8x8_to_f32(_mm_srli_si128(c, 8)), _mm256_loadu_ps(xp + 24), v1);
    total = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)s0), v0, total);
    total = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)s1), v1, total);
    return total;
}

static float dot_q6_k_row_avx2(const block_q6_k *blocks, const float *x, size_t nb)
{
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i m03 = _mm256_set1_epi8(3);
    const __m256i c32 = _mm256_set1_epi8(32);
    __m256 total = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        const block_q6_k *blk = &blocks[b];
        const float d = fp16_to_float(blk->d);
        const uint8_t *ql = blk->ql;
        const uint8_t *qh = blk->qh;
        const int8_t *sc = blk->scales;
        const float *xb = x + b * QK_K;
        for (int n = 0; n < 2; n++) {
            __m256i r0 = _mm256_loadu_si256((const __m256i *)ql);
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i h  = _mm256_loadu_si256((const __m256i *)qh);
            __m256i q1 = _mm256_or_si256(_mm256_and_si256(r0, m0f),
                _mm256_slli_epi16(_mm256_and_si256(h, m03), 4));
            __m256i q2 = _mm256_or_si256(_mm256_and_si256(r1, m0f),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), m03), 4));
            __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(r0, 4), m0f),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), m03), 4));
            __m256i q4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(r1, 4), m0f),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), m03), 4));
            q1 = _mm256_sub_epi8(q1, c32);
            q2 = _mm256_sub_epi8(q2, c32);
            q3 = _mm256_sub_epi8(q3, c32);
            q4 = _mm256_sub_epi8(q4, c32);
            const float *xs = xb + n * 128;
            total = q6_accum(total, q1, xs,      d, sc[0], sc[1]);
            total = q6_accum(total, q2, xs + 32, d, sc[2], sc[3]);
            total = q6_accum(total, q3, xs + 64, d, sc[4], sc[5]);
            total = q6_accum(total, q4, xs + 96, d, sc[6], sc[7]);
            ql += 64; qh += 32; sc += 8;
        }
    }
    return hsum256(total);
}
#endif /* KERNELS_HAVE_AVX2 */

float kernel_dot_q4_k(
    const void *weights,
    const float *x,
    size_t n
)
{
    const block_q4_k *blocks =
        (const block_q4_k *)weights;
#ifdef KERNELS_HAVE_AVX2
    if (n % QK_K == 0)
        return dot_q4_k_row_avx2(blocks, x, n / QK_K);
#endif
    size_t block_count =
        (n + QK_K - 1) / QK_K;
    float sum = 0.0f;
    for (size_t b = 0; b < block_count; b++) {
        size_t base = b * QK_K;
        size_t remaining = n - base;
        if (remaining > QK_K)
            remaining = QK_K;
        if (remaining == QK_K) {
            sum += dot_q4_k_block(
                &blocks[b],
                x + base
            );
            continue;
        }

        /*
         * Partial-block fallback.
         */
        float d = fp16_to_float(blocks[b].d);
        float dmin = fp16_to_float(blocks[b].dmin);
        int scales[8];
        int mins[8];
        q4_k_get_scales(
            &blocks[b],
            scales,
            mins
        );
        for (size_t i = 0; i < remaining; i++) {
            int group = (int)(i / 32);
            uint8_t packed =
                blocks[b].qs[i / 2];
            int q =
                (i & 1)
                ? packed >> 4
                : packed & 0x0F;
            float value =
                d * scales[group] * q -
                dmin * mins[group];
            sum += value * x[base + i];
        }
    }

    return sum;
}

/* ============================================================
 * Q4_K matrix-vector
 * ============================================================ */

static void matvec_q4_cpu(
    const void *matrix,
    const float *vector,
    float *output,
    size_t rows,
    size_t cols
)
{

    size_t blocks_per_row =
        (cols + QK_K - 1) / QK_K;
    const uint8_t *data =
        (const uint8_t *)matrix;
    #pragma omp parallel for schedule(static) if (rows * blocks_per_row >= 16384)
    for (size_t row = 0; row < rows; row++) {
        const void *row_data =
            data +
            row * blocks_per_row * Q4_K_SIZE;
        output[row] =
            kernel_dot_q4_k(
                row_data,
                vector,
                cols
            );
    }
}

/* ============================================================
 * Q6_K helpers
 * ============================================================ */

static float dot_q6_k_block(
    const block_q6_k *block,
    const float *x
)
{
    const float d =
        fp16_to_float(block->d);
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t *sc = block->scales;
    float sum = 0.0f;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 =
                (ql[l] & 0x0F) |
                (((qh[l] >> 0) & 0x03) << 4);
            int q2 =
                (ql[l + 32] & 0x0F) |
                (((qh[l] >> 2) & 0x03) << 4);
            int q3 =
                (ql[l] >> 4) |
                (((qh[l] >> 4) & 0x03) << 4);
            int q4 =
                (ql[l + 32] >> 4) |
                (((qh[l] >> 6) & 0x03) << 4);
            q1 -= 32;
            q2 -= 32;
            q3 -= 32;
            q4 -= 32;
            sum +=
                d * sc[is + 0] *
                q1 * x[n + l];
            sum +=
                d * sc[is + 2] *
                q2 * x[n + l + 32];
            sum +=
                d * sc[is + 4] *
                q3 * x[n + l + 64];
            sum +=
                d * sc[is + 6] *
                q4 * x[n + l + 96];
        }

        ql += 64;
        qh += 32;
        sc += 8;
    }

    return sum;
}

float kernel_dot_q6_k(
    const void *weights,
    const float *x,
    size_t n
)
{
    const block_q6_k *blocks =
        (const block_q6_k *)weights;
#ifdef KERNELS_HAVE_AVX2
    if (n % QK_K == 0)
        return dot_q6_k_row_avx2(blocks, x, n / QK_K);
#endif
    size_t block_count =
        (n + QK_K - 1) / QK_K;
    float sum = 0.0f;
    for (size_t b = 0; b < block_count; b++) {
        size_t base = b * QK_K;
        size_t remaining = n - base;
        if (remaining > QK_K)
            remaining = QK_K;
        if (remaining == QK_K) {
            sum += dot_q6_k_block(
                &blocks[b],
                x + base
            );
            continue;
        }

        /*
         * Partial-block fallback.
         */
        float d =
            fp16_to_float(blocks[b].d);
        for (size_t i = 0; i < remaining; i++) {
            int group = (int)(i / 16);
            uint8_t ql_byte =
                blocks[b].ql[i / 2];
            int low =
                (i & 1)
                ? ql_byte >> 4
                : ql_byte & 0x0F;
            uint8_t qh_byte =
                blocks[b].qh[i / 4];
            int high =
                (qh_byte >> ((i % 4) * 2)) & 3;
            int q =
                low | (high << 4);
            q -= 32;
            float value =
                d *
                (float)blocks[b].scales[group] *
                (float)q;
            sum += value * x[base + i];
        }
    }

    return sum;
}

/* ============================================================
 * Q6_K matrix-vector
 * ============================================================ */

static void matvec_q6_cpu(
    const void *matrix,
    const float *vector,
    float *output,
    size_t rows,
    size_t cols
)
{

    size_t blocks_per_row =
        (cols + QK_K - 1) / QK_K;
    const uint8_t *data =
        (const uint8_t *)matrix;
    #pragma omp parallel for schedule(static) if (rows * blocks_per_row >= 16384)
    for (size_t row = 0; row < rows; row++) {
        const void *row_data =
            data +
            row * blocks_per_row * Q6_K_SIZE;
        output[row] =
            kernel_dot_q6_k(
                row_data,
                vector,
                cols
            );
    }
}

/* ============================================================
 * Normalization
 * ============================================================ */

void kernel_rmsnorm(
    float *output,
    const float *input,
    const float *weight,
    size_t n,
    float epsilon
)
{
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; i++)
        sum_sq += input[i] * input[i];
    float mean =
        sum_sq / (float)n;
    float inv_rms =
        1.0f / sqrtf(mean + epsilon);
    for (size_t i = 0; i < n; i++) {
        output[i] =
            input[i] *
            inv_rms *
            weight[i];
    }
}

void kernel_l2norm(
    float *output,
    const float *input,
    size_t n,
    float epsilon
)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
        sum += input[i] * input[i];
    float inv_norm =
        1.0f / sqrtf(sum + epsilon);
    for (size_t i = 0; i < n; i++)
        output[i] =
            input[i] * inv_norm;
}

/* ============================================================
 * Activations
 * ============================================================ */

float kernel_sigmoid(float x)
{
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }

    float z = expf(x);
    return z / (1.0f + z);
}

float kernel_silu(float x)
{
    return x * kernel_sigmoid(x);
}

void kernel_silu_array(
    float *x,
    size_t n
)
{
    #pragma omp parallel for schedule(static) if (n >= (1u << 16))
    for (size_t i = 0; i < n; i++)
        x[i] = kernel_silu(x[i]);
}

/* ============================================================
 * Q4_K embedding
 * ============================================================ */

/* Decode one whole token row. Scales are decoded once per 256-block
 * instead of once per element. Requires hidden_size % 256 == 0. */
void kernel_embedding_row_q4_k(
    const void *weights,
    size_t token_id,
    float *output,
    size_t hidden_size
)
{
    const block_q4_k *blocks =
        (const block_q4_k *)weights + (token_id * hidden_size) / QK_K;
    size_t nb = hidden_size / QK_K;
    for (size_t b = 0; b < nb; b++) {
        const block_q4_k *blk = &blocks[b];
        float d = fp16_to_float(blk->d);
        float dmin = fp16_to_float(blk->dmin);
        int sc[8], mn[8];
        q4_k_get_scales(blk, sc, mn);
        float *o = output + b * QK_K;
        for (int g = 0; g < 4; g++) {
            const uint8_t *q = blk->qs + 32 * g;
            float s0 = d * (float)sc[2 * g],     m0 = dmin * (float)mn[2 * g];
            float s1 = d * (float)sc[2 * g + 1], m1 = dmin * (float)mn[2 * g + 1];
            for (int i = 0; i < 32; i++) {
                o[64 * g + i]      = s0 * (float)(q[i] & 0x0F) - m0;
                o[64 * g + 32 + i] = s1 * (float)(q[i] >> 4)   - m1;
            }
        }
    }
}

/* ============================================================
 * Causal convolution
 * ============================================================ */

void kernel_conv1d_causal_f32(
    const float *input,
    const float *history,
    const float *kernel,
    float *output,
    size_t channels,
    size_t kernel_size
)
{
    if (kernel_size == 0)
        return;
    size_t history_len =
        kernel_size - 1;
    for (size_t c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (size_t k = 0; k < history_len; k++) {
            sum +=
                history[k * channels + c] *
                kernel[c * kernel_size + k];
        }

        sum +=
            input[c] *
            kernel[c * kernel_size + history_len];
        output[c] = sum;
    }
}


/* ======================== Vulkan compute backend ======================== */
typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    VkDeviceSize size;
    VkDeviceSize allocation_size;
    uint32_t memory_type;
    int coherent;
} HostBuffer;

typedef struct ImportedBuffer {
    const void *host_ptr;
    size_t host_size;
    HostBuffer gpu;
    struct ImportedBuffer *next;
    struct ImportedBuffer *hnext;   /* hash-bucket chain */
    struct ImportedBuffer *rnext;   /* region chain */
    int is_region;                  /* matrices live at offsets inside this import */
    int is_driver;                  /* driver-allocated (vkAllocateMemory), not a host import */
} ImportedBuffer;

#define VK_BATCH_MAX 32
#define VK_BATCH_ALIGN 256u
#define IMPORT_BUCKETS 8192u
static inline size_t import_bucket(const void *p)
{
    return (size_t)(((uintptr_t)p >> 12) * 2654435761u) & (IMPORT_BUCKETS - 1u);
}

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
    PFN_vkGetMemoryHostPointerPropertiesEXT get_host_pointer_properties;
    int host_import_supported;
    VkDeviceSize min_host_alignment;
    int initialized;
    int disabled;
    int logged_failure;
    int logged_import_failure;
    VkDescriptorSetLayout descriptor_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline q4_pipeline;
    VkPipeline q6_pipeline;
    VkPipeline silu_pipeline;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet batch_sets[VK_BATCH_MAX];
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    HostBuffer vector_buffer;
    HostBuffer output_buffer;
    size_t vector_capacity;
    size_t output_capacity;
    ImportedBuffer *imports;
    ImportedBuffer *regions;
    ImportedBuffer *import_table[IMPORT_BUCKETS];
} VulkanState;

static VulkanState g_vk;
static pthread_mutex_t g_vk_mutex = PTHREAD_MUTEX_INITIALIZER;

static VkDeviceSize align_up(VkDeviceSize n, VkDeviceSize a) {
    return a ? ((n + a - 1u) / a) * a : n;
}

static int memory_type(uint32_t bits, VkMemoryPropertyFlags required,
                       VkMemoryPropertyFlags preferred, uint32_t *out,
                       VkMemoryPropertyFlags *actual) {
    int fallback = -1;
    for (uint32_t i = 0; i < g_vk.memory_properties.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) == 0) continue;
        VkMemoryPropertyFlags flags = g_vk.memory_properties.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) continue;
        if ((flags & preferred) == preferred) {
            *out = i;
            if (actual) *actual = flags;
            return 1;
        }
        if (fallback < 0) fallback = (int)i;
    }
    if (fallback >= 0) {
        *out = (uint32_t)fallback;
        if (actual) *actual = g_vk.memory_properties.memoryTypes[fallback].propertyFlags;
        return 1;
    }
    return 0;
}

static void destroy_host_buffer(HostBuffer *b) {
    if (!b || !g_vk.device) return;
    if (b->mapped && b->memory) vkUnmapMemory(g_vk.device, b->memory);
    if (b->buffer) vkDestroyBuffer(g_vk.device, b->buffer, NULL);
    if (b->memory) vkFreeMemory(g_vk.device, b->memory, NULL);
    memset(b, 0, sizeof(*b));
}

static int create_host_buffer(VkDeviceSize requested, VkBufferUsageFlags usage,
                              int cpu_reads, HostBuffer *out) {
    memset(out, 0, sizeof(*out));
    if (!requested) requested = 4;
    VkDeviceSize size = align_up(requested, 4);
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_vk.device, &bi, NULL, &out->buffer) != VK_SUCCESS) goto fail;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_vk.device, out->buffer, &req);
    uint32_t type_index;
    VkMemoryPropertyFlags actual;
    int got = 0;
    /* Buffers the CPU READS from (kernel outputs) must be in CACHED host memory: the default
     * host-visible coherent type on AMD is write-combined/uncached, where CPU reads run at
     * ~100-500 MB/s. CPU writes (input vectors) are fine there, so only outputs ask for it. */
    if (cpu_reads)
        got = memory_type(req.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &type_index, &actual);
    if (!got &&
        !memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &type_index, &actual)) goto fail;
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type_index;
    if (vkAllocateMemory(g_vk.device, &ai, NULL, &out->memory) != VK_SUCCESS) goto fail;
    if (vkBindBufferMemory(g_vk.device, out->buffer, out->memory, 0) != VK_SUCCESS) goto fail;
    if (vkMapMemory(g_vk.device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->mapped) != VK_SUCCESS) goto fail;
    out->size = size;
    out->allocation_size = req.size;
    out->memory_type = type_index;
    out->coherent = (actual & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    return 1;
fail:
    destroy_host_buffer(out);
    return 0;
}

static int flush_host_buffer(const HostBuffer *b) {
    if (b->coherent) return 1;
    VkMappedMemoryRange range = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
    range.memory = b->memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    return vkFlushMappedMemoryRanges(g_vk.device, 1, &range) == VK_SUCCESS;
}

static int invalidate_host_buffer(const HostBuffer *b) {
    if (b->coherent) return 1;
    VkMappedMemoryRange range = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
    range.memory = b->memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    return vkInvalidateMappedMemoryRanges(g_vk.device, 1, &range) == VK_SUCCESS;
}

static int device_has_extension(VkPhysicalDevice device, const char *name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, NULL, &count, NULL) != VK_SUCCESS) return 0;
    VkExtensionProperties *props = calloc(count ? count : 1, sizeof(*props));
    if (!props) return 0;
    int found = 0;
    if (vkEnumerateDeviceExtensionProperties(device, NULL, &count, props) == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; ++i) {
            if (strcmp(props[i].extensionName, name) == 0) { found = 1; break; }
        }
    }
    free(props);
    return found;
}

/* Compiled shaders, embedded at the end of this file. */
extern const uint32_t roaster_spv_q4k_v2[];
extern const size_t roaster_spv_q4k_v2_size;
extern const uint32_t roaster_spv_q6k_v2[];
extern const size_t roaster_spv_q6k_v2_size;
extern const uint32_t roaster_spv_silu_v1[];
extern const size_t roaster_spv_silu_v1_size;

static int load_shader(const uint32_t *code, size_t bytes, VkShaderModule *out) {
    VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = bytes;
    ci.pCode = code;
    return vkCreateShaderModule(g_vk.device, &ci, NULL, out) == VK_SUCCESS;
}

static int create_pipeline(const uint32_t *code, size_t bytes, VkPipeline *pipeline) {
    VkShaderModule module = VK_NULL_HANDLE;
    if (!load_shader(code, bytes, &module)) return 0;
    VkPipelineShaderStageCreateInfo stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    ci.stage = stage;
    ci.layout = g_vk.pipeline_layout;
    VkResult result = vkCreateComputePipelines(g_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, pipeline);
    vkDestroyShaderModule(g_vk.device, module, NULL);
    return result == VK_SUCCESS;
}

static void destroy_imports(void) {
    ImportedBuffer *it = g_vk.imports;
    while (it) {
        ImportedBuffer *next = it->next;
        if (it->gpu.buffer) vkDestroyBuffer(g_vk.device, it->gpu.buffer, NULL);
        if (it->gpu.memory) vkFreeMemory(g_vk.device, it->gpu.memory, NULL);
        free(it);
        it = next;
    }
    g_vk.imports = NULL;
}

static void shutdown_locked(void) {
    if (!g_vk.device && !g_vk.instance) { memset(&g_vk, 0, sizeof(g_vk)); return; }
    if (g_vk.device) {
        vkDeviceWaitIdle(g_vk.device);
        destroy_imports();
        destroy_host_buffer(&g_vk.vector_buffer);
        destroy_host_buffer(&g_vk.output_buffer);
        if (g_vk.fence) vkDestroyFence(g_vk.device, g_vk.fence, NULL);
        if (g_vk.command_pool) vkDestroyCommandPool(g_vk.device, g_vk.command_pool, NULL);
        if (g_vk.descriptor_pool) vkDestroyDescriptorPool(g_vk.device, g_vk.descriptor_pool, NULL);
        if (g_vk.q4_pipeline) vkDestroyPipeline(g_vk.device, g_vk.q4_pipeline, NULL);
        if (g_vk.q6_pipeline) vkDestroyPipeline(g_vk.device, g_vk.q6_pipeline, NULL);
        if (g_vk.silu_pipeline) vkDestroyPipeline(g_vk.device, g_vk.silu_pipeline, NULL);
        if (g_vk.pipeline_layout) vkDestroyPipelineLayout(g_vk.device, g_vk.pipeline_layout, NULL);
        if (g_vk.descriptor_layout) vkDestroyDescriptorSetLayout(g_vk.device, g_vk.descriptor_layout, NULL);
        vkDestroyDevice(g_vk.device, NULL);
    }
    if (g_vk.instance) vkDestroyInstance(g_vk.instance, NULL);
    memset(&g_vk, 0, sizeof(g_vk));
}

static int initialize_locked(void) {
    if (g_vk.initialized) return 1;
    if (g_vk.disabled) return 0;

    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "Roaster";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName = "Roaster";
    app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, NULL, &g_vk.instance) != VK_SUCCESS) goto fail;

    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(g_vk.instance, &count, NULL) != VK_SUCCESS || count == 0) goto fail;
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    if (!devices) goto fail;
    if (vkEnumeratePhysicalDevices(g_vk.instance, &count, devices) != VK_SUCCESS) { free(devices); goto fail; }
    int chosen = -1;
    for (uint32_t i = 0; chosen < 0 && i < count; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        if (p.vendorID == 0x1002 && p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) { chosen = (int)i; break; }
    }
    if (chosen < 0) {
        for (uint32_t i = 0; i < count; ++i) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devices[i], &p);
            if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) { chosen = (int)i; break; }
        }
    }
    if (chosen < 0) { free(devices); goto fail; }
    g_vk.physical = devices[chosen];
    free(devices);
    vkGetPhysicalDeviceProperties(g_vk.physical, &g_vk.properties);
    vkGetPhysicalDeviceMemoryProperties(g_vk.physical, &g_vk.memory_properties);

    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.physical, &qcount, NULL);
    VkQueueFamilyProperties *qprops = calloc(qcount ? qcount : 1, sizeof(*qprops));
    if (!qprops) goto fail;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.physical, &qcount, qprops);
    int qfound = 0;
    for (uint32_t i = 0; i < qcount; ++i) {
        if (qprops[i].queueCount && (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            g_vk.queue_family = i; qfound = 1; break;
        }
    }
    free(qprops);
    if (!qfound) goto fail;

    g_vk.host_import_supported = device_has_extension(g_vk.physical, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    g_vk.min_host_alignment = 4096u;
    if (g_vk.host_import_supported) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_properties = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT
        };
        VkPhysicalDeviceProperties2 properties2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &host_properties
        };
        vkGetPhysicalDeviceProperties2(g_vk.physical, &properties2);
        if (host_properties.minImportedHostPointerAlignment != 0)
            g_vk.min_host_alignment = host_properties.minImportedHostPointerAlignment;
    }
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = g_vk.queue_family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
    const char *extensions[1] = { VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    if (g_vk.host_import_supported) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = extensions; }
    if (vkCreateDevice(g_vk.physical, &dci, NULL, &g_vk.device) != VK_SUCCESS) goto fail;
    vkGetDeviceQueue(g_vk.device, g_vk.queue_family, 0, &g_vk.queue);
    if (g_vk.host_import_supported) {
        g_vk.get_host_pointer_properties = (PFN_vkGetMemoryHostPointerPropertiesEXT)
            vkGetDeviceProcAddr(g_vk.device, "vkGetMemoryHostPointerPropertiesEXT");
        if (!g_vk.get_host_pointer_properties) g_vk.host_import_supported = 0;
    }

    VkDescriptorSetLayoutBinding bindings[3];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 3; dlci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(g_vk.device, &dlci, NULL, &g_vk.descriptor_layout) != VK_SUCCESS) goto fail;
    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 12 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &g_vk.descriptor_layout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(g_vk.device, &plci, NULL, &g_vk.pipeline_layout) != VK_SUCCESS) goto fail;
    /* _v2 shaders process several rows per workgroup (see rows_per_wg in dispatch_matvec);
     * the old one-row-per-workgroup shaders must never be loaded by this host code. */
    if (!create_pipeline(roaster_spv_q4k_v2, roaster_spv_q4k_v2_size, &g_vk.q4_pipeline) ||
        !create_pipeline(roaster_spv_q6k_v2, roaster_spv_q6k_v2_size, &g_vk.q6_pipeline)) goto fail;
    /* Optional: without it the fused MoE path is simply unavailable. */
    if (!create_pipeline(roaster_spv_silu_v1, roaster_spv_silu_v1_size, &g_vk.silu_pipeline))
        g_vk.silu_pipeline = VK_NULL_HANDLE;

    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * VK_BATCH_MAX };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = VK_BATCH_MAX; dpci.poolSizeCount = 1; dpci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(g_vk.device, &dpci, NULL, &g_vk.descriptor_pool) != VK_SUCCESS) goto fail;
    {
        VkDescriptorSetLayout bl[VK_BATCH_MAX];
        for (int i = 0; i < VK_BATCH_MAX; ++i) bl[i] = g_vk.descriptor_layout;
        VkDescriptorSetAllocateInfo bai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        bai.descriptorPool = g_vk.descriptor_pool; bai.descriptorSetCount = VK_BATCH_MAX; bai.pSetLayouts = bl;
        if (vkAllocateDescriptorSets(g_vk.device, &bai, g_vk.batch_sets) != VK_SUCCESS) goto fail;
    }

    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpci.queueFamilyIndex = g_vk.queue_family;
    if (vkCreateCommandPool(g_vk.device, &cpci, NULL, &g_vk.command_pool) != VK_SUCCESS) goto fail;
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = g_vk.command_pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_vk.device, &cbai, &g_vk.command_buffer) != VK_SUCCESS) goto fail;
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(g_vk.device, &fci, NULL, &g_vk.fence) != VK_SUCCESS) goto fail;
    g_vk.initialized = 1;
    return 1;
fail:
    if (!g_vk.logged_failure) {
        fprintf(stderr, "Roaster Vulkan: initialization failed; using CPU kernels\n");
        g_vk.logged_failure = 1;
    }
    g_vk.disabled = 1;
    shutdown_locked();
    g_vk.disabled = 1;
    return 0;
}

static ImportedBuffer *find_import(const void *ptr, size_t size) {
    for (ImportedBuffer *it = g_vk.import_table[import_bucket(ptr)]; it; it = it->hnext)
        if (it->host_ptr == ptr && it->host_size == size) return it;
    return NULL;
}

/* Is [ptr, ptr+size) inside a registered region? Returns the import plus descriptor offset/range. */
static ImportedBuffer *find_region(const void *ptr, size_t size, VkDeviceSize *off, VkDeviceSize *range) {
    uintptr_t p = (uintptr_t)ptr;
    VkDeviceSize al = g_vk.properties.limits.minStorageBufferOffsetAlignment;
    for (ImportedBuffer *it = g_vk.regions; it; it = it->rnext) {
        uintptr_t b = (uintptr_t)it->host_ptr;
        if (p < b || p + size > b + it->host_size) continue;
        VkDeviceSize o = (VkDeviceSize)(p - b);
        if (al && (o % al)) return NULL;
        VkDeviceSize r = align_up((VkDeviceSize)size, 16);
        if (r > it->host_size - o) r = it->host_size - o;
        if (r > g_vk.properties.limits.maxStorageBufferRange) return NULL;
        *off = o; *range = r;
        return it;
    }
    return NULL;
}

static ImportedBuffer *import_host_buffer(const void *ptr, size_t size) {
    if (!g_vk.host_import_supported || !ptr || !size) return NULL;
    if (((uintptr_t)ptr % (uintptr_t)g_vk.min_host_alignment) != 0) return NULL;
    ImportedBuffer *entry = calloc(1, sizeof(*entry));
    if (!entry) return NULL;
    VkDeviceSize buffer_size = align_up((VkDeviceSize)size, g_vk.min_host_alignment);
    VkExternalMemoryBufferCreateInfo external = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.pNext = &external; bi.size = buffer_size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_vk.device, &bi, NULL, &entry->gpu.buffer) != VK_SUCCESS) goto fail;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_vk.device, entry->gpu.buffer, &req);
    long page_l = sysconf(_SC_PAGESIZE);
    VkDeviceSize page_size = page_l > 0 ? (VkDeviceSize)page_l : 4096u;
    if (req.size > align_up((VkDeviceSize)size, page_size)) goto fail;
    VkMemoryHostPointerPropertiesEXT host_props = { .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
    if (g_vk.get_host_pointer_properties(g_vk.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                         (void *)ptr, &host_props) != VK_SUCCESS) goto fail;
    uint32_t type_index;
    VkMemoryPropertyFlags actual;
    uint32_t bits = req.memoryTypeBits & host_props.memoryTypeBits;
    if (!memory_type(bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &type_index, &actual)) goto fail;
    /* Non-coherent imported host allocations are deliberately rejected: the original
       allocation is mapped by the application, not vkMapMemory, and must stay coherent. */
    if ((actual & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) goto fail;
    VkImportMemoryHostPointerInfoEXT import_info = { .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
    import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import_info.pHostPointer = (void *)ptr;
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.pNext = &import_info; ai.allocationSize = req.size; ai.memoryTypeIndex = type_index;
    if (vkAllocateMemory(g_vk.device, &ai, NULL, &entry->gpu.memory) != VK_SUCCESS) goto fail;
    if (vkBindBufferMemory(g_vk.device, entry->gpu.buffer, entry->gpu.memory, 0) != VK_SUCCESS) goto fail;
    entry->host_ptr = ptr; entry->host_size = size;
    entry->gpu.size = buffer_size; entry->gpu.allocation_size = req.size;
    entry->gpu.memory_type = type_index; entry->gpu.coherent = 1;
    entry->next = g_vk.imports; g_vk.imports = entry;
    size_t hb = import_bucket(ptr);
    entry->hnext = g_vk.import_table[hb]; g_vk.import_table[hb] = entry;
    return entry;
fail:
    if (entry->gpu.buffer) vkDestroyBuffer(g_vk.device, entry->gpu.buffer, NULL);
    if (entry->gpu.memory) vkFreeMemory(g_vk.device, entry->gpu.memory, NULL);
    free(entry);
    return NULL;
}

static int ensure_scratch(HostBuffer *buffer, size_t *capacity, size_t need, int cpu_reads) {
    if (need <= *capacity && buffer->buffer) return 1;
    destroy_host_buffer(buffer);
    size_t new_capacity = (size_t)align_up((VkDeviceSize)(need ? need : 4), 4096);
    if (!create_host_buffer(new_capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, cpu_reads, buffer)) return 0;
    *capacity = new_capacity;
    return 1;
}

/* ------------------------------------------------------------------
 * Asynchronous batched dispatch.
 *   dispatch_batch_begin(): upload, record and SUBMIT without waiting. g_vk_mutex stays locked
 *   until dispatch_batch_end(), so the caller may do CPU work in between.
 *   rg[i] = number of LEADING rows of job i that the GPU computes (0 = job not on the GPU).
 * ------------------------------------------------------------------ */
static size_t vk_align_up(size_t v, size_t a) { return (v + a - 1u) & ~(a - 1u); }

typedef struct {
    int active, n;
    const KernelMatvecJob *jobs;
    size_t rg[VK_BATCH_MAX];
    size_t out_off[VK_BATCH_MAX];
} PendingBatch;
static PendingBatch g_pending;

static int dispatch_batch_begin(const KernelMatvecJob *jobs, int n, const size_t *rg) {
    if (n <= 0 || n > VK_BATCH_MAX || g_pending.active) return 0;
    size_t vec_off[VK_BATCH_MAX], out_off[VK_BATCH_MAX], mat_size[VK_BATCH_MAX], bpr[VK_BATCH_MAX];
    int owner[VK_BATCH_MAX];
    size_t vec_total = 0, out_total = 0;
    int any = 0;
    for (int i = 0; i < n; ++i) {
        const KernelMatvecJob *j = &jobs[i];
        if (!j->matrix || !j->vector || !j->output || !j->rows || !j->cols ||
            j->rows > UINT32_MAX || j->cols > UINT32_MAX || (j->cols & 255u) || rg[i] > j->rows) return 0;
        size_t bs = j->is_q6 ? 210u : 144u;
        bpr[i] = (j->cols + 255u) / 256u;
        if (bpr[i] > SIZE_MAX / bs || j->rows > SIZE_MAX / (bpr[i] * bs)) return 0;
        mat_size[i] = j->rows * bpr[i] * bs;
        owner[i] = -1;
        vec_off[i] = out_off[i] = 0;
        if (rg[i] == 0) continue;
        any = 1;
        owner[i] = i;
        for (int p = 0; p < i; ++p)
            if (owner[p] == p && jobs[p].vector == j->vector && jobs[p].cols == j->cols) { owner[i] = p; break; }
        if (owner[i] == i) { vec_off[i] = vec_total; vec_total += vk_align_up(j->cols * sizeof(float), VK_BATCH_ALIGN); }
        else vec_off[i] = vec_off[owner[i]];
        out_off[i] = out_total;
        out_total += vk_align_up(rg[i] * sizeof(float), VK_BATCH_ALIGN);
    }
    if (!any) return 0;
    pthread_mutex_lock(&g_vk_mutex);
    if (!initialize_locked()) { pthread_mutex_unlock(&g_vk_mutex); return 0; }
    ImportedBuffer *weights[VK_BATCH_MAX];
    VkDeviceSize w_off[VK_BATCH_MAX], w_range[VK_BATCH_MAX];
    for (int i = 0; i < n; ++i) {
        weights[i] = NULL;
        w_off[i] = 0; w_range[i] = 0;
        if (rg[i] == 0) continue;
        weights[i] = find_region(jobs[i].matrix, mat_size[i], &w_off[i], &w_range[i]);
        if (!weights[i]) {
            weights[i] = find_import(jobs[i].matrix, mat_size[i]);
            if (!weights[i]) weights[i] = import_host_buffer(jobs[i].matrix, mat_size[i]);
            if (weights[i]) { w_off[i] = 0; w_range[i] = weights[i]->gpu.size; }
        }
        if (!weights[i]) {
            if (!g_vk.logged_import_failure) {
                fprintf(stderr, "Roaster Vulkan: existing weight allocation cannot be imported as a shared GPU buffer; using CPU fallback for these kernels\n");
                g_vk.logged_import_failure = 1;
            }
            pthread_mutex_unlock(&g_vk_mutex);
            return 0;
        }
    }
    if (!ensure_scratch(&g_vk.vector_buffer, &g_vk.vector_capacity, vec_total, 0) ||
        !ensure_scratch(&g_vk.output_buffer, &g_vk.output_capacity, out_total, 1)) {
        pthread_mutex_unlock(&g_vk_mutex);
        return 0;
    }
    for (int i = 0; i < n; ++i)
        if (rg[i] && owner[i] == i)
            memcpy((char *)g_vk.vector_buffer.mapped + vec_off[i], jobs[i].vector, jobs[i].cols * sizeof(float));
    if (!flush_host_buffer(&g_vk.vector_buffer)) { pthread_mutex_unlock(&g_vk_mutex); return 0; }

    uint32_t max_x = g_vk.properties.limits.maxComputeWorkGroupCount[0];
    uint32_t max_y = g_vk.properties.limits.maxComputeWorkGroupCount[1];
    VkDescriptorBufferInfo infos[VK_BATCH_MAX * 3];
    VkWriteDescriptorSet writes[VK_BATCH_MAX * 3];
    memset(writes, 0, sizeof(writes));
    int set_of[VK_BATCH_MAX];
    int nsets = 0;
    for (int i = 0; i < n; ++i) {
        set_of[i] = -1;
        if (rg[i] == 0) continue;
        int k = nsets++;
        set_of[i] = k;
        infos[3 * k + 0] = (VkDescriptorBufferInfo){ weights[i]->gpu.buffer, w_off[i], w_range[i] };
        infos[3 * k + 1] = (VkDescriptorBufferInfo){ g_vk.vector_buffer.buffer, vec_off[i], jobs[i].cols * sizeof(float) };
        infos[3 * k + 2] = (VkDescriptorBufferInfo){ g_vk.output_buffer.buffer, out_off[i], rg[i] * sizeof(float) };
        for (uint32_t b = 0; b < 3; ++b) {
            VkWriteDescriptorSet *w = &writes[3 * k + b];
            w->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w->dstSet = g_vk.batch_sets[k];
            w->dstBinding = b;
            w->descriptorCount = 1;
            w->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w->pBufferInfo = &infos[3 * k + b];
        }
    }
    vkUpdateDescriptorSets(g_vk.device, (uint32_t)(3 * nsets), writes, 0, NULL);

    if (vkResetCommandBuffer(g_vk.command_buffer, 0) != VK_SUCCESS) { pthread_mutex_unlock(&g_vk_mutex); return 0; }
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(g_vk.command_buffer, &begin) != VK_SUCCESS) { pthread_mutex_unlock(&g_vk_mutex); return 0; }
    int bound = -1;
    for (int i = 0; i < n; ++i) {
        if (rg[i] == 0) continue;
        int q6 = jobs[i].is_q6 ? 1 : 0;
        if (bound != q6) {
            vkCmdBindPipeline(g_vk.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, q6 ? g_vk.q6_pipeline : g_vk.q4_pipeline);
            bound = q6;
        }
        /* Must match ROWS_PER_WG in shaders/q4k_matvec.comp (8) and q6k_matvec.comp (16). */
        const uint32_t rows_per_wg = q6 ? 16u : 8u;
        uint32_t total_groups = ((uint32_t)rg[i] + rows_per_wg - 1u) / rows_per_wg;
        uint32_t groups_x = total_groups < max_x ? total_groups : max_x;
        uint32_t groups_y = (total_groups + groups_x - 1u) / groups_x;
        if (groups_y > max_y) { vkEndCommandBuffer(g_vk.command_buffer); pthread_mutex_unlock(&g_vk_mutex); return 0; }
        vkCmdBindDescriptorSets(g_vk.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                g_vk.pipeline_layout, 0, 1, &g_vk.batch_sets[set_of[i]], 0, NULL);
        uint32_t params[3] = { (uint32_t)rg[i], (uint32_t)jobs[i].cols, (uint32_t)bpr[i] };
        vkCmdPushConstants(g_vk.command_buffer, g_vk.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), params);
        vkCmdDispatch(g_vk.command_buffer, groups_x, groups_y, 1);
    }
    if (vkEndCommandBuffer(g_vk.command_buffer) != VK_SUCCESS) { pthread_mutex_unlock(&g_vk_mutex); return 0; }
    vkResetFences(g_vk.device, 1, &g_vk.fence);
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1; submit.pCommandBuffers = &g_vk.command_buffer;
    if (vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) {
        pthread_mutex_unlock(&g_vk_mutex);
        return 0;
    }
    g_pending.active = 1;
    g_pending.n = n;
    g_pending.jobs = jobs;
    for (int i = 0; i < n; ++i) { g_pending.rg[i] = rg[i]; g_pending.out_off[i] = out_off[i]; }
    return 1;   /* g_vk_mutex is still held: released by dispatch_batch_end() */
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Wait for the GPU by polling the fence (lowest latency); after 3 ms fall back to a blocking wait. */
static int wait_fence(void)
{
    double t0 = now_sec();
    for (;;) {
        VkResult r = vkGetFenceStatus(g_vk.device, g_vk.fence);
        if (r == VK_SUCCESS) return 1;
        if (r != VK_NOT_READY) return 0;
        if (now_sec() - t0 > 0.003) break;
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }
    return vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
}

/* Wait for the batch started by dispatch_batch_begin() and copy results out.
 * Returns 1 on success. Always releases g_vk_mutex. */
static int dispatch_batch_end(void) {
    if (!g_pending.active) return 0;
    PendingBatch *p = &g_pending;
    p->active = 0;
    int ok = wait_fence() && invalidate_host_buffer(&g_vk.output_buffer);
    if (ok)
        for (int i = 0; i < p->n; ++i)
            if (p->rg[i])
                memcpy(p->jobs[i].output, (char *)g_vk.output_buffer.mapped + p->out_off[i], p->rg[i] * sizeof(float));
    pthread_mutex_unlock(&g_vk_mutex);
    return ok;
}

void kernel_vk_release_host_ptr(const void *ptr) {
    if (!ptr) return;
    pthread_mutex_lock(&g_vk_mutex);
    if (g_vk.device) {
        ImportedBuffer **link = &g_vk.imports;
        while (*link) {
            ImportedBuffer *it = *link;
            if (it->host_ptr == ptr && !it->is_driver) {
                /* batches hold g_vk_mutex until their fence signals: nothing is in flight here. */
                ImportedBuffer **hl = &g_vk.import_table[import_bucket(ptr)];
                while (*hl && *hl != it) hl = &(*hl)->hnext;
                if (*hl) *hl = it->hnext;
                if (it->is_region) {
                    ImportedBuffer **rl = &g_vk.regions;
                    while (*rl && *rl != it) rl = &(*rl)->rnext;
                    if (*rl) *rl = it->rnext;
                }
                *link = it->next;
                if (it->gpu.buffer) vkDestroyBuffer(g_vk.device, it->gpu.buffer, NULL);
                if (it->gpu.memory) vkFreeMemory(g_vk.device, it->gpu.memory, NULL);
                free(it);
                continue;
            }
            link = &it->next;
        }
    }
    pthread_mutex_unlock(&g_vk_mutex);
}

int kernel_vk_register_region(const void *ptr, size_t size) {
    int ok = 0;
    pthread_mutex_lock(&g_vk_mutex);
    if (initialize_locked()) {
        ImportedBuffer *e = find_import(ptr, size);
        if (!e) e = import_host_buffer(ptr, size);
        if (e && !e->is_region) { e->is_region = 1; e->rnext = g_vk.regions; g_vk.regions = e; }
        ok = e != NULL;
    }
    pthread_mutex_unlock(&g_vk_mutex);
    return ok;
}


/* ------------------------------------------------------------------
 * Driver-allocated weight memory.
 * Weights live in memory from vkAllocateMemory: mapped for the CPU (weights are read straight
 * into it) but laid out by the driver for the GPU, which reads it much faster than imported
 * malloc memory. Allocations are carved from big blocks (bump allocator, 4 KB aligned) and
 * released together by kernel_vk_shutdown(). At most 85% of any heap is used.
 * ------------------------------------------------------------------ */
#define DM_BLOCK_BYTES ((size_t)256 << 20)
static ImportedBuffer *g_dm_cur = NULL;
static size_t g_dm_cur_used = 0;
static size_t g_dm_heap_used[VK_MAX_MEMORY_HEAPS];

static ImportedBuffer *dm_new_block(size_t bytes)
{
    const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags HC = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags CA = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    /* First choice: uncached (write-combined) host-visible memory; then any host-visible memory. */
    const VkMemoryPropertyFlags forbidden[2] = { CA, 0 };
    int done = 0;
    uint32_t chosen_heap = 0;
    VkMemoryRequirements req;
    ImportedBuffer *e = calloc(1, sizeof(*e));
    if (!e) return NULL;

    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_vk.device, &bi, NULL, &e->gpu.buffer) != VK_SUCCESS) goto fail;
    vkGetBufferMemoryRequirements(g_vk.device, e->gpu.buffer, &req);
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    for (int p = 0; p < 2 && !done; ++p) {
        for (uint32_t t = 0; t < g_vk.memory_properties.memoryTypeCount && !done; ++t) {
            if (!(req.memoryTypeBits & (1u << t))) continue;
            VkMemoryPropertyFlags f = g_vk.memory_properties.memoryTypes[t].propertyFlags;
            if ((f & (HV | HC)) != (HV | HC) || (f & forbidden[p])) continue;
            uint32_t hi = g_vk.memory_properties.memoryTypes[t].heapIndex;
            size_t cap = (size_t)(g_vk.memory_properties.memoryHeaps[hi].size / 100u * 85u);
            if (g_dm_heap_used[hi] + (size_t)req.size > cap) continue;
            ai.memoryTypeIndex = t;
            if (vkAllocateMemory(g_vk.device, &ai, NULL, &e->gpu.memory) != VK_SUCCESS) {
                e->gpu.memory = VK_NULL_HANDLE;
                continue;
            }
            done = 1;
            chosen_heap = hi;
        }
    }
    if (!done) goto fail;
    if (vkBindBufferMemory(g_vk.device, e->gpu.buffer, e->gpu.memory, 0) != VK_SUCCESS) goto fail;
    if (vkMapMemory(g_vk.device, e->gpu.memory, 0, VK_WHOLE_SIZE, 0, &e->gpu.mapped) != VK_SUCCESS) goto fail;
    g_dm_heap_used[chosen_heap] += (size_t)req.size;
    e->host_ptr = e->gpu.mapped;
    e->host_size = (size_t)req.size;
    e->gpu.size = req.size;
    e->gpu.coherent = 1;
    e->is_driver = 1;
    e->is_region = 1;
    e->next = g_vk.imports;  g_vk.imports = e;
    e->rnext = g_vk.regions; g_vk.regions = e;
    return e;
fail:
    if (e->gpu.buffer) vkDestroyBuffer(g_vk.device, e->gpu.buffer, NULL);
    if (e->gpu.memory) vkFreeMemory(g_vk.device, e->gpu.memory, NULL);
    free(e);
    return NULL;
}

void *kernel_vk_alloc_weights(size_t size, size_t *granted)
{
    if (granted) *granted = 0;
    if (!size) return NULL;
    size_t need = (size + 4095u) & ~(size_t)4095u;
    void *ret = NULL;
    pthread_mutex_lock(&g_vk_mutex);
    if (initialize_locked()) {
        if (!g_dm_cur || g_dm_cur_used + need > g_dm_cur->host_size) {
            size_t bs = need > DM_BLOCK_BYTES ? need : DM_BLOCK_BYTES;
            ImportedBuffer *b = dm_new_block(bs);
            if (!b && bs > need) b = dm_new_block(need);
            if (b) { g_dm_cur = b; g_dm_cur_used = 0; }
        }
        if (g_dm_cur && g_dm_cur_used + need <= g_dm_cur->host_size) {
            ret = (char *)g_dm_cur->host_ptr + g_dm_cur_used;
            g_dm_cur_used += need;
            if (granted) *granted = need;
        }
    }
    pthread_mutex_unlock(&g_vk_mutex);
    return ret;
}

int kernel_vk_owns_weights(const void *ptr)
{
    int own = 0;
    pthread_mutex_lock(&g_vk_mutex);
    for (ImportedBuffer *it = g_vk.regions; it && !own; it = it->rnext) {
        uintptr_t b = (uintptr_t)it->host_ptr;
        if (it->is_driver && (uintptr_t)ptr >= b && (uintptr_t)ptr < b + it->host_size) own = 1;
    }
    pthread_mutex_unlock(&g_vk_mutex);
    return own;
}

/* Driver memory is released as whole blocks by kernel_vk_shutdown(); nothing to do per pointer. */
int kernel_vk_free_weights(void *ptr) { return kernel_vk_owns_weights(ptr); }

void kernel_vk_shutdown(void) {
    pthread_mutex_lock(&g_vk_mutex);
    shutdown_locked();
    g_dm_cur = NULL; g_dm_cur_used = 0;
    memset(g_dm_heap_used, 0, sizeof(g_dm_heap_used));
    pthread_mutex_unlock(&g_vk_mutex);
}

/* ------------------------------------------------------------------
 * GPU-only execution of quantized matvecs.
 * Every row of every job goes to the GPU, and all jobs of a call share ONE command buffer, ONE
 * submit and ONE fence wait. The CPU kernels run only if Vulkan is unavailable or a GPU step
 * fails (ROASTER_VULKAN=0 forces that). While the GPU works, the CPU is free: use
 * kernel_matvec_quant_begin()/end() to overlap independent CPU work with it.
 * ------------------------------------------------------------------ */
static void cpu_rows(const KernelMatvecJob *j, size_t r0, size_t r1)
{
    if (r1 <= r0) return;
    size_t bpr = (j->cols + 255u) / 256u;
    size_t bs = j->is_q6 ? 210u : 144u;
    const char *m = (const char *)j->matrix + r0 * bpr * bs;
    if (j->is_q6) matvec_q6_cpu(m, j->vector, j->output + r0, r1 - r0, j->cols);
    else          matvec_q4_cpu(m, j->vector, j->output + r0, r1 - r0, j->cols);
}

static void cpu_fallback(const KernelMatvecJob *jobs, int n)
{
    for (int i = 0; i < n; ++i) cpu_rows(&jobs[i], 0, jobs[i].rows);
}

static int gpu_submit(const KernelMatvecJob *jobs, int n)
{
    size_t rg[VK_BATCH_MAX];
    for (int i = 0; i < n; ++i) rg[i] = jobs[i].rows;
    return dispatch_batch_begin(jobs, n, rg);
}

static int g_async_open = 0;
static const KernelMatvecJob *g_async_jobs = NULL;
static int g_async_n = 0;

void kernel_matvec_quant_end(void)
{
    if (!g_async_open) return;
    g_async_open = 0;
    if (!dispatch_batch_end())
        cpu_fallback(g_async_jobs, g_async_n);
}

static void run_sync(const KernelMatvecJob *jobs, int n)
{
    kernel_matvec_quant_end();
    for (int done = 0; done < n; ) {
        int chunk = n - done > VK_BATCH_MAX ? VK_BATCH_MAX : n - done;
        const KernelMatvecJob *c = jobs + done;
        if (!gpu_submit(c, chunk)) {
            cpu_fallback(c, chunk);
        } else {
            if (!dispatch_batch_end()) cpu_fallback(c, chunk);
        }
        done += chunk;
    }
}

void kernel_matvec_quant_batch(const KernelMatvecJob *jobs, int n)
{
    if (n > 0) run_sync(jobs, n);
}

int kernel_matvec_quant_begin(const KernelMatvecJob *jobs, int n)
{
    if (n <= 0) return 0;
    kernel_matvec_quant_end();
    if (n > VK_BATCH_MAX) { run_sync(jobs, n); return 0; }
    if (!gpu_submit(jobs, n)) { cpu_fallback(jobs, n); return 0; }
    g_async_open = 1;
    g_async_jobs = jobs;
    g_async_n = n;
    return 1;
}

void kernel_matvec_q4_k(const void *matrix, const float *vector, float *output, size_t rows, size_t cols)
{
    KernelMatvecJob j = { 0, matrix, vector, output, rows, cols };
    kernel_matvec_quant_batch(&j, 1);
}

void kernel_matvec_q6_k(const void *matrix, const float *vector, float *output, size_t rows, size_t cols)
{
    KernelMatvecJob j = { 1, matrix, vector, output, rows, cols };
    kernel_matvec_quant_batch(&j, 1);
}


/* ------------------------------------------------------------------
 * Fused MoE FFN: gate/up matvecs -> SiLU*up -> down matvecs in ONE command buffer.
 * Scratch layout (all inside g_vk.output_buffer, offsets 256-byte aligned):
 *   [gate0 up0 gate1 up1 ...]  [eh0 eh1 ...]  [out0 out1 ...]
 * The down matvec reads its input from the eh region and writes into its own out slot, so the
 * CPU never sees the intermediate activations.
 * ------------------------------------------------------------------ */
static int vk_weight_info(const void *m, size_t bytes, VkDescriptorBufferInfo *out)
{
    VkDeviceSize off = 0, range = 0;
    ImportedBuffer *b = find_region(m, bytes, &off, &range);
    if (!b) {
        b = find_import(m, bytes);
        if (!b) b = import_host_buffer(m, bytes);
        if (!b) return 0;
        off = 0;
        range = b->gpu.size;
    }
    out->buffer = b->gpu.buffer;
    out->offset = off;
    out->range = range;
    return 1;
}

static void fill_set(VkWriteDescriptorSet *w, VkDescriptorBufferInfo *inf, int k,
                     VkDescriptorBufferInfo a, VkDescriptorBufferInfo b, VkDescriptorBufferInfo c)
{
    inf[3 * k + 0] = a;
    inf[3 * k + 1] = b;
    inf[3 * k + 2] = c;
    for (uint32_t i = 0; i < 3; ++i) {
        VkWriteDescriptorSet *x = &w[3 * k + i];
        memset(x, 0, sizeof(*x));
        x->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        x->dstSet = g_vk.batch_sets[k];
        x->dstBinding = i;
        x->descriptorCount = 1;
        x->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x->pBufferInfo = &inf[3 * k + i];
    }
}

#define FFN_MAX_EXPERTS 9      /* 3 * 9 + 1 = 28 descriptor sets <= VK_BATCH_MAX */

int kernel_ffn_fused(const KernelFfnExpert *ex, int n, const float *x, size_t inter, size_t hidden)
{
    if (!ex || !x || n <= 0 || n > FFN_MAX_EXPERTS) return 0;
    if (!inter || !hidden || (inter & 255u) || (hidden & 255u)) return 0;
    kernel_matvec_quant_end();                    /* never overlap with an open split-phase batch */

    const size_t gu_stride = vk_align_up(inter * sizeof(float), VK_BATCH_ALIGN);
    const size_t d_stride = vk_align_up(hidden * sizeof(float), VK_BATCH_ALIGN);
    const size_t eh_base = (size_t)n * 2u * gu_stride;
    const size_t d_base = eh_base + (size_t)n * gu_stride;
    const size_t out_total = d_base + (size_t)n * d_stride;
    const size_t bpr_gu = hidden / 256u;          /* gate/up: cols = hidden */
    const size_t bpr_dn = inter / 256u;           /* down:    cols = inter  */
    VkDescriptorBufferInfo winfo[FFN_MAX_EXPERTS * 3];
    VkDescriptorBufferInfo infos[(3 * FFN_MAX_EXPERTS + 1) * 3];
    VkWriteDescriptorSet writes[(3 * FFN_MAX_EXPERTS + 1) * 3];
    int result = 0;

    pthread_mutex_lock(&g_vk_mutex);
    if (!initialize_locked() || !g_vk.silu_pipeline || g_pending.active) {
        pthread_mutex_unlock(&g_vk_mutex);
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        size_t bs_g = ex[i].gate_q6 ? 210u : 144u, bs_u = ex[i].up_q6 ? 210u : 144u, bs_d = ex[i].down_q6 ? 210u : 144u;
        size_t sg = inter * bpr_gu * bs_g, su = inter * bpr_gu * bs_u, sd = hidden * bpr_dn * bs_d;
        if (!ex[i].gate || !ex[i].up || !ex[i].down || !ex[i].out ||
            !vk_weight_info(ex[i].gate, sg, &winfo[3 * i + 0]) ||
            !vk_weight_info(ex[i].up,   su, &winfo[3 * i + 1]) ||
            !vk_weight_info(ex[i].down, sd, &winfo[3 * i + 2])) {
            pthread_mutex_unlock(&g_vk_mutex);
            return 0;
        }
    }
    if (!ensure_scratch(&g_vk.vector_buffer, &g_vk.vector_capacity, hidden * sizeof(float), 0) ||
        !ensure_scratch(&g_vk.output_buffer, &g_vk.output_capacity, out_total, 1)) {
        pthread_mutex_unlock(&g_vk_mutex);
        return 0;
    }
    memcpy(g_vk.vector_buffer.mapped, x, hidden * sizeof(float));
    if (!flush_host_buffer(&g_vk.vector_buffer)) { pthread_mutex_unlock(&g_vk_mutex); return 0; }

    VkBuffer ob = g_vk.output_buffer.buffer;
    VkDescriptorBufferInfo vecinfo = { g_vk.vector_buffer.buffer, 0, hidden * sizeof(float) };
    int nsets = 0;
    for (int i = 0; i < n; ++i) {                                       /* stage 1: gate, up */
        fill_set(writes, infos, nsets++, winfo[3 * i + 0], vecinfo,
                 (VkDescriptorBufferInfo){ ob, (VkDeviceSize)(2u * i) * gu_stride, inter * sizeof(float) });
        fill_set(writes, infos, nsets++, winfo[3 * i + 1], vecinfo,
                 (VkDescriptorBufferInfo){ ob, (VkDeviceSize)(2u * i + 1u) * gu_stride, inter * sizeof(float) });
    }
    const int silu_set = nsets;                                          /* stage 2: silu*up */
    fill_set(writes, infos, nsets++,
             (VkDescriptorBufferInfo){ ob, 0, eh_base },
             (VkDescriptorBufferInfo){ ob, 0, eh_base },
             (VkDescriptorBufferInfo){ ob, eh_base, (VkDeviceSize)n * gu_stride });
    for (int i = 0; i < n; ++i)                                          /* stage 3: down */
        fill_set(writes, infos, nsets++, winfo[3 * i + 2],
                 (VkDescriptorBufferInfo){ ob, eh_base + (VkDeviceSize)i * gu_stride, inter * sizeof(float) },
                 (VkDescriptorBufferInfo){ ob, d_base + (VkDeviceSize)i * d_stride, hidden * sizeof(float) });
    vkUpdateDescriptorSets(g_vk.device, (uint32_t)(3 * nsets), writes, 0, NULL);

    if (vkResetCommandBuffer(g_vk.command_buffer, 0) != VK_SUCCESS) { pthread_mutex_unlock(&g_vk_mutex); return 0; }
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(g_vk.command_buffer, &begin) != VK_SUCCESS) { pthread_mutex_unlock(&g_vk_mutex); return 0; }
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    VkCommandBuffer cb = g_vk.command_buffer;
    int bound = -1;
    for (int pass = 0; pass < 2; ++pass) {          /* pass 0: gate/up (sets 0..2n-1), pass 1: down */
        for (int j = 0; j < (pass == 0 ? 2 * n : n); ++j) {
            int q6, set;
            uint32_t rows, cols, bpr;
            if (pass == 0) {
                const KernelFfnExpert *e = &ex[j >> 1];
                q6 = (j & 1) ? e->up_q6 : e->gate_q6;
                set = j; rows = (uint32_t)inter; cols = (uint32_t)hidden; bpr = (uint32_t)bpr_gu;
            } else {
                q6 = ex[j].down_q6;
                set = 2 * n + 1 + j; rows = (uint32_t)hidden; cols = (uint32_t)inter; bpr = (uint32_t)bpr_dn;
            }
            q6 = q6 ? 1 : 0;
            if (bound != q6) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, q6 ? g_vk.q6_pipeline : g_vk.q4_pipeline);
                bound = q6;
            }
            const uint32_t rpw = q6 ? 16u : 8u;     /* must match ROWS_PER_WG in the shaders */
            uint32_t groups = (rows + rpw - 1u) / rpw;
            uint32_t params[3] = { rows, cols, bpr };
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipeline_layout, 0, 1,
                                    &g_vk.batch_sets[set], 0, NULL);
            vkCmdPushConstants(cb, g_vk.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), params);
            vkCmdDispatch(cb, groups, 1, 1);
        }
        if (pass == 0) {
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mb, 0, NULL, 0, NULL);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.silu_pipeline);
            bound = -1;
            uint32_t sp[3] = { (uint32_t)inter, (uint32_t)n, (uint32_t)(gu_stride / sizeof(float)) };
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.pipeline_layout, 0, 1,
                                    &g_vk.batch_sets[silu_set], 0, NULL);
            vkCmdPushConstants(cb, g_vk.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), sp);
            vkCmdDispatch(cb, (uint32_t)(((size_t)n * inter + 63u) / 64u), 1, 1);
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mb, 0, NULL, 0, NULL);
        }
    }
    if (vkEndCommandBuffer(cb) != VK_SUCCESS) { pthread_mutex_unlock(&g_vk_mutex); return 0; }
    vkResetFences(g_vk.device, 1, &g_vk.fence);
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    if (vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) { pthread_mutex_unlock(&g_vk_mutex); return 0; }
    int ok = wait_fence() && invalidate_host_buffer(&g_vk.output_buffer);
    if (ok) {
        for (int i = 0; i < n; ++i)
            memcpy(ex[i].out, (const char *)g_vk.output_buffer.mapped + d_base + (size_t)i * d_stride,
                   hidden * sizeof(float));
        result = 1;
    }
    pthread_mutex_unlock(&g_vk_mutex);
    return result;
}

/* BEGIN EMBEDDED SPIRV */
/* AUTO-GENERATED by shaders/embed_spv.py from q4k_matvec.comp / q6k_matvec.comp / silu_mul.comp. Do not edit. */
const uint32_t roaster_spv_q4k_v2[] = {
    0x07230203, 0x00010300, 0x0008000b, 0x000001d9, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00000064, 0x0000006a, 0x00000070,
    0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002,
    0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060005, 0x0000000d, 0x74796273,
    0x75762865, 0x31753b33, 0x0000003b, 0x00030005, 0x0000000b, 0x00000073, 0x00030005, 0x0000000c,
    0x00000069, 0x00080005, 0x00000016, 0x6c616373, 0x6e694d65, 0x33757628, 0x3b31753b, 0x663b3166,
    0x00003b31, 0x00030005, 0x00000012, 0x00000073, 0x00030005, 0x00000013, 0x0000006a, 0x00030005,
    0x00000014, 0x00006373, 0x00030005, 0x00000015, 0x00006e6d, 0x00030005, 0x0000002e, 0x00000061,
    0x00040005, 0x0000002f, 0x61726170, 0x0000006d, 0x00040005, 0x00000031, 0x61726170, 0x0000006d,
    0x00030005, 0x00000036, 0x00000062, 0x00040005, 0x00000039, 0x61726170, 0x0000006d, 0x00040005,
    0x0000003b, 0x61726170, 0x0000006d, 0x00030005, 0x0000003f, 0x00000076, 0x00040005, 0x00000042,
    0x61726170, 0x0000006d, 0x00040005, 0x00000044, 0x61726170, 0x0000006d, 0x00040005, 0x0000004b,
    0x61726170, 0x0000006d, 0x00040005, 0x0000004d, 0x61726170, 0x0000006d, 0x00040005, 0x00000056,
    0x61726170, 0x0000006d, 0x00040005, 0x00000058, 0x61726170, 0x0000006d, 0x00030005, 0x00000062,
    0x00646974, 0x00080005, 0x00000064, 0x4c5f6c67, 0x6c61636f, 0x6f766e49, 0x69746163, 0x44496e6f,
    0x00000000, 0x00030005, 0x00000069, 0x00006777, 0x00060005, 0x0000006a, 0x575f6c67, 0x476b726f,
    0x70756f72, 0x00004449, 0x00070005, 0x00000070, 0x4e5f6c67, 0x6f576d75, 0x72476b72, 0x7370756f,
    0x00000000, 0x00030005, 0x00000075, 0x00776f72, 0x00030005, 0x0000007c, 0x00627573, 0x00030005,
    0x00000080, 0x00000067, 0x00030005, 0x00000084, 0x00000068, 0x00040005, 0x00000088, 0x696c6176,
    0x00000064, 0x00040005, 0x0000008a, 0x61726150, 0x0000736d, 0x00050006, 0x0000008a, 0x00000000,
    0x73776f72, 0x00000000, 0x00050006, 0x0000008a, 0x00000001, 0x736c6f63, 0x00000000, 0x00070006,
    0x0000008a, 0x00000002, 0x636f6c62, 0x6550736b, 0x776f5272, 0x00000000, 0x00030005, 0x0000008c,
    0x00000070, 0x00030005, 0x00000092, 0x0000626e, 0x00040005, 0x00000095, 0x55776f72, 0x0074696e,
    0x00030005, 0x0000009b, 0x006d7573, 0x00030005, 0x0000009d, 0x00006274, 0x00040005, 0x000000a6,
    0x6b6c626e, 0x00000000, 0x00030005, 0x000000ad, 0x00007674, 0x00040005, 0x000000b1, 0x65736162,
    0x00000034, 0x00030005, 0x000000b4, 0x00000069, 0x00030005, 0x000000c2, 0x00007378, 0x00060005,
    0x000000c5, 0x74636556, 0x7542726f, 0x72656666, 0x00000000, 0x00050006, 0x000000c5, 0x00000000,
    0x61746164, 0x00000000, 0x00050005, 0x000000c7, 0x75706e69, 0x63655674, 0x00726f74, 0x00030005,
    0x000000d5, 0x0000626c, 0x00040005, 0x000000de, 0x74696e75, 0x00000000, 0x00030005, 0x000000e7,
    0x00726468, 0x00060005, 0x000000e9, 0x67696557, 0x75427468, 0x72656666, 0x00000000, 0x00050006,
    0x000000e9, 0x00000000, 0x61746164, 0x00000000, 0x00040005, 0x000000eb, 0x67696577, 0x00737468,
    0x00030005, 0x000000f0, 0x00000071, 0x00030005, 0x000000f9, 0x00006464, 0x00030005, 0x000000fd,
    0x00000073, 0x00040005, 0x00000102, 0x6f4c6373, 0x00000000, 0x00040005, 0x00000103, 0x6f4c6e6d,
    0x00000000, 0x00040005, 0x00000104, 0x61726170, 0x0000006d, 0x00040005, 0x00000106, 0x61726170,
    0x0000006d, 0x00040005, 0x00000107, 0x61726170, 0x0000006d, 0x00040005, 0x00000108, 0x61726170,
    0x0000006d, 0x00040005, 0x0000010f, 0x69486373, 0x00000000, 0x00040005, 0x00000110, 0x69486e6d,
    0x00000000, 0x00040005, 0x00000111, 0x61726170, 0x0000006d, 0x00040005, 0x00000113, 0x61726170,
    0x0000006d, 0x00040005, 0x00000114, 0x61726170, 0x0000006d, 0x00040005, 0x00000115, 0x61726170,
    0x0000006d, 0x00030005, 0x00000119, 0x00006978, 0x00030005, 0x00000123, 0x006f6c71, 0x00030005,
    0x00000124, 0x00696871, 0x00030005, 0x00000125, 0x006f6c78, 0x00030005, 0x00000126, 0x00696878,
    0x00030005, 0x00000127, 0x00000077, 0x00030005, 0x0000012f, 0x00000076, 0x00030005, 0x00000134,
    0x00006f6c, 0x00030005, 0x00000148, 0x00006968, 0x00030005, 0x0000015d, 0x00006c78, 0x00030005,
    0x00000163, 0x00006878, 0x00040005, 0x000001b3, 0x74726170, 0x006c6169, 0x00030005, 0x000001be,
    0x00000074, 0x00030005, 0x000001bf, 0x00000069, 0x00060005, 0x000001d1, 0x7074754f, 0x75427475,
    0x72656666, 0x00000000, 0x00050006, 0x000001d1, 0x00000000, 0x61746164, 0x00000000, 0x00060005,
    0x000001d3, 0x7074756f, 0x65567475, 0x726f7463, 0x00000000, 0x00040047, 0x00000064, 0x0000000b,
    0x0000001b, 0x00040047, 0x0000006a, 0x0000000b, 0x0000001a, 0x00040047, 0x00000070, 0x0000000b,
    0x00000018, 0x00030047, 0x0000008a, 0x00000002, 0x00050048, 0x0000008a, 0x00000000, 0x00000023,
    0x00000000, 0x00050048, 0x0000008a, 0x00000001, 0x00000023, 0x00000004, 0x00050048, 0x0000008a,
    0x00000002, 0x00000023, 0x00000008, 0x00040047, 0x000000c4, 0x00000006, 0x00000010, 0x00030047,
    0x000000c5, 0x00000002, 0x00040048, 0x000000c5, 0x00000000, 0x00000018, 0x00050048, 0x000000c5,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x000000c7, 0x00000018, 0x00040047, 0x000000c7,
    0x00000021, 0x00000001, 0x00040047, 0x000000c7, 0x00000022, 0x00000000, 0x00040047, 0x000000e8,
    0x00000006, 0x00000010, 0x00030047, 0x000000e9, 0x00000002, 0x00040048, 0x000000e9, 0x00000000,
    0x00000018, 0x00050048, 0x000000e9, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x000000eb,
    0x00000018, 0x00040047, 0x000000eb, 0x00000021, 0x00000000, 0x00040047, 0x000000eb, 0x00000022,
    0x00000000, 0x00040047, 0x000001d0, 0x00000006, 0x00000004, 0x00030047, 0x000001d1, 0x00000002,
    0x00040048, 0x000001d1, 0x00000000, 0x00000019, 0x00050048, 0x000001d1, 0x00000000, 0x00000023,
    0x00000000, 0x00030047, 0x000001d3, 0x00000019, 0x00040047, 0x000001d3, 0x00000021, 0x00000002,
    0x00040047, 0x000001d3, 0x00000022, 0x00000000, 0x00040047, 0x000001d8, 0x0000000b, 0x00000019,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
    0x00000000, 0x00040017, 0x00000007, 0x00000006, 0x00000003, 0x00040020, 0x00000008, 0x00000007,
    0x00000007, 0x00040020, 0x00000009, 0x00000007, 0x00000006, 0x00050021, 0x0000000a, 0x00000006,
    0x00000008, 0x00000009, 0x00030016, 0x0000000f, 0x00000020, 0x00040020, 0x00000010, 0x00000007,
    0x0000000f, 0x00070021, 0x00000011, 0x00000002, 0x00000008, 0x00000009, 0x00000010, 0x00000010,
    0x00040015, 0x00000019, 0x00000020, 0x00000001, 0x0004002b, 0x00000019, 0x0000001a, 0x00000002,
    0x0004002b, 0x00000006, 0x0000001f, 0x00000003, 0x0004002b, 0x00000006, 0x00000021, 0x00000008,
    0x0004002b, 0x00000006, 0x00000024, 0x000000ff, 0x0004002b, 0x00000006, 0x00000029, 0x00000004,
    0x00020014, 0x0000002a, 0x0004002b, 0x00000006, 0x00000034, 0x0000003f, 0x0004002b, 0x00000006,
    0x00000047, 0x0000000f, 0x0004002b, 0x00000019, 0x0000004f, 0x00000006, 0x0004002b, 0x00000019,
    0x00000051, 0x00000004, 0x00040020, 0x00000063, 0x00000001, 0x00000007, 0x0004003b, 0x00000063,
    0x00000064, 0x00000001, 0x0004002b, 0x00000006, 0x00000065, 0x00000000, 0x00040020, 0x00000066,
    0x00000001, 0x00000006, 0x0004003b, 0x00000063, 0x0000006a, 0x00000001, 0x0004002b, 0x00000006,
    0x0000006d, 0x00000001, 0x0004003b, 0x00000063, 0x00000070, 0x00000001, 0x0004002b, 0x00000019,
    0x00000079, 0x00000003, 0x0004002b, 0x00000006, 0x0000007e, 0x00000007, 0x0004002b, 0x00000019,
    0x00000082, 0x00000001, 0x00040020, 0x00000087, 0x00000007, 0x0000002a, 0x0005001e, 0x0000008a,
    0x00000006, 0x00000006, 0x00000006, 0x00040020, 0x0000008b, 0x00000009, 0x0000008a, 0x0004003b,
    0x0000008b, 0x0000008c, 0x00000009, 0x0004002b, 0x00000019, 0x0000008d, 0x00000000, 0x00040020,
    0x0000008e, 0x00000009, 0x00000006, 0x0004002b, 0x00000006, 0x00000099, 0x00000009, 0x0004002b,
    0x0000000f, 0x0000009c, 0x00000000, 0x0004002b, 0x00000006, 0x000000ab, 0x00000002, 0x0004002b,
    0x00000006, 0x000000ac, 0x00000108, 0x0004002b, 0x00000006, 0x000000af, 0x00000040, 0x00040017,
    0x000000be, 0x0000000f, 0x00000004, 0x0004002b, 0x00000006, 0x000000bf, 0x00000200, 0x0004001c,
    0x000000c0, 0x000000be, 0x000000bf, 0x00040020, 0x000000c1, 0x00000004, 0x000000c0, 0x0004003b,
    0x000000c1, 0x000000c2, 0x00000004, 0x0003001d, 0x000000c4, 0x000000be, 0x0003001e, 0x000000c5,
    0x000000c4, 0x00040020, 0x000000c6, 0x0000000c, 0x000000c5, 0x0004003b, 0x000000c6, 0x000000c7,
    0x0000000c, 0x00040020, 0x000000cb, 0x0000000c, 0x000000be, 0x00040020, 0x000000ce, 0x00000004,
    0x000000be, 0x00040017, 0x000000e5, 0x00000006, 0x00000004, 0x00040020, 0x000000e6, 0x00000007,
    0x000000e5, 0x0003001d, 0x000000e8, 0x000000e5, 0x0003001e, 0x000000e9, 0x000000e8, 0x00040020,
    0x000000ea, 0x0000000c, 0x000000e9, 0x0004003b, 0x000000ea, 0x000000eb, 0x0000000c, 0x00040020,
    0x000000ed, 0x0000000c, 0x000000e5, 0x00040017, 0x000000f7, 0x0000000f, 0x00000002, 0x00040020,
    0x000000f8, 0x00000007, 0x000000f7, 0x0004002b, 0x00000006, 0x0000011d, 0x00000010, 0x00040020,
    0x00000133, 0x00000007, 0x000000be, 0x0004002b, 0x00000019, 0x00000139, 0x00000008, 0x0004002b,
    0x00000019, 0x0000013e, 0x00000010, 0x0004002b, 0x00000019, 0x00000143, 0x00000018, 0x0004002b,
    0x00000019, 0x0000014e, 0x0000000c, 0x0004002b, 0x00000019, 0x00000153, 0x00000014, 0x0004002b,
    0x00000019, 0x00000158, 0x0000001c, 0x0004001c, 0x000001b1, 0x0000000f, 0x000000af, 0x00040020,
    0x000001b2, 0x00000004, 0x000001b1, 0x0004003b, 0x000001b2, 0x000001b3, 0x00000004, 0x00040020,
    0x000001b6, 0x00000004, 0x0000000f, 0x0003001d, 0x000001d0, 0x0000000f, 0x0003001e, 0x000001d1,
    0x000001d0, 0x00040020, 0x000001d2, 0x0000000c, 0x000001d1, 0x0004003b, 0x000001d2, 0x000001d3,
    0x0000000c, 0x00040020, 0x000001d6, 0x0000000c, 0x0000000f, 0x0006002c, 0x00000007, 0x000001d8,
    0x000000af, 0x0000006d, 0x0000006d, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x0004003b, 0x00000009, 0x00000062, 0x00000007, 0x0004003b, 0x00000009,
    0x00000069, 0x00000007, 0x0004003b, 0x00000009, 0x00000075, 0x00000007, 0x0004003b, 0x00000009,
    0x0000007c, 0x00000007, 0x0004003b, 0x00000009, 0x00000080, 0x00000007, 0x0004003b, 0x00000009,
    0x00000084, 0x00000007, 0x0004003b, 0x00000087, 0x00000088, 0x00000007, 0x0004003b, 0x00000009,
    0x00000092, 0x00000007, 0x0004003b, 0x00000009, 0x00000095, 0x00000007, 0x0004003b, 0x00000010,
    0x0000009b, 0x00000007, 0x0004003b, 0x00000009, 0x0000009d, 0x00000007, 0x0004003b, 0x00000009,
    0x000000a6, 0x00000007, 0x0004003b, 0x00000009, 0x000000ad, 0x00000007, 0x0004003b, 0x00000009,
    0x000000b1, 0x00000007, 0x0004003b, 0x00000009, 0x000000b4, 0x00000007, 0x0004003b, 0x00000009,
    0x000000d5, 0x00000007, 0x0004003b, 0x00000009, 0x000000de, 0x00000007, 0x0004003b, 0x000000e6,
    0x000000e7, 0x00000007, 0x0004003b, 0x000000e6, 0x000000f0, 0x00000007, 0x0004003b, 0x000000f8,
    0x000000f9, 0x00000007, 0x0004003b, 0x00000008, 0x000000fd, 0x00000007, 0x0004003b, 0x00000010,
    0x00000102, 0x00000007, 0x0004003b, 0x00000010, 0x00000103, 0x00000007, 0x0004003b, 0x00000008,
    0x00000104, 0x00000007, 0x0004003b, 0x00000009, 0x00000106, 0x00000007, 0x0004003b, 0x00000010,
    0x00000107, 0x00000007, 0x0004003b, 0x00000010, 0x00000108, 0x00000007, 0x0004003b, 0x00000010,
    0x0000010f, 0x00000007, 0x0004003b, 0x00000010, 0x00000110, 0x00000007, 0x0004003b, 0x00000008,
    0x00000111, 0x00000007, 0x0004003b, 0x00000009, 0x00000113, 0x00000007, 0x0004003b, 0x00000010,
    0x00000114, 0x00000007, 0x0004003b, 0x00000010, 0x00000115, 0x00000007, 0x0004003b, 0x00000009,
    0x00000119, 0x00000007, 0x0004003b, 0x00000010, 0x00000123, 0x00000007, 0x0004003b, 0x00000010,
    0x00000124, 0x00000007, 0x0004003b, 0x00000010, 0x00000125, 0x00000007, 0x0004003b, 0x00000010,
    0x00000126, 0x00000007, 0x0004003b, 0x00000009, 0x00000127, 0x00000007, 0x0004003b, 0x00000009,
    0x0000012f, 0x00000007, 0x0004003b, 0x00000133, 0x00000134, 0x00000007, 0x0004003b, 0x00000133,
    0x00000148, 0x00000007, 0x0004003b, 0x00000133, 0x0000015d, 0x00000007, 0x0004003b, 0x00000133,
    0x00000163, 0x00000007, 0x0004003b, 0x00000010, 0x000001be, 0x00000007, 0x0004003b, 0x00000009,
    0x000001bf, 0x00000007, 0x00050041, 0x00000066, 0x00000067, 0x00000064, 0x00000065, 0x0004003d,
    0x00000006, 0x00000068, 0x00000067, 0x0003003e, 0x00000062, 0x00000068, 0x00050041, 0x00000066,
    0x0000006b, 0x0000006a, 0x00000065, 0x0004003d, 0x00000006, 0x0000006c, 0x0000006b, 0x00050041,
    0x00000066, 0x0000006e, 0x0000006a, 0x0000006d, 0x0004003d, 0x00000006, 0x0000006f, 0x0000006e,
    0x00050041, 0x00000066, 0x00000071, 0x00000070, 0x00000065, 0x0004003d, 0x00000006, 0x00000072,
    0x00000071, 0x00050084, 0x00000006, 0x00000073, 0x0000006f, 0x00000072, 0x00050080, 0x00000006,
    0x00000074, 0x0000006c, 0x00000073, 0x0003003e, 0x00000069, 0x00000074, 0x0004003d, 0x00000006,
    0x00000076, 0x00000069, 0x00050084, 0x00000006, 0x00000077, 0x00000076, 0x00000021, 0x0004003d,
    0x00000006, 0x00000078, 0x00000062, 0x000500c2, 0x00000006, 0x0000007a, 0x00000078, 0x00000079,
    0x00050080, 0x00000006, 0x0000007b, 0x00000077, 0x0000007a, 0x0003003e, 0x00000075, 0x0000007b,
    0x0004003d, 0x00000006, 0x0000007d, 0x00000062, 0x000500c7, 0x00000006, 0x0000007f, 0x0000007d,
    0x0000007e, 0x0003003e, 0x0000007c, 0x0000007f, 0x0004003d, 0x00000006, 0x00000081, 0x0000007c,
    0x000500c2, 0x00000006, 0x00000083, 0x00000081, 0x00000082, 0x0003003e, 0x00000080, 0x00000083,
    0x0004003d, 0x00000006, 0x00000085, 0x0000007c, 0x000500c7, 0x00000006, 0x00000086, 0x00000085,
    0x0000006d, 0x0003003e, 0x00000084, 0x00000086, 0x0004003d, 0x00000006, 0x00000089, 0x00000075,
    0x00050041, 0x0000008e, 0x0000008f, 0x0000008c, 0x0000008d, 0x0004003d, 0x00000006, 0x00000090,
    0x0000008f, 0x000500b0, 0x0000002a, 0x00000091, 0x00000089, 0x00000090, 0x0003003e, 0x00000088,
    0x00000091, 0x00050041, 0x0000008e, 0x00000093, 0x0000008c, 0x0000001a, 0x0004003d, 0x00000006,
    0x00000094, 0x00000093, 0x0003003e, 0x00000092, 0x00000094, 0x0004003d, 0x00000006, 0x00000096,
    0x00000075, 0x0004003d, 0x00000006, 0x00000097, 0x00000092, 0x00050084, 0x00000006, 0x00000098,
    0x00000096, 0x00000097, 0x00050084, 0x00000006, 0x0000009a, 0x00000098, 0x00000099, 0x0003003e,
    0x00000095, 0x0000009a, 0x0003003e, 0x0000009b, 0x0000009c, 0x0003003e, 0x0000009d, 0x00000065,
    0x000200f9, 0x0000009e, 0x000200f8, 0x0000009e, 0x000400f6, 0x000000a0, 0x000000a1, 0x00000000,
    0x000200f9, 0x000000a2, 0x000200f8, 0x000000a2, 0x0004003d, 0x00000006, 0x000000a3, 0x0000009d,
    0x0004003d, 0x00000006, 0x000000a4, 0x00000092, 0x000500b0, 0x0000002a, 0x000000a5, 0x000000a3,
    0x000000a4, 0x000400fa, 0x000000a5, 0x0000009f, 0x000000a0, 0x000200f8, 0x0000009f, 0x0004003d,
    0x00000006, 0x000000a7, 0x00000092, 0x0004003d, 0x00000006, 0x000000a8, 0x0000009d, 0x00050082,
    0x00000006, 0x000000a9, 0x000000a7, 0x000000a8, 0x0007000c, 0x00000006, 0x000000aa, 0x00000001,
    0x00000026, 0x00000021, 0x000000a9, 0x0003003e, 0x000000a6, 0x000000aa, 0x000400e0, 0x000000ab,
    0x000000ab, 0x000000ac, 0x0004003d, 0x00000006, 0x000000ae, 0x000000a6, 0x00050084, 0x00000006,
    0x000000b0, 0x000000ae, 0x000000af, 0x0003003e, 0x000000ad, 0x000000b0, 0x0004003d, 0x00000006,
    0x000000b2, 0x0000009d, 0x00050084, 0x00000006, 0x000000b3, 0x000000b2, 0x000000af, 0x0003003e,
    0x000000b1, 0x000000b3, 0x0004003d, 0x00000006, 0x000000b5, 0x00000062, 0x0003003e, 0x000000b4,
    0x000000b5, 0x000200f9, 0x000000b6, 0x000200f8, 0x000000b6, 0x000400f6, 0x000000b8, 0x000000b9,
    0x00000000, 0x000200f9, 0x000000ba, 0x000200f8, 0x000000ba, 0x0004003d, 0x00000006, 0x000000bb,
    0x000000b4, 0x0004003d, 0x00000006, 0x000000bc, 0x000000ad, 0x000500b0, 0x0000002a, 0x000000bd,
    0x000000bb, 0x000000bc, 0x000400fa, 0x000000bd, 0x000000b7, 0x000000b8, 0x000200f8, 0x000000b7,
    0x0004003d, 0x00000006, 0x000000c3, 0x000000b4, 0x0004003d, 0x00000006, 0x000000c8, 0x000000b1,
    0x0004003d, 0x00000006, 0x000000c9, 0x000000b4, 0x00050080, 0x00000006, 0x000000ca, 0x000000c8,
    0x000000c9, 0x00060041, 0x000000cb, 0x000000cc, 0x000000c7, 0x0000008d, 0x000000ca, 0x0004003d,
    0x000000be, 0x000000cd, 0x000000cc, 0x00050041, 0x000000ce, 0x000000cf, 0x000000c2, 0x000000c3,
    0x0003003e, 0x000000cf, 0x000000cd, 0x000200f9, 0x000000b9, 0x000200f8, 0x000000b9, 0x0004003d,
    0x00000006, 0x000000d0, 0x000000b4, 0x00050080, 0x00000006, 0x000000d1, 0x000000d0, 0x000000af,
    0x0003003e, 0x000000b4, 0x000000d1, 0x000200f9, 0x000000b6, 0x000200f8, 0x000000b8, 0x000400e0,
    0x000000ab, 0x000000ab, 0x000000ac, 0x0004003d, 0x0000002a, 0x000000d2, 0x00000088, 0x000300f7,
    0x000000d4, 0x00000000, 0x000400fa, 0x000000d2, 0x000000d3, 0x000000d4, 0x000200f8, 0x000000d3,
    0x0003003e, 0x000000d5, 0x00000065, 0x000200f9, 0x000000d6, 0x000200f8, 0x000000d6, 0x000400f6,
    0x000000d8, 0x000000d9, 0x00000000, 0x000200f9, 0x000000da, 0x000200f8, 0x000000da, 0x0004003d,
    0x00000006, 0x000000db, 0x000000d5, 0x0004003d, 0x00000006, 0x000000dc, 0x000000a6, 0x000500b0,
    0x0000002a, 0x000000dd, 0x000000db, 0x000000dc, 0x000400fa, 0x000000dd, 0x000000d7, 0x000000d8,
    0x000200f8, 0x000000d7, 0x0004003d, 0x00000006, 0x000000df, 0x00000095, 0x0004003d, 0x00000006,
    0x000000e0, 0x0000009d, 0x0004003d, 0x00000006, 0x000000e1, 0x000000d5, 0x00050080, 0x00000006,
    0x000000e2, 0x000000e0, 0x000000e1, 0x00050084, 0x00000006, 0x000000e3, 0x000000e2, 0x00000099,
    0x00050080, 0x00000006, 0x000000e4, 0x000000df, 0x000000e3, 0x0003003e, 0x000000de, 0x000000e4,
    0x0004003d, 0x00000006, 0x000000ec, 0x000000de, 0x00060041, 0x000000ed, 0x000000ee, 0x000000eb,
    0x0000008d, 0x000000ec, 0x0004003d, 0x000000e5, 0x000000ef, 0x000000ee, 0x0003003e, 0x000000e7,
    0x000000ef, 0x0004003d, 0x00000006, 0x000000f1, 0x000000de, 0x00050080, 0x00000006, 0x000000f2,
    0x000000f1, 0x0000006d, 0x0004003d, 0x00000006, 0x000000f3, 0x0000007c, 0x00050080, 0x00000006,
    0x000000f4, 0x000000f2, 0x000000f3, 0x00060041, 0x000000ed, 0x000000f5, 0x000000eb, 0x0000008d,
    0x000000f4, 0x0004003d, 0x000000e5, 0x000000f6, 0x000000f5, 0x0003003e, 0x000000f0, 0x000000f6,
    0x00050041, 0x00000009, 0x000000fa, 0x000000e7, 0x00000065, 0x0004003d, 0x00000006, 0x000000fb,
    0x000000fa, 0x0006000c, 0x000000f7, 0x000000fc, 0x00000001, 0x0000003e, 0x000000fb, 0x0003003e,
    0x000000f9, 0x000000fc, 0x0004003d, 0x000000e5, 0x000000fe, 0x000000e7, 0x0008004f, 0x00000007,
    0x000000ff, 0x000000fe, 0x000000fe, 0x00000001, 0x00000002, 0x00000003, 0x0003003e, 0x000000fd,
    0x000000ff, 0x0004003d, 0x00000006, 0x00000100, 0x00000080, 0x00050084, 0x00000006, 0x00000101,
    0x000000ab, 0x00000100, 0x0004003d, 0x00000007, 0x00000105, 0x000000fd, 0x0003003e, 0x00000104,
    0x00000105, 0x0003003e, 0x00000106, 0x00000101, 0x00080039, 0x00000002, 0x00000109, 0x00000016,
    0x00000104, 0x00000106, 0x00000107, 0x00000108, 0x0004003d, 0x0000000f, 0x0000010a, 0x00000107,
    0x0003003e, 0x00000102, 0x0000010a, 0x0004003d, 0x0000000f, 0x0000010b, 0x00000108, 0x0003003e,
    0x00000103, 0x0000010b, 0x0004003d, 0x00000006, 0x0000010c, 0x00000080, 0x00050084, 0x00000006,
    0x0000010d, 0x000000ab, 0x0000010c, 0x00050080, 0x00000006, 0x0000010e, 0x0000010d, 0x0000006d,
    0x0004003d, 0x00000007, 0x00000112, 0x000000fd, 0x0003003e, 0x00000111, 0x00000112, 0x0003003e,
    0x00000113, 0x0000010e, 0x00080039, 0x00000002, 0x00000116, 0x00000016, 0x00000111, 0x00000113,
    0x00000114, 0x00000115, 0x0004003d, 0x0000000f, 0x00000117, 0x00000114, 0x0003003e, 0x0000010f,
    0x00000117, 0x0004003d, 0x0000000f, 0x00000118, 0x00000115, 0x0003003e, 0x00000110, 0x00000118,
    0x0004003d, 0x00000006, 0x0000011a, 0x000000d5, 0x00050084, 0x00000006, 0x0000011b, 0x0000011a,
    0x000000af, 0x0004003d, 0x00000006, 0x0000011c, 0x00000080, 0x00050084, 0x00000006, 0x0000011e,
    0x0000011c, 0x0000011d, 0x00050080, 0x00000006, 0x0000011f, 0x0000011b, 0x0000011e, 0x0004003d,
    0x00000006, 0x00000120, 0x00000084, 0x00050084, 0x00000006, 0x00000121, 0x00000120, 0x00000029,
    0x00050080, 0x00000006, 0x00000122, 0x0000011f, 0x00000121, 0x0003003e, 0x00000119, 0x00000122,
    0x0003003e, 0x00000123, 0x0000009c, 0x0003003e, 0x00000124, 0x0000009c, 0x0003003e, 0x00000125,
    0x0000009c, 0x0003003e, 0x00000126, 0x0000009c, 0x0003003e, 0x00000127, 0x00000065, 0x000200f9,
    0x00000128, 0x000200f8, 0x00000128, 0x000400f6, 0x0000012a, 0x0000012b, 0x00000000, 0x000200f9,
    0x0000012c, 0x000200f8, 0x0000012c, 0x0004003d, 0x00000006, 0x0000012d, 0x00000127, 0x000500b0,
    0x0000002a, 0x0000012e, 0x0000012d, 0x00000029, 0x000400fa, 0x0000012e, 0x00000129, 0x0000012a,
    0x000200f8, 0x00000129, 0x0004003d, 0x00000006, 0x00000130, 0x00000127, 0x00050041, 0x00000009,
    0x00000131, 0x000000f0, 0x00000130, 0x0004003d, 0x00000006, 0x00000132, 0x00000131, 0x0003003e,
    0x0000012f, 0x00000132, 0x0004003d, 0x00000006, 0x00000135, 0x0000012f, 0x000500c7, 0x00000006,
    0x00000136, 0x00000135, 0x00000047, 0x00040070, 0x0000000f, 0x00000137, 0x00000136, 0x0004003d,
    0x00000006, 0x00000138, 0x0000012f, 0x000500c2, 0x00000006, 0x0000013a, 0x00000138, 0x00000139,
    0x000500c7, 0x00000006, 0x0000013b, 0x0000013a, 0x00000047, 0x00040070, 0x0000000f, 0x0000013c,
    0x0000013b, 0x0004003d, 0x00000006, 0x0000013d, 0x0000012f, 0x000500c2, 0x00000006, 0x0000013f,
    0x0000013d, 0x0000013e, 0x000500c7, 0x00000006, 0x00000140, 0x0000013f, 0x00000047, 0x00040070,
    0x0000000f, 0x00000141, 0x00000140, 0x0004003d, 0x00000006, 0x00000142, 0x0000012f, 0x000500c2,
    0x00000006, 0x00000144, 0x00000142, 0x00000143, 0x000500c7, 0x00000006, 0x00000145, 0x00000144,
    0x00000047, 0x00040070, 0x0000000f, 0x00000146, 0x00000145, 0x00070050, 0x000000be, 0x00000147,
    0x00000137, 0x0000013c, 0x00000141, 0x00000146, 0x0003003e, 0x00000134, 0x00000147, 0x0004003d,
    0x00000006, 0x00000149, 0x0000012f, 0x000500c2, 0x00000006, 0x0000014a, 0x00000149, 0x00000051,
    0x000500c7, 0x00000006, 0x0000014b, 0x0000014a, 0x00000047, 0x00040070, 0x0000000f, 0x0000014c,
    0x0000014b, 0x0004003d, 0x00000006, 0x0000014d, 0x0000012f, 0x000500c2, 0x00000006, 0x0000014f,
    0x0000014d, 0x0000014e, 0x000500c7, 0x00000006, 0x00000150, 0x0000014f, 0x00000047, 0x00040070,
    0x0000000f, 0x00000151, 0x00000150, 0x0004003d, 0x00000006, 0x00000152, 0x0000012f, 0x000500c2,
    0x00000006, 0x00000154, 0x00000152, 0x00000153, 0x000500c7, 0x00000006, 0x00000155, 0x00000154,
    0x00000047, 0x00040070, 0x0000000f, 0x00000156, 0x00000155, 0x0004003d, 0x00000006, 0x00000157,
    0x0000012f, 0x000500c2, 0x00000006, 0x00000159, 0x00000157, 0x00000158, 0x000500c7, 0x00000006,
    0x0000015a, 0x00000159, 0x00000047, 0x00040070, 0x0000000f, 0x0000015b, 0x0000015a, 0x00070050,
    0x000000be, 0x0000015c, 0x0000014c, 0x00000151, 0x00000156, 0x0000015b, 0x0003003e, 0x00000148,
    0x0000015c, 0x0004003d, 0x00000006, 0x0000015e, 0x00000119, 0x0004003d, 0x00000006, 0x0000015f,
    0x00000127, 0x00050080, 0x00000006, 0x00000160, 0x0000015e, 0x0000015f, 0x00050041, 0x000000ce,
    0x00000161, 0x000000c2, 0x00000160, 0x0004003d, 0x000000be, 0x00000162, 0x00000161, 0x0003003e,
    0x0000015d, 0x00000162, 0x0004003d, 0x00000006, 0x00000164, 0x00000119, 0x00050080, 0x00000006,
    0x00000165, 0x00000164, 0x00000021, 0x0004003d, 0x00000006, 0x00000166, 0x00000127, 0x00050080,
    0x00000006, 0x00000167, 0x00000165, 0x00000166, 0x00050041, 0x000000ce, 0x00000168, 0x000000c2,
    0x00000167, 0x0004003d, 0x000000be, 0x00000169, 0x00000168, 0x0003003e, 0x00000163, 0x00000169,
    0x0004003d, 0x000000be, 0x0000016a, 0x00000134, 0x0004003d, 0x000000be, 0x0000016b, 0x0000015d,
    0x00050094, 0x0000000f, 0x0000016c, 0x0000016a, 0x0000016b, 0x0004003d, 0x0000000f, 0x0000016d,
    0x00000123, 0x00050081, 0x0000000f, 0x0000016e, 0x0000016d, 0x0000016c, 0x0003003e, 0x00000123,
    0x0000016e, 0x0004003d, 0x000000be, 0x0000016f, 0x00000148, 0x0004003d, 0x000000be, 0x00000170,
    0x00000163, 0x00050094, 0x0000000f, 0x00000171, 0x0000016f, 0x00000170, 0x0004003d, 0x0000000f,
    0x00000172, 0x00000124, 0x00050081, 0x0000000f, 0x00000173, 0x00000172, 0x00000171, 0x0003003e,
    0x00000124, 0x00000173, 0x00050041, 0x00000010, 0x00000174, 0x0000015d, 0x00000065, 0x0004003d,
    0x0000000f, 0x00000175, 0x00000174, 0x00050041, 0x00000010, 0x00000176, 0x0000015d, 0x0000006d,
    0x0004003d, 0x0000000f, 0x00000177, 0x00000176, 0x00050081, 0x0000000f, 0x00000178, 0x00000175,
    0x00000177, 0x00050041, 0x00000010, 0x00000179, 0x0000015d, 0x000000ab, 0x0004003d, 0x0000000f,
    0x0000017a, 0x00000179, 0x00050081, 0x0000000f, 0x0000017b, 0x00000178, 0x0000017a, 0x00050041,
    0x00000010, 0x0000017c, 0x0000015d, 0x0000001f, 0x0004003d, 0x0000000f, 0x0000017d, 0x0000017c,
    0x00050081, 0x0000000f, 0x0000017e, 0x0000017b, 0x0000017d, 0x0004003d, 0x0000000f, 0x0000017f,
    0x00000125, 0x00050081, 0x0000000f, 0x00000180, 0x0000017f, 0x0000017e, 0x0003003e, 0x00000125,
    0x00000180, 0x00050041, 0x00000010, 0x00000181, 0x00000163, 0x00000065, 0x0004003d, 0x0000000f,
    0x00000182, 0x00000181, 0x00050041, 0x00000010, 0x00000183, 0x00000163, 0x0000006d, 0x0004003d,
    0x0000000f, 0x00000184, 0x00000183, 0x00050081, 0x0000000f, 0x00000185, 0x00000182, 0x00000184,
    0x00050041, 0x00000010, 0x00000186, 0x00000163, 0x000000ab, 0x0004003d, 0x0000000f, 0x00000187,
    0x00000186, 0x00050081, 0x0000000f, 0x00000188, 0x00000185, 0x00000187, 0x00050041, 0x00000010,
    0x00000189, 0x00000163, 0x0000001f, 0x0004003d, 0x0000000f, 0x0000018a, 0x00000189, 0x00050081,
    0x0000000f, 0x0000018b, 0x00000188, 0x0000018a, 0x0004003d, 0x0000000f, 0x0000018c, 0x00000126,
    0x00050081, 0x0000000f, 0x0000018d, 0x0000018c, 0x0000018b, 0x0003003e, 0x00000126, 0x0000018d,
    0x000200f9, 0x0000012b, 0x000200f8, 0x0000012b, 0x0004003d, 0x00000006, 0x0000018e, 0x00000127,
    0x00050080, 0x00000006, 0x0000018f, 0x0000018e, 0x00000082, 0x0003003e, 0x00000127, 0x0000018f,
    0x000200f9, 0x00000128, 0x000200f8, 0x0000012a, 0x00050041, 0x00000010, 0x00000190, 0x000000f9,
    0x00000065, 0x0004003d, 0x0000000f, 0x00000191, 0x00000190, 0x0004003d, 0x0000000f, 0x00000192,
    0x00000102, 0x00050085, 0x0000000f, 0x00000193, 0x00000191, 0x00000192, 0x0004003d, 0x0000000f,
    0x00000194, 0x00000123, 0x00050085, 0x0000000f, 0x00000195, 0x00000193, 0x00000194, 0x00050041,
    0x00000010, 0x00000196, 0x000000f9, 0x0000006d, 0x0004003d, 0x0000000f, 0x00000197, 0x00000196,
    0x0004003d, 0x0000000f, 0x00000198, 0x00000103, 0x00050085, 0x0000000f, 0x00000199, 0x00000197,
    0x00000198, 0x0004003d, 0x0000000f, 0x0000019a, 0x00000125, 0x00050085, 0x0000000f, 0x0000019b,
    0x00000199, 0x0000019a, 0x00050083, 0x0000000f, 0x0000019c, 0x00000195, 0x0000019b, 0x00050041,
    0x00000010, 0x0000019d, 0x000000f9, 0x00000065, 0x0004003d, 0x0000000f, 0x0000019e, 0x0000019d,
    0x0004003d, 0x0000000f, 0x0000019f, 0x0000010f, 0x00050085, 0x0000000f, 0x000001a0, 0x0000019e,
    0x0000019f, 0x0004003d, 0x0000000f, 0x000001a1, 0x00000124, 0x00050085, 0x0000000f, 0x000001a2,
    0x000001a0, 0x000001a1, 0x00050081, 0x0000000f, 0x000001a3, 0x0000019c, 0x000001a2, 0x00050041,
    0x00000010, 0x000001a4, 0x000000f9, 0x0000006d, 0x0004003d, 0x0000000f, 0x000001a5, 0x000001a4,
    0x0004003d, 0x0000000f, 0x000001a6, 0x00000110, 0x00050085, 0x0000000f, 0x000001a7, 0x000001a5,
    0x000001a6, 0x0004003d, 0x0000000f, 0x000001a8, 0x00000126, 0x00050085, 0x0000000f, 0x000001a9,
    0x000001a7, 0x000001a8, 0x00050083, 0x0000000f, 0x000001aa, 0x000001a3, 0x000001a9, 0x0004003d,
    0x0000000f, 0x000001ab, 0x0000009b, 0x00050081, 0x0000000f, 0x000001ac, 0x000001ab, 0x000001aa,
    0x0003003e, 0x0000009b, 0x000001ac, 0x000200f9, 0x000000d9, 0x000200f8, 0x000000d9, 0x0004003d,
    0x00000006, 0x000001ad, 0x000000d5, 0x00050080, 0x00000006, 0x000001ae, 0x000001ad, 0x00000082,
    0x0003003e, 0x000000d5, 0x000001ae, 0x000200f9, 0x000000d6, 0x000200f8, 0x000000d8, 0x000200f9,
    0x000000d4, 0x000200f8, 0x000000d4, 0x000200f9, 0x000000a1, 0x000200f8, 0x000000a1, 0x0004003d,
    0x00000006, 0x000001af, 0x0000009d, 0x00050080, 0x00000006, 0x000001b0, 0x000001af, 0x00000021,
    0x0003003e, 0x0000009d, 0x000001b0, 0x000200f9, 0x0000009e, 0x000200f8, 0x000000a0, 0x0004003d,
    0x00000006, 0x000001b4, 0x00000062, 0x0004003d, 0x0000000f, 0x000001b5, 0x0000009b, 0x00050041,
    0x000001b6, 0x000001b7, 0x000001b3, 0x000001b4, 0x0003003e, 0x000001b7, 0x000001b5, 0x000400e0,
    0x000000ab, 0x000000ab, 0x000000ac, 0x0004003d, 0x00000006, 0x000001b8, 0x0000007c, 0x000500aa,
    0x0000002a, 0x000001b9, 0x000001b8, 0x00000065, 0x0004003d, 0x0000002a, 0x000001ba, 0x00000088,
    0x000500a7, 0x0000002a, 0x000001bb, 0x000001b9, 0x000001ba, 0x000300f7, 0x000001bd, 0x00000000,
    0x000400fa, 0x000001bb, 0x000001bc, 0x000001bd, 0x000200f8, 0x000001bc, 0x0003003e, 0x000001be,
    0x0000009c, 0x0003003e, 0x000001bf, 0x00000065, 0x000200f9, 0x000001c0, 0x000200f8, 0x000001c0,
    0x000400f6, 0x000001c2, 0x000001c3, 0x00000000, 0x000200f9, 0x000001c4, 0x000200f8, 0x000001c4,
    0x0004003d, 0x00000006, 0x000001c5, 0x000001bf, 0x000500b0, 0x0000002a, 0x000001c6, 0x000001c5,
    0x00000021, 0x000400fa, 0x000001c6, 0x000001c1, 0x000001c2, 0x000200f8, 0x000001c1, 0x0004003d,
    0x00000006, 0x000001c7, 0x00000062, 0x0004003d, 0x00000006, 0x000001c8, 0x000001bf, 0x00050080,
    0x00000006, 0x000001c9, 0x000001c7, 0x000001c8, 0x00050041, 0x000001b6, 0x000001ca, 0x000001b3,
    0x000001c9, 0x0004003d, 0x0000000f, 0x000001cb, 0x000001ca, 0x0004003d, 0x0000000f, 0x000001cc,
    0x000001be, 0x00050081, 0x0000000f, 0x000001cd, 0x000001cc, 0x000001cb, 0x0003003e, 0x000001be,
    0x000001cd, 0x000200f9, 0x000001c3, 0x000200f8, 0x000001c3, 0x0004003d, 0x00000006, 0x000001ce,
    0x000001bf, 0x00050080, 0x00000006, 0x000001cf, 0x000001ce, 0x00000082, 0x0003003e, 0x000001bf,
    0x000001cf, 0x000200f9, 0x000001c0, 0x000200f8, 0x000001c2, 0x0004003d, 0x00000006, 0x000001d4,
    0x00000075, 0x0004003d, 0x0000000f, 0x000001d5, 0x000001be, 0x00060041, 0x000001d6, 0x000001d7,
    0x000001d3, 0x0000008d, 0x000001d4, 0x0003003e, 0x000001d7, 0x000001d5, 0x000200f9, 0x000001bd,
    0x000200f8, 0x000001bd, 0x000100fd, 0x00010038, 0x00050036, 0x00000006, 0x0000000d, 0x00000000,
    0x0000000a, 0x00030037, 0x00000008, 0x0000000b, 0x00030037, 0x00000009, 0x0000000c, 0x000200f8,
    0x0000000e, 0x0004003d, 0x00000006, 0x00000018, 0x0000000c, 0x000500c2, 0x00000006, 0x0000001b,
    0x00000018, 0x0000001a, 0x00050041, 0x00000009, 0x0000001c, 0x0000000b, 0x0000001b, 0x0004003d,
    0x00000006, 0x0000001d, 0x0000001c, 0x0004003d, 0x00000006, 0x0000001e, 0x0000000c, 0x000500c7,
    0x00000006, 0x00000020, 0x0000001e, 0x0000001f, 0x00050084, 0x00000006, 0x00000022, 0x00000020,
    0x00000021, 0x000500c2, 0x00000006, 0x00000023, 0x0000001d, 0x00000022, 0x000500c7, 0x00000006,
    0x00000025, 0x00000023, 0x00000024, 0x000200fe, 0x00000025, 0x00010038, 0x00050036, 0x00000002,
    0x00000016, 0x00000000, 0x00000011, 0x00030037, 0x00000008, 0x00000012, 0x00030037, 0x00000009,
    0x00000013, 0x00030037, 0x00000010, 0x00000014, 0x00030037, 0x00000010, 0x00000015, 0x000200f8,
    0x00000017, 0x0004003b, 0x00000009, 0x0000002e, 0x00000007, 0x0004003b, 0x00000008, 0x0000002f,
    0x00000007, 0x0004003b, 0x00000009, 0x00000031, 0x00000007, 0x0004003b, 0x00000009, 0x00000036,
    0x00000007, 0x0004003b, 0x00000008, 0x00000039, 0x00000007, 0x0004003b, 0x00000009, 0x0000003b,
    0x00000007, 0x0004003b, 0x00000009, 0x0000003f, 0x00000007, 0x0004003b, 0x00000008, 0x00000042,
    0x00000007, 0x0004003b, 0x00000009, 0x00000044, 0x00000007, 0x0004003b, 0x00000008, 0x0000004b,
    0x00000007, 0x0004003b, 0x00000009, 0x0000004d, 0x00000007, 0x0004003b, 0x00000008, 0x00000056,
    0x00000007, 0x0004003b, 0x00000009, 0x00000058, 0x00000007, 0x0004003d, 0x00000006, 0x00000028,
    0x00000013, 0x000500b0, 0x0000002a, 0x0000002b, 0x00000028, 0x00000029, 0x000300f7, 0x0000002d,
    0x00000000, 0x000400fa, 0x0000002b, 0x0000002c, 0x0000003e, 0x000200f8, 0x0000002c, 0x0004003d,
    0x00000007, 0x00000030, 0x00000012, 0x0003003e, 0x0000002f, 0x00000030, 0x0004003d, 0x00000006,
    0x00000032, 0x00000013, 0x0003003e, 0x00000031, 0x00000032, 0x00060039, 0x00000006, 0x00000033,
    0x0000000d, 0x0000002f, 0x00000031, 0x000500c7, 0x00000006, 0x00000035, 0x00000033, 0x00000034,
    0x0003003e, 0x0000002e, 0x00000035, 0x0004003d, 0x00000006, 0x00000037, 0x00000013, 0x00050080,
    0x00000006, 0x00000038, 0x00000037, 0x00000029, 0x0004003d, 0x00000007, 0x0000003a, 0x00000012,
    0x0003003e, 0x00000039, 0x0000003a, 0x0003003e, 0x0000003b, 0x00000038, 0x00060039, 0x00000006,
    0x0000003c, 0x0000000d, 0x00000039, 0x0000003b, 0x000500c7, 0x00000006, 0x0000003d, 0x0000003c,
    0x00000034, 0x0003003e, 0x00000036, 0x0000003d, 0x000200f9, 0x0000002d, 0x000200f8, 0x0000003e,
    0x0004003d, 0x00000006, 0x00000040, 0x00000013, 0x00050080, 0x00000006, 0x00000041, 0x00000040,
    0x00000029, 0x0004003d, 0x00000007, 0x00000043, 0x00000012, 0x0003003e, 0x00000042, 0x00000043,
    0x0003003e, 0x00000044, 0x00000041, 0x00060039, 0x00000006, 0x00000045, 0x0000000d, 0x00000042,
    0x00000044, 0x0003003e, 0x0000003f, 0x00000045, 0x0004003d, 0x00000006, 0x00000046, 0x0000003f,
    0x000500c7, 0x00000006, 0x00000048, 0x00000046, 0x00000047, 0x0004003d, 0x00000006, 0x00000049,
    0x00000013, 0x00050082, 0x00000006, 0x0000004a, 0x00000049, 0x00000029, 0x0004003d, 0x00000007,
    0x0000004c, 0x00000012, 0x0003003e, 0x0000004b, 0x0000004c, 0x0003003e, 0x0000004d, 0x0000004a,
    0x00060039, 0x00000006, 0x0000004e, 0x0000000d, 0x0000004b, 0x0000004d, 0x000500c2, 0x00000006,
    0x00000050, 0x0000004e, 0x0000004f, 0x000500c4, 0x00000006, 0x00000052, 0x00000050, 0x00000051,
    0x000500c5, 0x00000006, 0x00000053, 0x00000048, 0x00000052, 0x0003003e, 0x0000002e, 0x00000053,
    0x0004003d, 0x00000006, 0x00000054, 0x0000003f, 0x000500c2, 0x00000006, 0x00000055, 0x00000054,
    0x00000051, 0x0004003d, 0x00000007, 0x00000057, 0x00000012, 0x0003003e, 0x00000056, 0x00000057,
    0x0004003d, 0x00000006, 0x00000059, 0x00000013, 0x0003003e, 0x00000058, 0x00000059, 0x00060039,
    0x00000006, 0x0000005a, 0x0000000d, 0x00000056, 0x00000058, 0x000500c2, 0x00000006, 0x0000005b,
    0x0000005a, 0x0000004f, 0x000500c4, 0x00000006, 0x0000005c, 0x0000005b, 0x00000051, 0x000500c5,
    0x00000006, 0x0000005d, 0x00000055, 0x0000005c, 0x0003003e, 0x00000036, 0x0000005d, 0x000200f9,
    0x0000002d, 0x000200f8, 0x0000002d, 0x0004003d, 0x00000006, 0x0000005e, 0x0000002e, 0x00040070,
    0x0000000f, 0x0000005f, 0x0000005e, 0x0003003e, 0x00000014, 0x0000005f, 0x0004003d, 0x00000006,
    0x00000060, 0x00000036, 0x00040070, 0x0000000f, 0x00000061, 0x00000060, 0x0003003e, 0x00000015,
    0x00000061, 0x000100fd, 0x00010038,
};
const size_t roaster_spv_q4k_v2_size = sizeof(roaster_spv_q4k_v2);
const uint32_t roaster_spv_q6k_v2[] = {
    0x07230203, 0x00010300, 0x0008000b, 0x0000022d, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00000080, 0x00000085, 0x0000008a,
    0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002,
    0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00050005, 0x0000000b, 0x3631646c,
    0x3b317528, 0x00000000, 0x00040005, 0x0000000a, 0x65747962, 0x0066664f, 0x00050005, 0x00000010,
    0x74796273, 0x31752865, 0x0000003b, 0x00030005, 0x0000000f, 0x0000006f, 0x00030005, 0x00000012,
    0x00000077, 0x00060005, 0x00000020, 0x67696557, 0x75427468, 0x72656666, 0x00000000, 0x00050006,
    0x00000020, 0x00000000, 0x61746164, 0x00000000, 0x00040005, 0x00000022, 0x67696577, 0x00737468,
    0x00030005, 0x00000038, 0x00000061, 0x00030005, 0x0000003c, 0x00000062, 0x00030005, 0x00000041,
    0x00000063, 0x00030005, 0x00000046, 0x00000064, 0x00030005, 0x0000004b, 0x00000065, 0x00030005,
    0x00000069, 0x00000076, 0x00030005, 0x0000007d, 0x00646974, 0x00080005, 0x00000080, 0x4c5f6c67,
    0x6c61636f, 0x6f766e49, 0x69746163, 0x44496e6f, 0x00000000, 0x00030005, 0x00000084, 0x00006777,
    0x00060005, 0x00000085, 0x575f6c67, 0x476b726f, 0x70756f72, 0x00004449, 0x00070005, 0x0000008a,
    0x4e5f6c67, 0x6f576d75, 0x72476b72, 0x7370756f, 0x00000000, 0x00030005, 0x0000008f, 0x00776f72,
    0x00030005, 0x00000096, 0x00627573, 0x00030005, 0x00000099, 0x0000006e, 0x00030005, 0x0000009d,
    0x00000068, 0x00040005, 0x000000a1, 0x696c6176, 0x00000064, 0x00040005, 0x000000a3, 0x61726150,
    0x0000736d, 0x00050006, 0x000000a3, 0x00000000, 0x73776f72, 0x00000000, 0x00050006, 0x000000a3,
    0x00000001, 0x736c6f63, 0x00000000, 0x00070006, 0x000000a3, 0x00000002, 0x636f6c62, 0x6550736b,
    0x776f5272, 0x00000000, 0x00030005, 0x000000a5, 0x00000070, 0x00030005, 0x000000aa, 0x0000626e,
    0x00040005, 0x000000ad, 0x42776f72, 0x00657479, 0x00030005, 0x000000b4, 0x006d7573, 0x00030005,
    0x000000b6, 0x00006274, 0x00040005, 0x000000bf, 0x6b6c626e, 0x00000000, 0x00030005, 0x000000c5,
    0x00007674, 0x00040005, 0x000000c9, 0x65736162, 0x00000034, 0x00030005, 0x000000cc, 0x00000069,
    0x00030005, 0x000000da, 0x00007378, 0x00060005, 0x000000dd, 0x74636556, 0x7542726f, 0x72656666,
    0x00000000, 0x00050006, 0x000000dd, 0x00000000, 0x61746164, 0x00000000, 0x00050005, 0x000000df,
    0x75706e69, 0x63655674, 0x00726f74, 0x00030005, 0x000000ed, 0x0000626c, 0x00030005, 0x000000f6,
    0x006b6c62, 0x00030005, 0x000000fe, 0x00000041, 0x00040005, 0x00000106, 0x61726170, 0x0000006d,
    0x00030005, 0x00000108, 0x00000042, 0x00040005, 0x00000112, 0x61726170, 0x0000006d, 0x00030005,
    0x00000114, 0x00000048, 0x00040005, 0x0000011e, 0x61726170, 0x0000006d, 0x00030005, 0x00000120,
    0x00006273, 0x00030005, 0x00000129, 0x00003173, 0x00040005, 0x0000012a, 0x61726170, 0x0000006d,
    0x00030005, 0x0000012d, 0x00003273, 0x00040005, 0x00000130, 0x61726170, 0x0000006d, 0x00030005,
    0x00000132, 0x00003373, 0x00040005, 0x00000135, 0x61726170, 0x0000006d, 0x00030005, 0x00000137,
    0x00003473, 0x00040005, 0x0000013b, 0x61726170, 0x0000006d, 0x00040005, 0x0000013d, 0x66664f64,
    0x00000000, 0x00030005, 0x00000141, 0x00000064, 0x00030005, 0x0000014f, 0x00006978, 0x00030005,
    0x00000158, 0x00003174, 0x00030005, 0x00000159, 0x00003274, 0x00030005, 0x0000015a, 0x00003374,
    0x00030005, 0x0000015b, 0x00003474, 0x00030005, 0x0000015c, 0x00000077, 0x00030005, 0x00000164,
    0x00000061, 0x00030005, 0x00000168, 0x00000062, 0x00030005, 0x0000016c, 0x00006868, 0x00030005,
    0x00000170, 0x0000006a, 0x00030005, 0x00000178, 0x00006873, 0x00030005, 0x0000017b, 0x00006261,
    0x00030005, 0x00000180, 0x00006262, 0x00030005, 0x00000185, 0x00006268, 0x00030005, 0x0000018b,
    0x00003171, 0x00030005, 0x0000019a, 0x00003271, 0x00030005, 0x000001a7, 0x00003371, 0x00030005,
    0x000001b4, 0x00003471, 0x00040005, 0x00000207, 0x74726170, 0x006c6169, 0x00030005, 0x00000212,
    0x00000074, 0x00030005, 0x00000213, 0x00000069, 0x00060005, 0x00000225, 0x7074754f, 0x75427475,
    0x72656666, 0x00000000, 0x00050006, 0x00000225, 0x00000000, 0x61746164, 0x00000000, 0x00060005,
    0x00000227, 0x7074756f, 0x65567475, 0x726f7463, 0x00000000, 0x00040047, 0x0000001f, 0x00000006,
    0x00000004, 0x00030047, 0x00000020, 0x00000002, 0x00040048, 0x00000020, 0x00000000, 0x00000018,
    0x00050048, 0x00000020, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000022, 0x00000018,
    0x00040047, 0x00000022, 0x00000021, 0x00000000, 0x00040047, 0x00000022, 0x00000022, 0x00000000,
    0x00040047, 0x00000080, 0x0000000b, 0x0000001b, 0x00040047, 0x00000085, 0x0000000b, 0x0000001a,
    0x00040047, 0x0000008a, 0x0000000b, 0x00000018, 0x00030047, 0x000000a3, 0x00000002, 0x00050048,
    0x000000a3, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x000000a3, 0x00000001, 0x00000023,
    0x00000004, 0x00050048, 0x000000a3, 0x00000002, 0x00000023, 0x00000008, 0x00040047, 0x000000dc,
    0x00000006, 0x00000010, 0x00030047, 0x000000dd, 0x00000002, 0x00040048, 0x000000dd, 0x00000000,
    0x00000018, 0x00050048, 0x000000dd, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x000000df,
    0x00000018, 0x00040047, 0x000000df, 0x00000021, 0x00000001, 0x00040047, 0x000000df, 0x00000022,
    0x00000000, 0x00040047, 0x00000224, 0x00000006, 0x00000004, 0x00030047, 0x00000225, 0x00000002,
    0x00040048, 0x00000225, 0x00000000, 0x00000019, 0x00050048, 0x00000225, 0x00000000, 0x00000023,
    0x00000000, 0x00030047, 0x00000227, 0x00000019, 0x00040047, 0x00000227, 0x00000021, 0x00000002,
    0x00040047, 0x00000227, 0x00000022, 0x00000000, 0x00040047, 0x0000022c, 0x0000000b, 0x00000019,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
    0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017, 0x00000008, 0x00000006,
    0x00000004, 0x00040021, 0x00000009, 0x00000008, 0x00000007, 0x00030016, 0x0000000d, 0x00000020,
    0x00040021, 0x0000000e, 0x0000000d, 0x00000007, 0x00040015, 0x00000014, 0x00000020, 0x00000001,
    0x0004002b, 0x00000014, 0x00000015, 0x00000002, 0x0004002b, 0x00000006, 0x00000018, 0x00000002,
    0x0004002b, 0x00000006, 0x0000001a, 0x00000000, 0x00020014, 0x0000001b, 0x0003001d, 0x0000001f,
    0x00000006, 0x0003001e, 0x00000020, 0x0000001f, 0x00040020, 0x00000021, 0x0000000c, 0x00000020,
    0x0004003b, 0x00000021, 0x00000022, 0x0000000c, 0x0004002b, 0x00000014, 0x00000023, 0x00000000,
    0x00040020, 0x00000025, 0x0000000c, 0x00000006, 0x0004002b, 0x00000006, 0x00000029, 0x00000001,
    0x0004002b, 0x00000006, 0x00000032, 0x00000003, 0x0004002b, 0x00000006, 0x0000004d, 0x00000004,
    0x0004002b, 0x00000014, 0x00000052, 0x00000010, 0x0004002b, 0x00000006, 0x00000070, 0x00000008,
    0x0004002b, 0x00000006, 0x00000073, 0x000000ff, 0x0004002b, 0x00000014, 0x00000076, 0x00000018,
    0x00040017, 0x0000007e, 0x00000006, 0x00000003, 0x00040020, 0x0000007f, 0x00000001, 0x0000007e,
    0x0004003b, 0x0000007f, 0x00000080, 0x00000001, 0x00040020, 0x00000081, 0x00000001, 0x00000006,
    0x0004003b, 0x0000007f, 0x00000085, 0x00000001, 0x0004003b, 0x0000007f, 0x0000008a, 0x00000001,
    0x0004002b, 0x00000006, 0x00000091, 0x00000010, 0x0004002b, 0x00000014, 0x0000009b, 0x00000001,
    0x00040020, 0x000000a0, 0x00000007, 0x0000001b, 0x0005001e, 0x000000a3, 0x00000006, 0x00000006,
    0x00000006, 0x00040020, 0x000000a4, 0x00000009, 0x000000a3, 0x0004003b, 0x000000a4, 0x000000a5,
    0x00000009, 0x00040020, 0x000000a6, 0x00000009, 0x00000006, 0x0004002b, 0x00000006, 0x000000b1,
    0x000000d2, 0x00040020, 0x000000b3, 0x00000007, 0x0000000d, 0x0004002b, 0x0000000d, 0x000000b5,
    0x00000000, 0x0004002b, 0x00000006, 0x000000c4, 0x00000108, 0x0004002b, 0x00000006, 0x000000c7,
    0x00000040, 0x00040017, 0x000000d6, 0x0000000d, 0x00000004, 0x0004002b, 0x00000006, 0x000000d7,
    0x00000200, 0x0004001c, 0x000000d8, 0x000000d6, 0x000000d7, 0x00040020, 0x000000d9, 0x00000004,
    0x000000d8, 0x0004003b, 0x000000d9, 0x000000da, 0x00000004, 0x0003001d, 0x000000dc, 0x000000d6,
    0x0003001e, 0x000000dd, 0x000000dc, 0x00040020, 0x000000de, 0x0000000c, 0x000000dd, 0x0004003b,
    0x000000de, 0x000000df, 0x0000000c, 0x00040020, 0x000000e3, 0x0000000c, 0x000000d6, 0x00040020,
    0x000000e6, 0x00000004, 0x000000d6, 0x00040020, 0x000000fd, 0x00000007, 0x00000008, 0x0004002b,
    0x00000006, 0x0000010d, 0x00000020, 0x0004002b, 0x00000006, 0x00000116, 0x00000080, 0x0004002b,
    0x00000006, 0x00000122, 0x000000c0, 0x0004002b, 0x00000006, 0x00000139, 0x00000006, 0x0004002b,
    0x00000006, 0x0000013f, 0x000000d0, 0x0004002b, 0x00000006, 0x0000014a, 0x0000ffff, 0x00040017,
    0x0000014c, 0x0000000d, 0x00000002, 0x00040020, 0x0000018a, 0x00000007, 0x000000d6, 0x0004002b,
    0x00000006, 0x0000018e, 0x0000000f, 0x0004002b, 0x00000014, 0x00000192, 0x00000004, 0x0004002b,
    0x00000014, 0x00000196, 0x00000020, 0x0004002b, 0x00000014, 0x000001b9, 0x00000006, 0x0004002b,
    0x00000006, 0x000001e3, 0x00000018, 0x0004001c, 0x00000205, 0x0000000d, 0x000000c7, 0x00040020,
    0x00000206, 0x00000004, 0x00000205, 0x0004003b, 0x00000206, 0x00000207, 0x00000004, 0x00040020,
    0x0000020a, 0x00000004, 0x0000000d, 0x0003001d, 0x00000224, 0x0000000d, 0x0003001e, 0x00000225,
    0x00000224, 0x00040020, 0x00000226, 0x0000000c, 0x00000225, 0x0004003b, 0x00000226, 0x00000227,
    0x0000000c, 0x00040020, 0x0000022a, 0x0000000c, 0x0000000d, 0x0006002c, 0x0000007e, 0x0000022c,
    0x000000c7, 0x00000029, 0x00000029, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x0000007d, 0x00000007, 0x0004003b, 0x00000007,
    0x00000084, 0x00000007, 0x0004003b, 0x00000007, 0x0000008f, 0x00000007, 0x0004003b, 0x00000007,
    0x00000096, 0x00000007, 0x0004003b, 0x00000007, 0x00000099, 0x00000007, 0x0004003b, 0x00000007,
    0x0000009d, 0x00000007, 0x0004003b, 0x000000a0, 0x000000a1, 0x00000007, 0x0004003b, 0x00000007,
    0x000000aa, 0x00000007, 0x0004003b, 0x00000007, 0x000000ad, 0x00000007, 0x0004003b, 0x000000b3,
    0x000000b4, 0x00000007, 0x0004003b, 0x00000007, 0x000000b6, 0x00000007, 0x0004003b, 0x00000007,
    0x000000bf, 0x00000007, 0x0004003b, 0x00000007, 0x000000c5, 0x00000007, 0x0004003b, 0x00000007,
    0x000000c9, 0x00000007, 0x0004003b, 0x00000007, 0x000000cc, 0x00000007, 0x0004003b, 0x00000007,
    0x000000ed, 0x00000007, 0x0004003b, 0x00000007, 0x000000f6, 0x00000007, 0x0004003b, 0x000000fd,
    0x000000fe, 0x00000007, 0x0004003b, 0x00000007, 0x00000106, 0x00000007, 0x0004003b, 0x000000fd,
    0x00000108, 0x00000007, 0x0004003b, 0x00000007, 0x00000112, 0x00000007, 0x0004003b, 0x000000fd,
    0x00000114, 0x00000007, 0x0004003b, 0x00000007, 0x0000011e, 0x00000007, 0x0004003b, 0x00000007,
    0x00000120, 0x00000007, 0x0004003b, 0x000000b3, 0x00000129, 0x00000007, 0x0004003b, 0x00000007,
    0x0000012a, 0x00000007, 0x0004003b, 0x000000b3, 0x0000012d, 0x00000007, 0x0004003b, 0x00000007,
    0x00000130, 0x00000007, 0x0004003b, 0x000000b3, 0x00000132, 0x00000007, 0x0004003b, 0x00000007,
    0x00000135, 0x00000007, 0x0004003b, 0x000000b3, 0x00000137, 0x00000007, 0x0004003b, 0x00000007,
    0x0000013b, 0x00000007, 0x0004003b, 0x00000007, 0x0000013d, 0x00000007, 0x0004003b, 0x000000b3,
    0x00000141, 0x00000007, 0x0004003b, 0x00000007, 0x0000014f, 0x00000007, 0x0004003b, 0x000000b3,
    0x00000158, 0x00000007, 0x0004003b, 0x000000b3, 0x00000159, 0x00000007, 0x0004003b, 0x000000b3,
    0x0000015a, 0x00000007, 0x0004003b, 0x000000b3, 0x0000015b, 0x00000007, 0x0004003b, 0x00000007,
    0x0000015c, 0x00000007, 0x0004003b, 0x00000007, 0x00000164, 0x00000007, 0x0004003b, 0x00000007,
    0x00000168, 0x00000007, 0x0004003b, 0x00000007, 0x0000016c, 0x00000007, 0x0004003b, 0x00000007,
    0x00000170, 0x00000007, 0x0004003b, 0x00000007, 0x00000178, 0x00000007, 0x0004003b, 0x00000007,
    0x0000017b, 0x00000007, 0x0004003b, 0x00000007, 0x00000180, 0x00000007, 0x0004003b, 0x00000007,
    0x00000185, 0x00000007, 0x0004003b, 0x0000018a, 0x0000018b, 0x00000007, 0x0004003b, 0x0000018a,
    0x0000019a, 0x00000007, 0x0004003b, 0x0000018a, 0x000001a7, 0x00000007, 0x0004003b, 0x0000018a,
    0x000001b4, 0x00000007, 0x0004003b, 0x000000b3, 0x00000212, 0x00000007, 0x0004003b, 0x00000007,
    0x00000213, 0x00000007, 0x00050041, 0x00000081, 0x00000082, 0x00000080, 0x0000001a, 0x0004003d,
    0x00000006, 0x00000083, 0x00000082, 0x0003003e, 0x0000007d, 0x00000083, 0x00050041, 0x00000081,
    0x00000086, 0x00000085, 0x0000001a, 0x0004003d, 0x00000006, 0x00000087, 0x00000086, 0x00050041,
    0x00000081, 0x00000088, 0x00000085, 0x00000029, 0x0004003d, 0x00000006, 0x00000089, 0x00000088,
    0x00050041, 0x00000081, 0x0000008b, 0x0000008a, 0x0000001a, 0x0004003d, 0x00000006, 0x0000008c,
    0x0000008b, 0x00050084, 0x00000006, 0x0000008d, 0x00000089, 0x0000008c, 0x00050080, 0x00000006,
    0x0000008e, 0x00000087, 0x0000008d, 0x0003003e, 0x00000084, 0x0000008e, 0x0004003d, 0x00000006,
    0x00000090, 0x00000084, 0x00050084, 0x00000006, 0x00000092, 0x00000090, 0x00000091, 0x0004003d,
    0x00000006, 0x00000093, 0x0000007d, 0x000500c2, 0x00000006, 0x00000094, 0x00000093, 0x00000015,
    0x00050080, 0x00000006, 0x00000095, 0x00000092, 0x00000094, 0x0003003e, 0x0000008f, 0x00000095,
    0x0004003d, 0x00000006, 0x00000097, 0x0000007d, 0x000500c7, 0x00000006, 0x00000098, 0x00000097,
    0x00000032, 0x0003003e, 0x00000096, 0x00000098, 0x0004003d, 0x00000006, 0x0000009a, 0x00000096,
    0x000500c2, 0x00000006, 0x0000009c, 0x0000009a, 0x0000009b, 0x0003003e, 0x00000099, 0x0000009c,
    0x0004003d, 0x00000006, 0x0000009e, 0x00000096, 0x000500c7, 0x00000006, 0x0000009f, 0x0000009e,
    0x00000029, 0x0003003e, 0x0000009d, 0x0000009f, 0x0004003d, 0x00000006, 0x000000a2, 0x0000008f,
    0x00050041, 0x000000a6, 0x000000a7, 0x000000a5, 0x00000023, 0x0004003d, 0x00000006, 0x000000a8,
    0x000000a7, 0x000500b0, 0x0000001b, 0x000000a9, 0x000000a2, 0x000000a8, 0x0003003e, 0x000000a1,
    0x000000a9, 0x00050041, 0x000000a6, 0x000000ab, 0x000000a5, 0x00000015, 0x0004003d, 0x00000006,
    0x000000ac, 0x000000ab, 0x0003003e, 0x000000aa, 0x000000ac, 0x0004003d, 0x00000006, 0x000000ae,
    0x0000008f, 0x0004003d, 0x00000006, 0x000000af, 0x000000aa, 0x00050084, 0x00000006, 0x000000b0,
    0x000000ae, 0x000000af, 0x00050084, 0x00000006, 0x000000b2, 0x000000b0, 0x000000b1, 0x0003003e,
    0x000000ad, 0x000000b2, 0x0003003e, 0x000000b4, 0x000000b5, 0x0003003e, 0x000000b6, 0x0000001a,
    0x000200f9, 0x000000b7, 0x000200f8, 0x000000b7, 0x000400f6, 0x000000b9, 0x000000ba, 0x00000000,
    0x000200f9, 0x000000bb, 0x000200f8, 0x000000bb, 0x0004003d, 0x00000006, 0x000000bc, 0x000000b6,
    0x0004003d, 0x00000006, 0x000000bd, 0x000000aa, 0x000500b0, 0x0000001b, 0x000000be, 0x000000bc,
    0x000000bd, 0x000400fa, 0x000000be, 0x000000b8, 0x000000b9, 0x000200f8, 0x000000b8, 0x0004003d,
    0x00000006, 0x000000c0, 0x000000aa, 0x0004003d, 0x00000006, 0x000000c1, 0x000000b6, 0x00050082,
    0x00000006, 0x000000c2, 0x000000c0, 0x000000c1, 0x0007000c, 0x00000006, 0x000000c3, 0x00000001,
    0x00000026, 0x00000070, 0x000000c2, 0x0003003e, 0x000000bf, 0x000000c3, 0x000400e0, 0x00000018,
    0x00000018, 0x000000c4, 0x0004003d, 0x00000006, 0x000000c6, 0x000000bf, 0x00050084, 0x00000006,
    0x000000c8, 0x000000c6, 0x000000c7, 0x0003003e, 0x000000c5, 0x000000c8, 0x0004003d, 0x00000006,
    0x000000ca, 0x000000b6, 0x00050084, 0x00000006, 0x000000cb, 0x000000ca, 0x000000c7, 0x0003003e,
    0x000000c9, 0x000000cb, 0x0004003d, 0x00000006, 0x000000cd, 0x0000007d, 0x0003003e, 0x000000cc,
    0x000000cd, 0x000200f9, 0x000000ce, 0x000200f8, 0x000000ce, 0x000400f6, 0x000000d0, 0x000000d1,
    0x00000000, 0x000200f9, 0x000000d2, 0x000200f8, 0x000000d2, 0x0004003d, 0x00000006, 0x000000d3,
    0x000000cc, 0x0004003d, 0x00000006, 0x000000d4, 0x000000c5, 0x000500b0, 0x0000001b, 0x000000d5,
    0x000000d3, 0x000000d4, 0x000400fa, 0x000000d5, 0x000000cf, 0x000000d0, 0x000200f8, 0x000000cf,
    0x0004003d, 0x00000006, 0x000000db, 0x000000cc, 0x0004003d, 0x00000006, 0x000000e0, 0x000000c9,
    0x0004003d, 0x00000006, 0x000000e1, 0x000000cc, 0x00050080, 0x00000006, 0x000000e2, 0x000000e0,
    0x000000e1, 0x00060041, 0x000000e3, 0x000000e4, 0x000000df, 0x00000023, 0x000000e2, 0x0004003d,
    0x000000d6, 0x000000e5, 0x000000e4, 0x00050041, 0x000000e6, 0x000000e7, 0x000000da, 0x000000db,
    0x0003003e, 0x000000e7, 0x000000e5, 0x000200f9, 0x000000d1, 0x000200f8, 0x000000d1, 0x0004003d,
    0x00000006, 0x000000e8, 0x000000cc, 0x00050080, 0x00000006, 0x000000e9, 0x000000e8, 0x000000c7,
    0x0003003e, 0x000000cc, 0x000000e9, 0x000200f9, 0x000000ce, 0x000200f8, 0x000000d0, 0x000400e0,
    0x00000018, 0x00000018, 0x000000c4, 0x0004003d, 0x0000001b, 0x000000ea, 0x000000a1, 0x000300f7,
    0x000000ec, 0x00000000, 0x000400fa, 0x000000ea, 0x000000eb, 0x000000ec, 0x000200f8, 0x000000eb,
    0x0003003e, 0x000000ed, 0x0000001a, 0x000200f9, 0x000000ee, 0x000200f8, 0x000000ee, 0x000400f6,
    0x000000f0, 0x000000f1, 0x00000000, 0x000200f9, 0x000000f2, 0x000200f8, 0x000000f2, 0x0004003d,
    0x00000006, 0x000000f3, 0x000000ed, 0x0004003d, 0x00000006, 0x000000f4, 0x000000bf, 0x000500b0,
    0x0000001b, 0x000000f5, 0x000000f3, 0x000000f4, 0x000400fa, 0x000000f5, 0x000000ef, 0x000000f0,
    0x000200f8, 0x000000ef, 0x0004003d, 0x00000006, 0x000000f7, 0x000000ad, 0x0004003d, 0x00000006,
    0x000000f8, 0x000000b6, 0x0004003d, 0x00000006, 0x000000f9, 0x000000ed, 0x00050080, 0x00000006,
    0x000000fa, 0x000000f8, 0x000000f9, 0x00050084, 0x00000006, 0x000000fb, 0x000000fa, 0x000000b1,
    0x00050080, 0x00000006, 0x000000fc, 0x000000f7, 0x000000fb, 0x0003003e, 0x000000f6, 0x000000fc,
    0x0004003d, 0x00000006, 0x000000ff, 0x000000f6, 0x0004003d, 0x00000006, 0x00000100, 0x00000099,
    0x00050084, 0x00000006, 0x00000101, 0x000000c7, 0x00000100, 0x00050080, 0x00000006, 0x00000102,
    0x000000ff, 0x00000101, 0x0004003d, 0x00000006, 0x00000103, 0x0000009d, 0x00050084, 0x00000006,
    0x00000104, 0x00000091, 0x00000103, 0x00050080, 0x00000006, 0x00000105, 0x00000102, 0x00000104,
    0x0003003e, 0x00000106, 0x00000105, 0x00050039, 0x00000008, 0x00000107, 0x0000000b, 0x00000106,
    0x0003003e, 0x000000fe, 0x00000107, 0x0004003d, 0x00000006, 0x00000109, 0x000000f6, 0x0004003d,
    0x00000006, 0x0000010a, 0x00000099, 0x00050084, 0x00000006, 0x0000010b, 0x000000c7, 0x0000010a,
    0x00050080, 0x00000006, 0x0000010c, 0x00000109, 0x0000010b, 0x00050080, 0x00000006, 0x0000010e,
    0x0000010c, 0x0000010d, 0x0004003d, 0x00000006, 0x0000010f, 0x0000009d, 0x00050084, 0x00000006,
    0x00000110, 0x00000091, 0x0000010f, 0x00050080, 0x00000006, 0x00000111, 0x0000010e, 0x00000110,
    0x0003003e, 0x00000112, 0x00000111, 0x00050039, 0x00000008, 0x00000113, 0x0000000b, 0x00000112,
    0x0003003e, 0x00000108, 0x00000113, 0x0004003d, 0x00000006, 0x00000115, 0x000000f6, 0x00050080,
    0x00000006, 0x00000117, 0x00000115, 0x00000116, 0x0004003d, 0x00000006, 0x00000118, 0x00000099,
    0x00050084, 0x00000006, 0x00000119, 0x0000010d, 0x00000118, 0x00050080, 0x00000006, 0x0000011a,
    0x00000117, 0x00000119, 0x0004003d, 0x00000006, 0x0000011b, 0x0000009d, 0x00050084, 0x00000006,
    0x0000011c, 0x00000091, 0x0000011b, 0x00050080, 0x00000006, 0x0000011d, 0x0000011a, 0x0000011c,
    0x0003003e, 0x0000011e, 0x0000011d, 0x00050039, 0x00000008, 0x0000011f, 0x0000000b, 0x0000011e,
    0x0003003e, 0x00000114, 0x0000011f, 0x0004003d, 0x00000006, 0x00000121, 0x000000f6, 0x00050080,
    0x00000006, 0x00000123, 0x00000121, 0x00000122, 0x0004003d, 0x00000006, 0x00000124, 0x00000099,
    0x00050084, 0x00000006, 0x00000125, 0x00000070, 0x00000124, 0x00050080, 0x00000006, 0x00000126,
    0x00000123, 0x00000125, 0x0004003d, 0x00000006, 0x00000127, 0x0000009d, 0x00050080, 0x00000006,
    0x00000128, 0x00000126, 0x00000127, 0x0003003e, 0x00000120, 0x00000128, 0x0004003d, 0x00000006,
    0x0000012b, 0x00000120, 0x0003003e, 0x0000012a, 0x0000012b, 0x00050039, 0x0000000d, 0x0000012c,
    0x00000010, 0x0000012a, 0x0003003e, 0x00000129, 0x0000012c, 0x0004003d, 0x00000006, 0x0000012e,
    0x00000120, 0x00050080, 0x00000006, 0x0000012f, 0x0000012e, 0x00000018, 0x0003003e, 0x00000130,
    0x0000012f, 0x00050039, 0x0000000d, 0x00000131, 0x00000010, 0x00000130, 0x0003003e, 0x0000012d,
    0x00000131, 0x0004003d, 0x00000006, 0x00000133, 0x00000120, 0x00050080, 0x00000006, 0x00000134,
    0x00000133, 0x0000004d, 0x0003003e, 0x00000135, 0x00000134, 0x00050039, 0x0000000d, 0x00000136,
    0x00000010, 0x00000135, 0x0003003e, 0x00000132, 0x00000136, 0x0004003d, 0x00000006, 0x00000138,
    0x00000120, 0x00050080, 0x00000006, 0x0000013a, 0x00000138, 0x00000139, 0x0003003e, 0x0000013b,
    0x0000013a, 0x00050039, 0x0000000d, 0x0000013c, 0x00000010, 0x0000013b, 0x0003003e, 0x00000137,
    0x0000013c, 0x0004003d, 0x00000006, 0x0000013e, 0x000000f6, 0x00050080, 0x00000006, 0x00000140,
    0x0000013e, 0x0000013f, 0x0003003e, 0x0000013d, 0x00000140, 0x0004003d, 0x00000006, 0x00000142,
    0x0000013d, 0x000500c2, 0x00000006, 0x00000143, 0x00000142, 0x00000015, 0x00060041, 0x00000025,
    0x00000144, 0x00000022, 0x00000023, 0x00000143, 0x0004003d, 0x00000006, 0x00000145, 0x00000144,
    0x0004003d, 0x00000006, 0x00000146, 0x0000013d, 0x000500c7, 0x00000006, 0x00000147, 0x00000146,
    0x00000018, 0x00050084, 0x00000006, 0x00000148, 0x00000147, 0x00000070, 0x000500c2, 0x00000006,
    0x00000149, 0x00000145, 0x00000148, 0x000500c7, 0x00000006, 0x0000014b, 0x00000149, 0x0000014a,
    0x0006000c, 0x0000014c, 0x0000014d, 0x00000001, 0x0000003e, 0x0000014b, 0x00050051, 0x0000000d,
    0x0000014e, 0x0000014d, 0x00000000, 0x0003003e, 0x00000141, 0x0000014e, 0x0004003d, 0x00000006,
    0x00000150, 0x000000ed, 0x00050084, 0x00000006, 0x00000151, 0x00000150, 0x000000c7, 0x0004003d,
    0x00000006, 0x00000152, 0x00000099, 0x00050084, 0x00000006, 0x00000153, 0x00000152, 0x0000010d,
    0x00050080, 0x00000006, 0x00000154, 0x00000151, 0x00000153, 0x0004003d, 0x00000006, 0x00000155,
    0x0000009d, 0x00050084, 0x00000006, 0x00000156, 0x00000155, 0x0000004d, 0x00050080, 0x00000006,
    0x00000157, 0x00000154, 0x00000156, 0x0003003e, 0x0000014f, 0x00000157, 0x0003003e, 0x00000158,
    0x000000b5, 0x0003003e, 0x00000159, 0x000000b5, 0x0003003e, 0x0000015a, 0x000000b5, 0x0003003e,
    0x0000015b, 0x000000b5, 0x0003003e, 0x0000015c, 0x0000001a, 0x000200f9, 0x0000015d, 0x000200f8,
    0x0000015d, 0x000400f6, 0x0000015f, 0x00000160, 0x00000000, 0x000200f9, 0x00000161, 0x000200f8,
    0x00000161, 0x0004003d, 0x00000006, 0x00000162, 0x0000015c, 0x000500b0, 0x0000001b, 0x00000163,
    0x00000162, 0x0000004d, 0x000400fa, 0x00000163, 0x0000015e, 0x0000015f, 0x000200f8, 0x0000015e,
    0x0004003d, 0x00000006, 0x00000165, 0x0000015c, 0x00050041, 0x00000007, 0x00000166, 0x000000fe,
    0x00000165, 0x0004003d, 0x00000006, 0x00000167, 0x00000166, 0x0003003e, 0x00000164, 0x00000167,
    0x0004003d, 0x00000006, 0x00000169, 0x0000015c, 0x00050041, 0x00000007, 0x0000016a, 0x00000108,
    0x00000169, 0x0004003d, 0x00000006, 0x0000016b, 0x0000016a, 0x0003003e, 0x00000168, 0x0000016b,
    0x0004003d, 0x00000006, 0x0000016d, 0x0000015c, 0x00050041, 0x00000007, 0x0000016e, 0x00000114,
    0x0000016d, 0x0004003d, 0x00000006, 0x0000016f, 0x0000016e, 0x0003003e, 0x0000016c, 0x0000016f,
    0x0003003e, 0x00000170, 0x0000001a, 0x000200f9, 0x00000171, 0x000200f8, 0x00000171, 0x000400f6,
    0x00000173, 0x00000174, 0x00000000, 0x000200f9, 0x00000175, 0x000200f8, 0x00000175, 0x0004003d,
    0x00000006, 0x00000176, 0x00000170, 0x000500b0, 0x0000001b, 0x00000177, 0x00000176, 0x0000004d,
    0x000400fa, 0x00000177, 0x00000172, 0x00000173, 0x000200f8, 0x00000172, 0x0004003d, 0x00000006,
    0x00000179, 0x00000170, 0x00050084, 0x00000006, 0x0000017a, 0x00000070, 0x00000179, 0x0003003e,
    0x00000178, 0x0000017a, 0x0004003d, 0x00000006, 0x0000017c, 0x00000164, 0x0004003d, 0x00000006,
    0x0000017d, 0x00000178, 0x000500c2, 0x00000006, 0x0000017e, 0x0000017c, 0x0000017d, 0x000500c7,
    0x00000006, 0x0000017f, 0x0000017e, 0x00000073, 0x0003003e, 0x0000017b, 0x0000017f, 0x0004003d,
    0x00000006, 0x00000181, 0x00000168, 0x0004003d, 0x00000006, 0x00000182, 0x00000178, 0x000500c2,
    0x00000006, 0x00000183, 0x00000181, 0x00000182, 0x000500c7, 0x00000006, 0x00000184, 0x00000183,
    0x00000073, 0x0003003e, 0x00000180, 0x00000184, 0x0004003d, 0x00000006, 0x00000186, 0x0000016c,
    0x0004003d, 0x00000006, 0x00000187, 0x00000178, 0x000500c2, 0x00000006, 0x00000188, 0x00000186,
    0x00000187, 0x000500c7, 0x00000006, 0x00000189, 0x00000188, 0x00000073, 0x0003003e, 0x00000185,
    0x00000189, 0x0004003d, 0x00000006, 0x0000018c, 0x00000170, 0x0004003d, 0x00000006, 0x0000018d,
    0x0000017b, 0x000500c7, 0x00000006, 0x0000018f, 0x0000018d, 0x0000018e, 0x0004003d, 0x00000006,
    0x00000190, 0x00000185, 0x000500c7, 0x00000006, 0x00000191, 0x00000190, 0x00000032, 0x000500c4,
    0x00000006, 0x00000193, 0x00000191, 0x00000192, 0x000500c5, 0x00000006, 0x00000194, 0x0000018f,
    0x00000193, 0x0004007c, 0x00000014, 0x00000195, 0x00000194, 0x00050082, 0x00000014, 0x00000197,
    0x00000195, 0x00000196, 0x0004006f, 0x0000000d, 0x00000198, 0x00000197, 0x00050041, 0x000000b3,
    0x00000199, 0x0000018b, 0x0000018c, 0x0003003e, 0x00000199, 0x00000198, 0x0004003d, 0x00000006,
    0x0000019b, 0x00000170, 0x0004003d, 0x00000006, 0x0000019c, 0x00000180, 0x000500c7, 0x00000006,
    0x0000019d, 0x0000019c, 0x0000018e, 0x0004003d, 0x00000006, 0x0000019e, 0x00000185, 0x000500c2,
    0x00000006, 0x0000019f, 0x0000019e, 0x00000015, 0x000500c7, 0x00000006, 0x000001a0, 0x0000019f,
    0x00000032, 0x000500c4, 0x00000006, 0x000001a1, 0x000001a0, 0x00000192, 0x000500c5, 0x00000006,
    0x000001a2, 0x0000019d, 0x000001a1, 0x0004007c, 0x00000014, 0x000001a3, 0x000001a2, 0x00050082,
    0x00000014, 0x000001a4, 0x000001a3, 0x00000196, 0x0004006f, 0x0000000d, 0x000001a5, 0x000001a4,
    0x00050041, 0x000000b3, 0x000001a6, 0x0000019a, 0x0000019b, 0x0003003e, 0x000001a6, 0x000001a5,
    0x0004003d, 0x00000006, 0x000001a8, 0x00000170, 0x0004003d, 0x00000006, 0x000001a9, 0x0000017b,
    0x000500c2, 0x00000006, 0x000001aa, 0x000001a9, 0x00000192, 0x0004003d, 0x00000006, 0x000001ab,
    0x00000185, 0x000500c2, 0x00000006, 0x000001ac, 0x000001ab, 0x00000192, 0x000500c7, 0x00000006,
    0x000001ad, 0x000001ac, 0x00000032, 0x000500c4, 0x00000006, 0x000001ae, 0x000001ad, 0x00000192,
    0x000500c5, 0x00000006, 0x000001af, 0x000001aa, 0x000001ae, 0x0004007c, 0x00000014, 0x000001b0,
    0x000001af, 0x00050082, 0x00000014, 0x000001b1, 0x000001b0, 0x00000196, 0x0004006f, 0x0000000d,
    0x000001b2, 0x000001b1, 0x00050041, 0x000000b3, 0x000001b3, 0x000001a7, 0x000001a8, 0x0003003e,
    0x000001b3, 0x000001b2, 0x0004003d, 0x00000006, 0x000001b5, 0x00000170, 0x0004003d, 0x00000006,
    0x000001b6, 0x00000180, 0x000500c2, 0x00000006, 0x000001b7, 0x000001b6, 0x00000192, 0x0004003d,
    0x00000006, 0x000001b8, 0x00000185, 0x000500c2, 0x00000006, 0x000001ba, 0x000001b8, 0x000001b9,
    0x000500c7, 0x00000006, 0x000001bb, 0x000001ba, 0x00000032, 0x000500c4, 0x00000006, 0x000001bc,
    0x000001bb, 0x00000192, 0x000500c5, 0x00000006, 0x000001bd, 0x000001b7, 0x000001bc, 0x0004007c,
    0x00000014, 0x000001be, 0x000001bd, 0x00050082, 0x00000014, 0x000001bf, 0x000001be, 0x00000196,
    0x0004006f, 0x0000000d, 0x000001c0, 0x000001bf, 0x00050041, 0x000000b3, 0x000001c1, 0x000001b4,
    0x000001b5, 0x0003003e, 0x000001c1, 0x000001c0, 0x000200f9, 0x00000174, 0x000200f8, 0x00000174,
    0x0004003d, 0x00000006, 0x000001c2, 0x00000170, 0x00050080, 0x00000006, 0x000001c3, 0x000001c2,
    0x0000009b, 0x0003003e, 0x00000170, 0x000001c3, 0x000200f9, 0x00000171, 0x000200f8, 0x00000173,
    0x0004003d, 0x000000d6, 0x000001c4, 0x0000018b, 0x0004003d, 0x00000006, 0x000001c5, 0x0000014f,
    0x0004003d, 0x00000006, 0x000001c6, 0x0000015c, 0x00050080, 0x00000006, 0x000001c7, 0x000001c5,
    0x000001c6, 0x00050041, 0x000000e6, 0x000001c8, 0x000000da, 0x000001c7, 0x0004003d, 0x000000d6,
    0x000001c9, 0x000001c8, 0x00050094, 0x0000000d, 0x000001ca, 0x000001c4, 0x000001c9, 0x0004003d,
    0x0000000d, 0x000001cb, 0x00000158, 0x00050081, 0x0000000d, 0x000001cc, 0x000001cb, 0x000001ca,
    0x0003003e, 0x00000158, 0x000001cc, 0x0004003d, 0x000000d6, 0x000001cd, 0x0000019a, 0x0004003d,
    0x00000006, 0x000001ce, 0x0000014f, 0x00050080, 0x00000006, 0x000001cf, 0x000001ce, 0x00000070,
    0x0004003d, 0x00000006, 0x000001d0, 0x0000015c, 0x00050080, 0x00000006, 0x000001d1, 0x000001cf,
    0x000001d0, 0x00050041, 0x000000e6, 0x000001d2, 0x000000da, 0x000001d1, 0x0004003d, 0x000000d6,
    0x000001d3, 0x000001d2, 0x00050094, 0x0000000d, 0x000001d4, 0x000001cd, 0x000001d3, 0x0004003d,
    0x0000000d, 0x000001d5, 0x00000159, 0x00050081, 0x0000000d, 0x000001d6, 0x000001d5, 0x000001d4,
    0x0003003e, 0x00000159, 0x000001d6, 0x0004003d, 0x000000d6, 0x000001d7, 0x000001a7, 0x0004003d,
    0x00000006, 0x000001d8, 0x0000014f, 0x00050080, 0x00000006, 0x000001d9, 0x000001d8, 0x00000091,
    0x0004003d, 0x00000006, 0x000001da, 0x0000015c, 0x00050080, 0x00000006, 0x000001db, 0x000001d9,
    0x000001da, 0x00050041, 0x000000e6, 0x000001dc, 0x000000da, 0x000001db, 0x0004003d, 0x000000d6,
    0x000001dd, 0x000001dc, 0x00050094, 0x0000000d, 0x000001de, 0x000001d7, 0x000001dd, 0x0004003d,
    0x0000000d, 0x000001df, 0x0000015a, 0x00050081, 0x0000000d, 0x000001e0, 0x000001df, 0x000001de,
    0x0003003e, 0x0000015a, 0x000001e0, 0x0004003d, 0x000000d6, 0x000001e1, 0x000001b4, 0x0004003d,
    0x00000006, 0x000001e2, 0x0000014f, 0x00050080, 0x00000006, 0x000001e4, 0x000001e2, 0x000001e3,
    0x0004003d, 0x00000006, 0x000001e5, 0x0000015c, 0x00050080, 0x00000006, 0x000001e6, 0x000001e4,
    0x000001e5, 0x00050041, 0x000000e6, 0x000001e7, 0x000000da, 0x000001e6, 0x0004003d, 0x000000d6,
    0x000001e8, 0x000001e7, 0x00050094, 0x0000000d, 0x000001e9, 0x000001e1, 0x000001e8, 0x0004003d,
    0x0000000d, 0x000001ea, 0x0000015b, 0x00050081, 0x0000000d, 0x000001eb, 0x000001ea, 0x000001e9,
    0x0003003e, 0x0000015b, 0x000001eb, 0x000200f9, 0x00000160, 0x000200f8, 0x00000160, 0x0004003d,
    0x00000006, 0x000001ec, 0x0000015c, 0x00050080, 0x00000006, 0x000001ed, 0x000001ec, 0x0000009b,
    0x0003003e, 0x0000015c, 0x000001ed, 0x000200f9, 0x0000015d, 0x000200f8, 0x0000015f, 0x0004003d,
    0x0000000d, 0x000001ee, 0x00000141, 0x0004003d, 0x0000000d, 0x000001ef, 0x00000129, 0x0004003d,
    0x0000000d, 0x000001f0, 0x00000158, 0x00050085, 0x0000000d, 0x000001f1, 0x000001ef, 0x000001f0,
    0x0004003d, 0x0000000d, 0x000001f2, 0x0000012d, 0x0004003d, 0x0000000d, 0x000001f3, 0x00000159,
    0x00050085, 0x0000000d, 0x000001f4, 0x000001f2, 0x000001f3, 0x00050081, 0x0000000d, 0x000001f5,
    0x000001f1, 0x000001f4, 0x0004003d, 0x0000000d, 0x000001f6, 0x00000132, 0x0004003d, 0x0000000d,
    0x000001f7, 0x0000015a, 0x00050085, 0x0000000d, 0x000001f8, 0x000001f6, 0x000001f7, 0x00050081,
    0x0000000d, 0x000001f9, 0x000001f5, 0x000001f8, 0x0004003d, 0x0000000d, 0x000001fa, 0x00000137,
    0x0004003d, 0x0000000d, 0x000001fb, 0x0000015b, 0x00050085, 0x0000000d, 0x000001fc, 0x000001fa,
    0x000001fb, 0x00050081, 0x0000000d, 0x000001fd, 0x000001f9, 0x000001fc, 0x00050085, 0x0000000d,
    0x000001fe, 0x000001ee, 0x000001fd, 0x0004003d, 0x0000000d, 0x000001ff, 0x000000b4, 0x00050081,
    0x0000000d, 0x00000200, 0x000001ff, 0x000001fe, 0x0003003e, 0x000000b4, 0x00000200, 0x000200f9,
    0x000000f1, 0x000200f8, 0x000000f1, 0x0004003d, 0x00000006, 0x00000201, 0x000000ed, 0x00050080,
    0x00000006, 0x00000202, 0x00000201, 0x0000009b, 0x0003003e, 0x000000ed, 0x00000202, 0x000200f9,
    0x000000ee, 0x000200f8, 0x000000f0, 0x000200f9, 0x000000ec, 0x000200f8, 0x000000ec, 0x000200f9,
    0x000000ba, 0x000200f8, 0x000000ba, 0x0004003d, 0x00000006, 0x00000203, 0x000000b6, 0x00050080,
    0x00000006, 0x00000204, 0x00000203, 0x00000070, 0x0003003e, 0x000000b6, 0x00000204, 0x000200f9,
    0x000000b7, 0x000200f8, 0x000000b9, 0x0004003d, 0x00000006, 0x00000208, 0x0000007d, 0x0004003d,
    0x0000000d, 0x00000209, 0x000000b4, 0x00050041, 0x0000020a, 0x0000020b, 0x00000207, 0x00000208,
    0x0003003e, 0x0000020b, 0x00000209, 0x000400e0, 0x00000018, 0x00000018, 0x000000c4, 0x0004003d,
    0x00000006, 0x0000020c, 0x00000096, 0x000500aa, 0x0000001b, 0x0000020d, 0x0000020c, 0x0000001a,
    0x0004003d, 0x0000001b, 0x0000020e, 0x000000a1, 0x000500a7, 0x0000001b, 0x0000020f, 0x0000020d,
    0x0000020e, 0x000300f7, 0x00000211, 0x00000000, 0x000400fa, 0x0000020f, 0x00000210, 0x00000211,
    0x000200f8, 0x00000210, 0x0003003e, 0x00000212, 0x000000b5, 0x0003003e, 0x00000213, 0x0000001a,
    0x000200f9, 0x00000214, 0x000200f8, 0x00000214, 0x000400f6, 0x00000216, 0x00000217, 0x00000000,
    0x000200f9, 0x00000218, 0x000200f8, 0x00000218, 0x0004003d, 0x00000006, 0x00000219, 0x00000213,
    0x000500b0, 0x0000001b, 0x0000021a, 0x00000219, 0x0000004d, 0x000400fa, 0x0000021a, 0x00000215,
    0x00000216, 0x000200f8, 0x00000215, 0x0004003d, 0x00000006, 0x0000021b, 0x0000007d, 0x0004003d,
    0x00000006, 0x0000021c, 0x00000213, 0x00050080, 0x00000006, 0x0000021d, 0x0000021b, 0x0000021c,
    0x00050041, 0x0000020a, 0x0000021e, 0x00000207, 0x0000021d, 0x0004003d, 0x0000000d, 0x0000021f,
    0x0000021e, 0x0004003d, 0x0000000d, 0x00000220, 0x00000212, 0x00050081, 0x0000000d, 0x00000221,
    0x00000220, 0x0000021f, 0x0003003e, 0x00000212, 0x00000221, 0x000200f9, 0x00000217, 0x000200f8,
    0x00000217, 0x0004003d, 0x00000006, 0x00000222, 0x00000213, 0x00050080, 0x00000006, 0x00000223,
    0x00000222, 0x0000009b, 0x0003003e, 0x00000213, 0x00000223, 0x000200f9, 0x00000214, 0x000200f8,
    0x00000216, 0x0004003d, 0x00000006, 0x00000228, 0x0000008f, 0x0004003d, 0x0000000d, 0x00000229,
    0x00000212, 0x00060041, 0x0000022a, 0x0000022b, 0x00000227, 0x00000023, 0x00000228, 0x0003003e,
    0x0000022b, 0x00000229, 0x000200f9, 0x00000211, 0x000200f8, 0x00000211, 0x000100fd, 0x00010038,
    0x00050036, 0x00000008, 0x0000000b, 0x00000000, 0x00000009, 0x00030037, 0x00000007, 0x0000000a,
    0x000200f8, 0x0000000c, 0x0004003b, 0x00000007, 0x00000012, 0x00000007, 0x0004003b, 0x00000007,
    0x00000038, 0x00000007, 0x0004003b, 0x00000007, 0x0000003c, 0x00000007, 0x0004003b, 0x00000007,
    0x00000041, 0x00000007, 0x0004003b, 0x00000007, 0x00000046, 0x00000007, 0x0004003b, 0x00000007,
    0x0000004b, 0x00000007, 0x0004003d, 0x00000006, 0x00000013, 0x0000000a, 0x000500c2, 0x00000006,
    0x00000016, 0x00000013, 0x00000015, 0x0003003e, 0x00000012, 0x00000016, 0x0004003d, 0x00000006,
    0x00000017, 0x0000000a, 0x000500c7, 0x00000006, 0x00000019, 0x00000017, 0x00000018, 0x000500aa,
    0x0000001b, 0x0000001c, 0x00000019, 0x0000001a, 0x000300f7, 0x0000001e, 0x00000000, 0x000400fa,
    0x0000001c, 0x0000001d, 0x0000001e, 0x000200f8, 0x0000001d, 0x0004003d, 0x00000006, 0x00000024,
    0x00000012, 0x00060041, 0x00000025, 0x00000026, 0x00000022, 0x00000023, 0x00000024, 0x0004003d,
    0x00000006, 0x00000027, 0x00000026, 0x0004003d, 0x00000006, 0x00000028, 0x00000012, 0x00050080,
    0x00000006, 0x0000002a, 0x00000028, 0x00000029, 0x00060041, 0x00000025, 0x0000002b, 0x00000022,
    0x00000023, 0x0000002a, 0x0004003d, 0x00000006, 0x0000002c, 0x0000002b, 0x0004003d, 0x00000006,
    0x0000002d, 0x00000012, 0x00050080, 0x00000006, 0x0000002e, 0x0000002d, 0x00000018, 0x00060041,
    0x00000025, 0x0000002f, 0x00000022, 0x00000023, 0x0000002e, 0x0004003d, 0x00000006, 0x00000030,
    0x0000002f, 0x0004003d, 0x00000006, 0x00000031, 0x00000012, 0x00050080, 0x00000006, 0x00000033,
    0x00000031, 0x00000032, 0x00060041, 0x00000025, 0x00000034, 0x00000022, 0x00000023, 0x00000033,
    0x0004003d, 0x00000006, 0x00000035, 0x00000034, 0x00070050, 0x00000008, 0x00000036, 0x00000027,
    0x0000002c, 0x00000030, 0x00000035, 0x000200fe, 0x00000036, 0x000200f8, 0x0000001e, 0x0004003d,
    0x00000006, 0x00000039, 0x00000012, 0x00060041, 0x00000025, 0x0000003a, 0x00000022, 0x00000023,
    0x00000039, 0x0004003d, 0x00000006, 0x0000003b, 0x0000003a, 0x0003003e, 0x00000038, 0x0000003b,
    0x0004003d, 0x00000006, 0x0000003d, 0x00000012, 0x00050080, 0x00000006, 0x0000003e, 0x0000003d,
    0x00000029, 0x00060041, 0x00000025, 0x0000003f, 0x00000022, 0x00000023, 0x0000003e, 0x0004003d,
    0x00000006, 0x00000040, 0x0000003f, 0x0003003e, 0x0000003c, 0x00000040, 0x0004003d, 0x00000006,
    0x00000042, 0x00000012, 0x00050080, 0x00000006, 0x00000043, 0x00000042, 0x00000018, 0x00060041,
    0x00000025, 0x00000044, 0x00000022, 0x00000023, 0x00000043, 0x0004003d, 0x00000006, 0x00000045,
    0x00000044, 0x0003003e, 0x00000041, 0x00000045, 0x0004003d, 0x00000006, 0x00000047, 0x00000012,
    0x00050080, 0x00000006, 0x00000048, 0x00000047, 0x00000032, 0x00060041, 0x00000025, 0x00000049,
    0x00000022, 0x00000023, 0x00000048, 0x0004003d, 0x00000006, 0x0000004a, 0x00000049, 0x0003003e,
    0x00000046, 0x0000004a, 0x0004003d, 0x00000006, 0x0000004c, 0x00000012, 0x00050080, 0x00000006,
    0x0000004e, 0x0000004c, 0x0000004d, 0x00060041, 0x00000025, 0x0000004f, 0x00000022, 0x00000023,
    0x0000004e, 0x0004003d, 0x00000006, 0x00000050, 0x0000004f, 0x0003003e, 0x0000004b, 0x00000050,
    0x0004003d, 0x00000006, 0x00000051, 0x00000038, 0x000500c2, 0x00000006, 0x00000053, 0x00000051,
    0x00000052, 0x0004003d, 0x00000006, 0x00000054, 0x0000003c, 0x000500c4, 0x00000006, 0x00000055,
    0x00000054, 0x00000052, 0x000500c5, 0x00000006, 0x00000056, 0x00000053, 0x00000055, 0x0004003d,
    0x00000006, 0x00000057, 0x0000003c, 0x000500c2, 0x00000006, 0x00000058, 0x00000057, 0x00000052,
    0x0004003d, 0x00000006, 0x00000059, 0x00000041, 0x000500c4, 0x00000006, 0x0000005a, 0x00000059,
    0x00000052, 0x000500c5, 0x00000006, 0x0000005b, 0x00000058, 0x0000005a, 0x0004003d, 0x00000006,
    0x0000005c, 0x00000041, 0x000500c2, 0x00000006, 0x0000005d, 0x0000005c, 0x00000052, 0x0004003d,
    0x00000006, 0x0000005e, 0x00000046, 0x000500c4, 0x00000006, 0x0000005f, 0x0000005e, 0x00000052,
    0x000500c5, 0x00000006, 0x00000060, 0x0000005d, 0x0000005f, 0x0004003d, 0x00000006, 0x00000061,
    0x00000046, 0x000500c2, 0x00000006, 0x00000062, 0x00000061, 0x00000052, 0x0004003d, 0x00000006,
    0x00000063, 0x0000004b, 0x000500c4, 0x00000006, 0x00000064, 0x00000063, 0x00000052, 0x000500c5,
    0x00000006, 0x00000065, 0x00000062, 0x00000064, 0x00070050, 0x00000008, 0x00000066, 0x00000056,
    0x0000005b, 0x00000060, 0x00000065, 0x000200fe, 0x00000066, 0x00010038, 0x00050036, 0x0000000d,
    0x00000010, 0x00000000, 0x0000000e, 0x00030037, 0x00000007, 0x0000000f, 0x000200f8, 0x00000011,
    0x0004003b, 0x00000007, 0x00000069, 0x00000007, 0x0004003d, 0x00000006, 0x0000006a, 0x0000000f,
    0x000500c2, 0x00000006, 0x0000006b, 0x0000006a, 0x00000015, 0x00060041, 0x00000025, 0x0000006c,
    0x00000022, 0x00000023, 0x0000006b, 0x0004003d, 0x00000006, 0x0000006d, 0x0000006c, 0x0004003d,
    0x00000006, 0x0000006e, 0x0000000f, 0x000500c7, 0x00000006, 0x0000006f, 0x0000006e, 0x00000032,
    0x00050084, 0x00000006, 0x00000071, 0x0000006f, 0x00000070, 0x000500c2, 0x00000006, 0x00000072,
    0x0000006d, 0x00000071, 0x000500c7, 0x00000006, 0x00000074, 0x00000072, 0x00000073, 0x0003003e,
    0x00000069, 0x00000074, 0x0004003d, 0x00000006, 0x00000075, 0x00000069, 0x000500c4, 0x00000006,
    0x00000077, 0x00000075, 0x00000076, 0x0004007c, 0x00000014, 0x00000078, 0x00000077, 0x000500c3,
    0x00000014, 0x00000079, 0x00000078, 0x00000076, 0x0004006f, 0x0000000d, 0x0000007a, 0x00000079,
    0x000200fe, 0x0000007a, 0x00010038,
};
const size_t roaster_spv_q6k_v2_size = sizeof(roaster_spv_q6k_v2);
const uint32_t roaster_spv_silu_v1[] = {
    0x07230203, 0x00010300, 0x0008000b, 0x00000063, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00786469, 0x00080005, 0x0000000b,
    0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00040005, 0x00000011,
    0x61726150, 0x0000736d, 0x00050006, 0x00000011, 0x00000000, 0x65746e69, 0x00000072, 0x00040006,
    0x00000011, 0x00000001, 0x0000006e, 0x00050006, 0x00000011, 0x00000002, 0x69727473, 0x00006564,
    0x00030005, 0x00000013, 0x00000070, 0x00030005, 0x00000022, 0x00000065, 0x00030005, 0x00000027,
    0x0000006a, 0x00030005, 0x00000030, 0x00000067, 0x00060005, 0x00000032, 0x65746147, 0x75427055,
    0x72656666, 0x00000000, 0x00050006, 0x00000032, 0x00000000, 0x61746164, 0x00000000, 0x00030005,
    0x00000034, 0x00007567, 0x00030005, 0x00000041, 0x00000075, 0x00050005, 0x0000004e, 0x75426845,
    0x72656666, 0x00000000, 0x00050006, 0x0000004e, 0x00000000, 0x61746164, 0x00000000, 0x00030005,
    0x00000050, 0x00006865, 0x00040047, 0x0000000b, 0x0000000b, 0x0000001c, 0x00030047, 0x00000011,
    0x00000002, 0x00050048, 0x00000011, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000011,
    0x00000001, 0x00000023, 0x00000004, 0x00050048, 0x00000011, 0x00000002, 0x00000023, 0x00000008,
    0x00040047, 0x00000031, 0x00000006, 0x00000004, 0x00030047, 0x00000032, 0x00000002, 0x00040048,
    0x00000032, 0x00000000, 0x00000018, 0x00050048, 0x00000032, 0x00000000, 0x00000023, 0x00000000,
    0x00030047, 0x00000034, 0x00000018, 0x00040047, 0x00000034, 0x00000021, 0x00000000, 0x00040047,
    0x00000034, 0x00000022, 0x00000000, 0x00040047, 0x0000004d, 0x00000006, 0x00000004, 0x00030047,
    0x0000004e, 0x00000002, 0x00040048, 0x0000004e, 0x00000000, 0x00000019, 0x00050048, 0x0000004e,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000050, 0x00000019, 0x00040047, 0x00000050,
    0x00000021, 0x00000002, 0x00040047, 0x00000050, 0x00000022, 0x00000000, 0x00040047, 0x00000062,
    0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015,
    0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017,
    0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009, 0x0004003b,
    0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000, 0x00040020,
    0x0000000d, 0x00000001, 0x00000006, 0x0005001e, 0x00000011, 0x00000006, 0x00000006, 0x00000006,
    0x00040020, 0x00000012, 0x00000009, 0x00000011, 0x0004003b, 0x00000012, 0x00000013, 0x00000009,
    0x00040015, 0x00000014, 0x00000020, 0x00000001, 0x0004002b, 0x00000014, 0x00000015, 0x00000001,
    0x00040020, 0x00000016, 0x00000009, 0x00000006, 0x0004002b, 0x00000014, 0x00000019, 0x00000000,
    0x00020014, 0x0000001d, 0x00030016, 0x0000002e, 0x00000020, 0x00040020, 0x0000002f, 0x00000007,
    0x0000002e, 0x0003001d, 0x00000031, 0x0000002e, 0x0003001e, 0x00000032, 0x00000031, 0x00040020,
    0x00000033, 0x0000000c, 0x00000032, 0x0004003b, 0x00000033, 0x00000034, 0x0000000c, 0x0004002b,
    0x00000006, 0x00000035, 0x00000002, 0x0004002b, 0x00000014, 0x00000038, 0x00000002, 0x00040020,
    0x0000003e, 0x0000000c, 0x0000002e, 0x0004002b, 0x00000006, 0x00000044, 0x00000001, 0x0003001d,
    0x0000004d, 0x0000002e, 0x0003001e, 0x0000004e, 0x0000004d, 0x00040020, 0x0000004f, 0x0000000c,
    0x0000004e, 0x0004003b, 0x0000004f, 0x00000050, 0x0000000c, 0x0004002b, 0x0000002e, 0x00000058,
    0x3f800000, 0x0004002b, 0x00000006, 0x00000061, 0x00000040, 0x0006002c, 0x00000009, 0x00000062,
    0x00000061, 0x00000044, 0x00000044, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0004003b, 0x00000007,
    0x00000022, 0x00000007, 0x0004003b, 0x00000007, 0x00000027, 0x00000007, 0x0004003b, 0x0000002f,
    0x00000030, 0x00000007, 0x0004003b, 0x0000002f, 0x00000041, 0x00000007, 0x00050041, 0x0000000d,
    0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x0003003e,
    0x00000008, 0x0000000f, 0x0004003d, 0x00000006, 0x00000010, 0x00000008, 0x00050041, 0x00000016,
    0x00000017, 0x00000013, 0x00000015, 0x0004003d, 0x00000006, 0x00000018, 0x00000017, 0x00050041,
    0x00000016, 0x0000001a, 0x00000013, 0x00000019, 0x0004003d, 0x00000006, 0x0000001b, 0x0000001a,
    0x00050084, 0x00000006, 0x0000001c, 0x00000018, 0x0000001b, 0x000500ae, 0x0000001d, 0x0000001e,
    0x00000010, 0x0000001c, 0x000300f7, 0x00000020, 0x00000000, 0x000400fa, 0x0000001e, 0x0000001f,
    0x00000020, 0x000200f8, 0x0000001f, 0x000100fd, 0x000200f8, 0x00000020, 0x0004003d, 0x00000006,
    0x00000023, 0x00000008, 0x00050041, 0x00000016, 0x00000024, 0x00000013, 0x00000019, 0x0004003d,
    0x00000006, 0x00000025, 0x00000024, 0x00050086, 0x00000006, 0x00000026, 0x00000023, 0x00000025,
    0x0003003e, 0x00000022, 0x00000026, 0x0004003d, 0x00000006, 0x00000028, 0x00000008, 0x0004003d,
    0x00000006, 0x00000029, 0x00000022, 0x00050041, 0x00000016, 0x0000002a, 0x00000013, 0x00000019,
    0x0004003d, 0x00000006, 0x0000002b, 0x0000002a, 0x00050084, 0x00000006, 0x0000002c, 0x00000029,
    0x0000002b, 0x00050082, 0x00000006, 0x0000002d, 0x00000028, 0x0000002c, 0x0003003e, 0x00000027,
    0x0000002d, 0x0004003d, 0x00000006, 0x00000036, 0x00000022, 0x00050084, 0x00000006, 0x00000037,
    0x00000035, 0x00000036, 0x00050041, 0x00000016, 0x00000039, 0x00000013, 0x00000038, 0x0004003d,
    0x00000006, 0x0000003a, 0x00000039, 0x00050084, 0x00000006, 0x0000003b, 0x00000037, 0x0000003a,
    0x0004003d, 0x00000006, 0x0000003c, 0x00000027, 0x00050080, 0x00000006, 0x0000003d, 0x0000003b,
    0x0000003c, 0x00060041, 0x0000003e, 0x0000003f, 0x00000034, 0x00000019, 0x0000003d, 0x0004003d,
    0x0000002e, 0x00000040, 0x0000003f, 0x0003003e, 0x00000030, 0x00000040, 0x0004003d, 0x00000006,
    0x00000042, 0x00000022, 0x00050084, 0x00000006, 0x00000043, 0x00000035, 0x00000042, 0x00050080,
    0x00000006, 0x00000045, 0x00000043, 0x00000044, 0x00050041, 0x00000016, 0x00000046, 0x00000013,
    0x00000038, 0x0004003d, 0x00000006, 0x00000047, 0x00000046, 0x00050084, 0x00000006, 0x00000048,
    0x00000045, 0x00000047, 0x0004003d, 0x00000006, 0x00000049, 0x00000027, 0x00050080, 0x00000006,
    0x0000004a, 0x00000048, 0x00000049, 0x00060041, 0x0000003e, 0x0000004b, 0x00000034, 0x00000019,
    0x0000004a, 0x0004003d, 0x0000002e, 0x0000004c, 0x0000004b, 0x0003003e, 0x00000041, 0x0000004c,
    0x0004003d, 0x00000006, 0x00000051, 0x00000022, 0x00050041, 0x00000016, 0x00000052, 0x00000013,
    0x00000038, 0x0004003d, 0x00000006, 0x00000053, 0x00000052, 0x00050084, 0x00000006, 0x00000054,
    0x00000051, 0x00000053, 0x0004003d, 0x00000006, 0x00000055, 0x00000027, 0x00050080, 0x00000006,
    0x00000056, 0x00000054, 0x00000055, 0x0004003d, 0x0000002e, 0x00000057, 0x00000030, 0x0004003d,
    0x0000002e, 0x00000059, 0x00000030, 0x0004007f, 0x0000002e, 0x0000005a, 0x00000059, 0x0006000c,
    0x0000002e, 0x0000005b, 0x00000001, 0x0000001b, 0x0000005a, 0x00050081, 0x0000002e, 0x0000005c,
    0x00000058, 0x0000005b, 0x00050088, 0x0000002e, 0x0000005d, 0x00000057, 0x0000005c, 0x0004003d,
    0x0000002e, 0x0000005e, 0x00000041, 0x00050085, 0x0000002e, 0x0000005f, 0x0000005d, 0x0000005e,
    0x00060041, 0x0000003e, 0x00000060, 0x00000050, 0x00000019, 0x00000056, 0x0003003e, 0x00000060,
    0x0000005f, 0x000100fd, 0x00010038,
};
const size_t roaster_spv_silu_v1_size = sizeof(roaster_spv_silu_v1);
/* END EMBEDDED SPIRV */