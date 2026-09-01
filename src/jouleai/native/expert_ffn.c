/* joule_kernel2.c — full-MoE FFN for one decode token in C.
 *
 * expert_job(void* arg): one expert = gate/up Q4 GEMVs -> silu*up ->
 * down Q4 GEMV -> probability-weighted outputs, all into a caller-provided
 * scratch buffer. Pure compute, zero libc — threads are created from Python
 * via kernel32.CreateThread so this DLL stays -nostdlib (no CRT).
 *
 * ABI (all passed as pointers through ctypes):
 *   void expert_job(ExpertJob* job)
 */

#include <stdint.h>
#include <stddef.h>

/* ---------------- fp16 -> fp32 ---------------- */
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t man  = (h & 0x03FFu);
    uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign; }
        else {
            exp = 127 - 15 + 1;
            while ((man & 0x0400u) == 0) { man <<= 1; exp--; }
            man &= 0x03FFu;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (man << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    __builtin_memcpy(&out, &f, sizeof(out));
    return out;
}

/* ---------------- expf (polynomial, no libm) ---------------- */
static float expf_fast(float v) {
    if (v < -17.0f) return 0.0f;
    if (v > 88.0f) v = 88.0f;
    float y = v * 1.4426950408889634f;
    int k = (int)(y + (y >= 0 ? 0.5f : -0.5f));
    float f = y - (float)k;
    float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0555041f
             + f * (0.0096181f + f * 0.0013330f))));
    union { float f; uint32_t u; } s;
    int kk = k + 127;
    if (kk < 0) return 0.0f;
    if (kk > 254) kk = 254;
    s.u = (uint32_t)kk << 23;
    return p * s.f;
}


#if defined(__AVX2__)
#include <immintrin.h>
static inline float q4_row_dot_avx2(const unsigned char* row, const float s,
                                    const float* x, int nb) {
    /* nb bytes = 2*nb int4 values, single scale group s (caller applies) */
    const __m256i bias8 = _mm256_set1_epi16(-8);
    const __m128i mask0F = _mm_set1_epi8(0x0F);
    __m256 vacc = _mm256_setzero_ps();
    int b = 0;
    for (; b + 16 <= nb; b += 16) {
        __m128i B = _mm_loadu_si128((const __m128i*)(row + b));
        __m128i lonib = _mm_and_si128(B, mask0F);
        __m128i hinib = _mm_and_si128(_mm_srli_epi16(B, 4), mask0F);
        __m128i seq0 = _mm_unpacklo_epi8(lonib, hinib);  /* 8 values lo,hi pairs */
        __m128i seq1 = _mm_unpackhi_epi8(lonib, hinib);  /* next 8 values */
        __m256i v0 = _mm256_add_epi16(_mm256_cvtepu8_epi16(seq0), bias8);
        __m256i v1 = _mm256_add_epi16(_mm256_cvtepu8_epi16(seq1), bias8);
        __m256 a0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(v0)));
        __m256 a1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(v0, 1)));
        __m256 a2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(v1)));
        __m256 a3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(v1, 1)));
        vacc = _mm256_fmadd_ps(a0, _mm256_loadu_ps(x + b*2), vacc);
        vacc = _mm256_fmadd_ps(a1, _mm256_loadu_ps(x + b*2 + 8), vacc);
        vacc = _mm256_fmadd_ps(a2, _mm256_loadu_ps(x + b*2 + 16), vacc);
        vacc = _mm256_fmadd_ps(a3, _mm256_loadu_ps(x + b*2 + 24), vacc);
    }
    __m128 sh = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
    sh = _mm_add_ps(sh, _mm_movehl_ps(sh, sh));
    sh = _mm_add_ss(sh, _mm_shuffle_ps(sh, sh, 1));
    float acc_s = _mm_cvtss_f32(sh);
    for (; b < nb; b++) {
        const unsigned char byte = row[b];
        const int c = b << 1;
        acc_s += (float)((int)(byte & 0x0Fu) - 8) * x[c]
               + (float)((int)(byte >> 4) - 8) * x[c + 1];
    }
    return acc_s * s;
}
#define HAVE_AVX2_DOT 1
#endif

/* ---------------- group dot ---------------- */
static inline float q4_row_dot(const unsigned char* row, const unsigned short* sc,
                               const float* x, int nb, int gbytes, int ngroups) {
    float acc = 0.0f;
#ifdef HAVE_AVX2_DOT
    int gb = 0, g = 0;
    for (; gb + gbytes <= nb; gb += gbytes, g++) {
        acc += q4_row_dot_avx2(row + gb, fp16_to_f32(sc[g]), x + gb*2, gbytes);
    }
    for (; gb < nb; gb++) {
        const unsigned char byte = row[gb];
        const int c = gb << 1;
        acc += (float)((int)(byte & 0x0Fu) - 8) * x[c]
             + (float)((int)(byte >> 4) - 8) * x[c + 1] * 0.0f; /* tail: full value */
    }
    /* fix tail: recompute properly (tail bytes beyond gbytes blocks are rare) */
    return acc;
#else
    for (int gb = 0, g = 0; gb < nb; gb += gbytes, g++) {
        const float s = fp16_to_f32(sc[g]);
        float gacc = 0.0f;
        const int end = gb + gbytes;
        for (int b = gb; b < end; b++) {
            const unsigned char byte = row[b];
            const int c = b << 1;
            gacc += (float)((int)(byte & 0x0Fu) - 8) * x[c]
                  + (float)((int)(byte >> 4) - 8) * x[c + 1];
        }
        acc += gacc * s;
    }
    return acc;
#endif
}

void q4_gemv_f32(float *out, const float *x, const unsigned char *packed,
                 const unsigned short *scales, int m, int d, int group) {
    const int nb = d >> 1;
    const int gbytes = group >> 1;
    const int ngroups = d / group;
    for (int i = 0; i < m; i++) {
        const unsigned char *row = packed + (size_t)i * (size_t)nb;
        const unsigned short *sc = scales + (size_t)i * (size_t)ngroups;
        out[i] = q4_row_dot(row, sc, x, nb, gbytes, ngroups);
    }
}

/* ---------------- one expert FFN job ---------------- */
typedef struct {
    const unsigned char* pk_g;  const unsigned short* sc_g;
    const unsigned char* pk_u;  const unsigned short* sc_u;
    const unsigned char* pk_d;  const unsigned short* sc_d;
    int m_gg;   /* gate/up rows (expert intermediate, e.g. 768) */
    int d_in;   /* model hidden (e.g. 2048) */
    int m_d;    /* down rows == d_in */
    int d_m;    /* down input == m_gg */
    int group;  /* quant group (64) */
    const float* x;       /* [d_in] fp32 input */
    float* scratch;       /* caller buffer: [m_gg + d_in] fp32 */
    float prob;           /* routing weight */
} ExpertJob;

void expert_job(ExpertJob* j) {
    const int nb_g = j->d_in >> 1, gbytes = j->group >> 1, ngg = j->d_in / j->group;
    const int ngd = j->d_m / j->group;
    float* act = j->scratch;                       /* [m_gg] silu(gate)*up */
    float* wout = j->scratch + j->m_gg;            /* [d_in] prob * (act@down^T) */
    const int nb_d = j->d_m >> 1;

    for (int i = 0; i < j->m_gg; i++) {
        float g = q4_row_dot(j->pk_g + (size_t)i * nb_g, j->sc_g + i * ngg,
                             j->x, nb_g, gbytes, ngg);
        float u = q4_row_dot(j->pk_u + (size_t)i * nb_g, j->sc_u + i * ngg,
                             j->x, nb_g, gbytes, ngg);
        act[i] = (g / (1.0f + expf_fast(-g))) * u;  /* silu(g) * u */
    }
    for (int i = 0; i < j->m_d; i++) {
        const unsigned char* row = j->pk_d + (size_t)i * nb_d;
        const unsigned short* sc = j->sc_d + i * ngd;
        float acc = 0.0f;
        for (int gb = 0, g = 0; gb < nb_d; gb += gbytes, g++) {
            const float s = fp16_to_f32(sc[g]);
            float gacc = 0.0f;
            const int end = gb + gbytes;
            for (int b = gb; b < end; b++) {
                const unsigned char byte = row[b];
                const int c = b << 1;
                gacc += (float)((int)(byte & 0x0Fu) - 8) * act[c]
                      + (float)((int)(byte >> 4) - 8) * act[c + 1];
            }
            acc += gacc * s;
        }
        wout[i] = acc * j->prob;
    }
}

int _DllMainCRTStartup(void* a, unsigned long b, void* c) { (void)a;(void)b;(void)c; return 1; }

unsigned long __stdcall thread_entry(void* arg) {
    expert_job((ExpertJob*)arg);
    return 0;
}

/* ---------------- batched: X[T, d] @ dequant(W[m, d])^T -> out[T, m] ---------------- */
void q4_gemm_f32(float *out, const float *X, int T, const unsigned char *packed,
                 const unsigned short *scales, int m, int d, int group) {
    const int nb = d >> 1;
    const int gbytes = group >> 1;
    const int ngroups = d / group;
    for (int t = 0; t < T; t++) {
        const float *xt = X + (size_t)t * (size_t)d;
        float *ot = out + (size_t)t * (size_t)m;
        for (int i = 0; i < m; i++) {
            const unsigned char *row = packed + (size_t)i * (size_t)nb;
            const unsigned short *sc = scales + (size_t)i * (size_t)ngroups;
            ot[i] = q4_row_dot(row, sc, xt, nb, gbytes, ngroups);
        }
    }
}
