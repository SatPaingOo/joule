/* joule_kernel3.c — full decode-token kernel (48 layers, one call/token).
 *
 * Removes ALL Python per-layer overhead: route, expert FFN (fused Q4 GEMV),
 * attention (QK-norm + RoPE + SDPA), residual norms, and the lm_head GEMV
 * all happen inside C. Weights are resident bf16 in RAM; experts are the
 * packed-Q4 records (also resident). One call per generated token.
 *
 * BATCH path (decode_layers_batch): B sequences decode together; every
 * weight row is read ONCE and dotted against all B activation vectors, so
 * the aggregate tok/s scales with B (weights read once per B tokens).
 * All batched GEMMs run on a persistent thread pool with a ggml-exact
 * spin barrier (fixed worker ids, main participates, monotonic counters) —
 * no per-op CreateThread.
 *
 * Build (zig, nostdlib): zig cc -O3 -mcpu=native -I. -nostdlib -shared \
 *     decode_kernel.c -o decode_kernel.dll -lkernel32
 */

#include <stdint.h>
#include <stddef.h>
static int g_max_layers = -1;  /* -1 = all from cfg */
static volatile int g_trace = 0;
static volatile float g_h_norm = 0.0f;
static volatile float g_ffn_in_norm = 0.0f;
float h_norm_get(void) { return g_h_norm; }
float ffn_in_norm_get(void) { return g_ffn_in_norm; }
void trace_set(int v) { g_trace = v; }
int trace_get(void) { return g_trace; }
void lm_head_parallel(const float* W, const float* x, float* out, int m, int d);
static void matvec_f32_par(float* out, const float* W, const float* x, int m, int d, int nthreads);
static void matvec_f32_B(float* out, const unsigned short* W, const float* x,
                         int m, int d, int B);
void* __stdcall GetModuleHandleA(const char*);
void* __stdcall GetProcAddress(void*, const char*);
void* __stdcall CreateThread(void*, size_t, unsigned long (__stdcall*)(void*), void*, unsigned long, unsigned long*);
unsigned long __stdcall WaitForMultipleObjects(unsigned long, void* const*, int, unsigned long);
void __stdcall SetThreadAffinityMask(void*, unsigned long);
void* malloc(size_t n);          /* defined at the bottom (kernel32 heap) */
void free(void* p);
void* memset(void* d, int c, unsigned long n);
void* memcpy(void* d, const void* s, unsigned long n);

/* max batch size the workspace supports */
#define BMAX 16

/* ---------------- bf16 <-> f32 ---------------- */
static inline float bf16_to_f32(uint16_t h) {
    uint32_t f = ((uint32_t)(h & 0x8000u)) << 16
               | ((uint32_t)(h & 0x7FFFu)) << 16;
    union { uint32_t u; float f; } c;
    c.u = f;
    return c.f;
}
/* load 8 bf16 -> 8 fp32 (bf16 << 16 = fp32 bits, no rounding needed) */
#if defined(__AVX2__)
#include <immintrin.h>
static inline __m256 load_bf16_ps(const unsigned short* p) {
    __m128i lo = _mm_loadu_si128((const __m128i*)p);   /* 8 bf16 */
    __m256i v = _mm256_cvtepu16_epi32(lo);             /* 8 i32 */
    v = _mm256_slli_epi32(v, 16);                      /* bf16 -> fp32 bits */
    return _mm256_castsi256_ps(v);
}
#endif
static inline float fp16_to_f32(uint16_t h) {
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
    union { uint32_t u; float f; } c;
    c.u = f;
    return c.f;
}

/* ---------------- exp (no libm) ---------------- */
static inline float sqrtf_fast(float x) {
    union { float f; uint32_t u; } v = { x };
    v.u = (v.u + 0x3F800000u) >> 1;   /* initial guess */
    v.f = 0.5f * (v.f + x / v.f);      /* one Newton step */
    return v.f;
}
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

/* ---------------- Q4 row dot (group-64, fp16 scales) ---------------- */

#if defined(__AVX2__)
#include <immintrin.h>
static inline float q4_row_dot(const unsigned char* row, const unsigned short* sc,
                               const float* x, int nb, int gbytes, int ngroups) {
    float acc = 0.0f;
    for (int gb = 0, g = 0; gb + gbytes <= nb; gb += gbytes, g++) {
        const float s = fp16_to_f32(sc[g]);
        const __m256i bias8 = _mm256_set1_epi16(-8);
        const __m128i mask0F = _mm_set1_epi8(0x0F);
        __m256 vacc = _mm256_setzero_ps();
        int b = gb;
        for (; b + 16 <= gb + gbytes; b += 16) {
            __m128i B = _mm_loadu_si128((const __m128i*)(row + b));
            __m128i lonib = _mm_and_si128(B, mask0F);
            __m128i hinib = _mm_and_si128(_mm_srli_epi16(B, 4), mask0F);
            __m128i seq0 = _mm_unpacklo_epi8(lonib, hinib);
            __m128i seq1 = _mm_unpackhi_epi8(lonib, hinib);
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
        acc += _mm_cvtss_f32(sh) * s;
    }
    return acc;
}
#else
static inline float q4_row_dot(const unsigned char* row, const unsigned short* sc,
                               const float* x, int nb, int gbytes, int ngroups) {
    float acc = 0.0f;
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
}
#endif

/* ---------------- int8 matvec (AVX-512 VNNI, Q8_0-style) ----------------
 * Fixed weights quantized to int8 (per-row scale). The fp32 activation is
 * quantized to int8 on-the-fly (per-tensor scale — hidden-state magnitude is
 * stable), then vpmaddubsw (int8 x int8 -> int16) + vpmaddwd (-> int32) +
 * dequant. ~3-5x faster than fp32 matvec (multiply in hardware on 8-bit
 * lanes). Dispatch: compiled under __AVX512VNNI__, runtime CPU-feature check
 * picks int8 vs the bf16/fp32 fallback.
 */
#if defined(__AVX512VNNI__)
#include <immintrin.h>
/* forward decl — quantize_x_i8 is defined below but used by matvec_i8_B first */
static inline void quantize_x_i8(const float* x, signed char* xq, int d, float* sx_out);

/* per-row int8 matvec: out[m] = dequant_i8(W[m,d]) @ x[d]
 * W stored as UNSIGNED int8 [m,d] (biased +128); Ws fp32 scale [m];
 * x fp32 [d] quantized to signed int8 on the fly (per-tensor scale).
 * vpmaddubsw does u8 x s8, so the weight bias must be subtracted:
 *   dot = dpbusd(Wu, xq) - 128 * sum(xq)   (per row, same sum for all rows)
 */
void matvec_i8_vnni(float* out, const unsigned char* W, const float* Ws,
                    const float* x, int m, int d) {
    float xmax = 1e-12f;
    for (int j = 0; j < d; j++) { float a = x[j] < 0 ? -x[j] : x[j]; if (a > xmax) xmax = a; }
    const float sx = xmax / 127.0f;
    __m512 xs = _mm512_set1_ps(sx);
    signed char xq[16384];
    int j = 0;
    for (; j + 16 <= d; j += 16) {
        __m512 xv = _mm512_loadu_ps(x + j);
        __m512 xr = _mm512_roundscale_ps(_mm512_div_ps(xv, xs), 0);
        __m512i xi = _mm512_cvtps_epi32(xr);
        __m128i x8 = _mm512_cvtepi32_epi8(xi);   /* 16 signed int8 */
        _mm_storeu_si128((__m128i*)(xq + j), x8);
    }
    for (; j < d; j++) xq[j] = (signed char)(x[j] / sx);
    /* sum(xq) once (for the bias correction) */
    int32_t xsum = 0;
    for (j = 0; j < d; j++) xsum += xq[j];
    const int d8 = (d + 63) / 64;
    __m512i acc = _mm512_setzero_si512();
    const __m512i bias128 = _mm512_set1_epi32(128 * 64);  /* 128 * 64 elems per 512b */
    (void)bias128;
    for (int i = 0; i < m; i++) {
        const __m512i* w = (const __m512i*)(W + (size_t)i * d);
        const __m512i* a = (const __m512i*)xq;
        acc = _mm512_setzero_si512();
        for (int k = 0; k < d8; k++) {
            acc = _mm512_dpbusd_epi32(acc, w[k], a[k]);
        }
        int32_t dot = _mm512_reduce_add_epi32(acc);
        dot -= 128 * xsum;   /* un-bias the unsigned weights */
        out[i] = (float)dot * sx * Ws[i];
    }
}
#endif

/* ---------------- RMSNorm ---------------- */
static inline void rms_norm(float* out, const float* x, const unsigned short* w,
                            int d, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf_fast(ss / d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * bf16_to_f32(w[i]);
}

/* ---------------- bf16 matvec: out[m] = W[m,d] @ x[d] ---------------- */
/* W is bf16 (uint16); dequantized in registers (half the bandwidth of fp32) */
static void matvec_f32(float* out, const unsigned short* W, const float* x,
                        int m, int d) {
    for (int i = 0; i < m; i++) {
        const unsigned short* row = W + (size_t)i * d;
        float acc = 0.0f;
#if defined(__AVX2__)
        __m256 vacc = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= d; j += 8) {
            vacc = _mm256_fmadd_ps(load_bf16_ps(row + j),
                                   _mm256_loadu_ps(x + j), vacc);
        }
        __m128 sh = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
        sh = _mm_add_ps(sh, _mm_movehl_ps(sh, sh));
        sh = _mm_add_ss(sh, _mm_shuffle_ps(sh, sh, 1));
        acc = _mm_cvtss_f32(sh);
        for (; j < d; j++) acc += bf16_to_f32(row[j]) * x[j];
#else
        for (int j = 0; j < d; j++) acc += bf16_to_f32(row[j]) * x[j];
#endif
        out[i] = acc;
    }
}

/* ---------------- RoPE (rotate-half convention, matches GenericStreamer) ---- */
static inline void rope_apply(float* q, const float* cos, const float* sin, int hd) {
    const int h = hd >> 1;
    for (int i = 0; i < h; i++) {
        float a = q[i], b = q[i + h];
        q[i]     = a * cos[i] - b * sin[i];
        q[i + h] = a * sin[i] + b * cos[i];
    }
}

/* ---------------- config ---------------- */
typedef struct {
    int L, d, n_heads, n_kv, hd, E, topk, V;
    float eps;
    int maxT;
    int intermediate;
    int qk_norm_type;    /* 0 none | 1 per-head (qwen3, after head view, w=[hd])
                            | 2 whole (olmoe, before head view, w=[d]) */
    int bias_qkv;        /* 1: QKV have fp32 biases (qwen2 family) */
    int norm_topk_prob;  /* 1: renormalize top-k routing weights (qwen3_moe);
                            0: raw softmax weights (olmoe, mixtral) */
    int expert_i8;       /* 1: expert rows are int8 (+128 bias, per-row fp32
                            scale, Q8_0) — the VNNI FFN path (quality + speed);
                            0: Q4 nibbles (group-64 fp16 scales) */
    int expert_bf16;     /* 1: expert rows are bf16 (uint16) — the exact tier
                            (64-token identity; no quantization drift). The
                            Q4/i8 pointers are ignored; expert_pk[] holds
                            bf16 row pointers instead. */
} KernelCfg;

typedef struct {
    const unsigned short* embed;    /* [V, d]  bf16 */
    const unsigned short* lm_head;  /* [V, d]  fp32 (kept fp32: dequant ALU) */
    const unsigned short* final_norm; /* [d]    bf16 */
    const unsigned short* norm1;    /* [L, d]  bf16 */
    const unsigned short* norm2;    /* [L, d]  bf16 */
    const unsigned short* wq;       /* [L, H*hd, d] bf16 */
    const unsigned short* wk;
    const unsigned short* wv;
    const unsigned short* wo;
    const float* bq;                /* [L, H*hd] fp32 QKV bias (qwen2) */
    const float* bk;                /* [L, KH*hd] fp32 */
    const float* bv;                /* [L, KH*hd] fp32 */
    const unsigned short* qn;       /* [L, hd] per-head | [L, H*hd] whole (bf16) */
    const unsigned short* kn;       /* [L, hd] per-head | [L, KH*hd] whole (bf16) */
    const unsigned short* gate_w;   /* [L, E, d] bf16 (MoE router) */
    const unsigned short* w1;       /* [L, m, d] dense gate_proj (bf16) */
    const unsigned short* w2;       /* [L, m, d] dense up_proj (bf16) */
    const unsigned short* w3;       /* [L, d, m] dense down_proj (bf16) */
    /* int8 Q8_0 variants (precision=int8): weight + per-row fp32 scale */
    const unsigned char* wq_i8;
    const float* wq_i8s;
    const unsigned char* wk_i8;
    const float* wk_i8s;
    const unsigned char* wv_i8;
    const float* wv_i8s;
    const unsigned char* wo_i8;
    const float* wo_i8s;
    const unsigned char* lm_i8;     /* [V, d] int8 lm_head */
    const float* lm_i8s;            /* [V] per-row scale */
    /* experts Q4: for each layer l, expert e, part p in (gate,up,down):
       packed = expert_pk[p][l*E + e]; scales = expert_sc[p][l*E + e] */
    const unsigned char* const* expert_pk[3];
    const unsigned short* const* expert_sc[3];
    const float* cos;               /* [maxT, hd] fp32 (computed) */
    const float* sin;
    int use_i8;                     /* nonzero: attention uses int8 QKV */
} KernelW;

/* ---------------- KV cache (per layer) ---------------- */
typedef struct {
    float* k;  /* [maxT, n_kv, hd] */
    float* v;
} KVCache;

/* ---------------- batch workspace (heap, sized from KernelCfg) ---------
 * Each buffer is a SEPARATE malloc'd array sized from the model's actual
 * dimensions (d, hd, H, KH, E, topk, intermediate, BMAX). No offset math —
 * avoids the aliasing bugs a single flat buffer introduced in the Entry 55
 * refactor. The allocation is cached by a shape signature: a model switch
 * (new NativeDecoder, /v1/model/<name>) changes dims -> realloc. This is
 * what makes the kernel run ANY model shape without static caps (the old
 * fixed arrays overflowed on m>12288, e.g. Mistral-7B/Qwen2.5-7B). */
#define BMAX 16
#define UMAX 512

typedef struct {
    float* h; float* tmp; float* h2;      /* [B][d] */
    float* q; float* k; float* v; float* att;  /* [B][H*hd] / [B][KH*hd] */
    float* act; float* y;                 /* [U][B][m] / [U][B][d] */
    float* scores;                        /* [B][E] */
    int* top; float* tw;                  /* [B][topk] */
    int* sel; int* uidx; float* uw; int* uh;  /* union maps */
    signed char* xq; signed char* aq;     /* int8 expert path: quantized x/act
                                             [B][max(d,m)] per worker slot */
    size_t bytes;
} BatchWS;

static BatchWS g_ws;   /* serialized by the session pool — one decode at a time */
static long g_ws_sig = 0;
static int g_E = 0;    /* runtime expert count / intermediate (q4_dot_test) */
static int g_m = 0;

static long ws_sig(const KernelCfg* c) {
    long h = 1469598103934665603LL;
    const int vals[7] = { c->d, c->hd, c->n_heads, c->n_kv, c->E, c->topk,
                          c->intermediate };
    for (int i = 0; i < 7; i++) { h ^= vals[i]; h *= 1099511628211LL; }
    return h;
}

static void* ws_malloc(size_t n) { return malloc(n ? n : 1); }

static void ws_free_all(void) {
    free(g_ws.h); free(g_ws.tmp); free(g_ws.h2);
    free(g_ws.q); free(g_ws.k); free(g_ws.v); free(g_ws.att);
    free(g_ws.act); free(g_ws.y);
    free(g_ws.scores); free(g_ws.top); free(g_ws.tw);
    free(g_ws.sel); free(g_ws.uidx); free(g_ws.uw); free(g_ws.uh);
    free(g_ws.xq); free(g_ws.aq);
    memset(&g_ws, 0, sizeof g_ws);
    g_ws_sig = 0;
}

static int ws_init(const KernelCfg* c, int B) {
    (void)B;                              /* sized for BMAX, not the call's B */
    const int d = c->d, hd = c->hd, H = c->n_heads, KH = c->n_kv;
    const int E = c->E, topk = c->topk, m = c->intermediate;
    int U = (E > 0 ? E : 1);
    if (U > UMAX) U = UMAX;
    if (topk < 0 || topk > 16) return 0;  /* top/tw/uidx/uw sized for <=16 */
    const int kk = (topk > 0 ? topk : 1); /* dense (E==0): routing unused */
    if (d < 1 || d > 65536 || m < 1 || m > 262144) return 0;  /* sanity */
    const int Bw = BMAX;
    const int hhd = H * hd, khd = KH * hd;
    long sig = ws_sig(c);
    if (sig == g_ws_sig) return 1;        /* already sized for this shape */
    ws_free_all();
    g_ws.h   = (float*)ws_malloc((size_t)Bw * d * sizeof(float));
    g_ws.tmp = (float*)ws_malloc((size_t)Bw * d * sizeof(float));
    g_ws.h2  = (float*)ws_malloc((size_t)Bw * d * sizeof(float));
    g_ws.q   = (float*)ws_malloc((size_t)Bw * hhd * sizeof(float));
    g_ws.k   = (float*)ws_malloc((size_t)Bw * khd * sizeof(float));
    g_ws.v   = (float*)ws_malloc((size_t)Bw * khd * sizeof(float));
    g_ws.att = (float*)ws_malloc((size_t)Bw * hhd * sizeof(float));
    g_ws.act = (float*)ws_malloc((size_t)U * Bw * m * sizeof(float));
    g_ws.y   = (float*)ws_malloc((size_t)U * Bw * d * sizeof(float));
    g_ws.scores = (float*)ws_malloc((size_t)Bw * E * sizeof(float));
    g_ws.top = (int*)ws_malloc((size_t)Bw * kk * sizeof(int));
    g_ws.tw  = (float*)ws_malloc((size_t)Bw * kk * sizeof(float));
    g_ws.sel = (int*)ws_malloc((size_t)UMAX * sizeof(int));
    g_ws.uidx = (int*)ws_malloc((size_t)Bw * kk * sizeof(int));
    g_ws.uw  = (float*)ws_malloc((size_t)Bw * kk * sizeof(float));
    g_ws.uh  = (int*)ws_malloc((size_t)(UMAX + 1) * sizeof(int));
    g_ws.xq = (signed char*)ws_malloc((size_t)Bw * (d > m ? d : m));
    g_ws.aq = (signed char*)ws_malloc((size_t)(Bw * 17) * (d > m ? d : m));
    if (!g_ws.h || !g_ws.tmp || !g_ws.h2 || !g_ws.q || !g_ws.k || !g_ws.v ||
        !g_ws.att || !g_ws.act || !g_ws.y || !g_ws.scores || !g_ws.top ||
        !g_ws.tw || !g_ws.sel || !g_ws.uidx || !g_ws.uw || !g_ws.uh ||
        !g_ws.xq || !g_ws.aq) {
        ws_free_all();
        return 0;                         /* OOM — callers bail */
    }
    g_ws_sig = sig;
    g_E = E; g_m = m;                     /* runtime shape for q4_dot_test */
    return 1;
}

/* ---------------- forward helpers ---------------- */
static void layer_attn(const KernelCfg* c, const KernelW* W, KVCache* kv,
                       int l, int pos, const float* x, float* out) {
    const int d = c->d, hd = c->hd, H = c->n_heads, KH = c->n_kv;
    g_trace = 1;
    if (!ws_init(c, 1)) return;
    /* per-layer scratch from the shared workspace (serialized) */
    float* q = g_ws.q; float* k = g_ws.k; float* v = g_ws.v;
    float* att = g_ws.att;
    g_trace = 2;
#if defined(__AVX512VNNI__)
    if (W->use_i8) {
        matvec_i8_vnni(q, W->wq_i8 + (size_t)l * (H * hd) * d,
                       W->wq_i8s + (size_t)l * (H * hd), x, H * hd, d);
        matvec_i8_vnni(k, W->wk_i8 + (size_t)l * (KH * hd) * d,
                       W->wk_i8s + (size_t)l * (KH * hd), x, KH * hd, d);
        matvec_i8_vnni(v, W->wv_i8 + (size_t)l * (KH * hd) * d,
                       W->wv_i8s + (size_t)l * (KH * hd), x, KH * hd, d);
    } else
#endif
    {
        matvec_f32_B(q, W->wq + (size_t)l * (size_t)(H * hd) * d, x, H * hd, d, 1);
        matvec_f32_B(k, W->wk + (size_t)l * (size_t)(KH * hd) * d, x, KH * hd, d, 1);
        matvec_f32_B(v, W->wv + (size_t)l * (size_t)(KH * hd) * d, x, KH * hd, d, 1);
    }
    /* QKV bias (qwen2 family) */
    if (c->bias_qkv) {
        const float* bq = W->bq + (size_t)l * (H * hd);
        const float* bk = W->bk + (size_t)l * (KH * hd);
        const float* bv = W->bv + (size_t)l * (KH * hd);
        for (int i = 0; i < H * hd; i++) q[i] += bq[i];
        for (int i = 0; i < KH * hd; i++) { k[i] += bk[i]; v[i] += bv[i]; }
    }
    /* whole-vector QK-norm (olmoe): RMS over q (H*hd) / k (KH*hd) BEFORE the
     * head view; weights [L, vec]. Applied instead of per-head, not with it. */
    if (c->qk_norm_type == 2) {
        const unsigned short* wnq = W->qn + (size_t)l * (H * hd);
        const unsigned short* wnk = W->kn + (size_t)l * (KH * hd);
        float ss = 0; for (int i = 0; i < H * hd; i++) ss += q[i] * q[i];
        float inv = 1.0f / sqrtf_fast(ss / (H * hd) + c->eps);
        for (int i = 0; i < H * hd; i++) q[i] = q[i] * inv * bf16_to_f32(wnq[i]);
        ss = 0; for (int i = 0; i < KH * hd; i++) ss += k[i] * k[i];
        inv = 1.0f / sqrtf_fast(ss / (KH * hd) + c->eps);
        for (int i = 0; i < KH * hd; i++) k[i] = k[i] * inv * bf16_to_f32(wnk[i]);
    }
    /* per-head QK-norm (RMS over hd) — only for per_head models (qwen3) */
    if (c->qk_norm_type == 1) {
    for (int h = 0; h < H; h++) {
        float* qh = q + (size_t)h * hd;
        const unsigned short* wn = W->qn + (size_t)l * hd;
        float ss = 0; for (int i = 0; i < hd; i++) ss += qh[i] * qh[i];
        float inv = 1.0f / sqrtf_fast(ss / hd + c->eps);
        for (int i = 0; i < hd; i++) qh[i] = qh[i] * inv * bf16_to_f32(wn[i]);
    }
    for (int h = 0; h < KH; h++) {
        float* kh = k + (size_t)h * hd;
        const unsigned short* wn = W->kn + (size_t)l * hd;
        float ss = 0; for (int i = 0; i < hd; i++) ss += kh[i] * kh[i];
        float inv = 1.0f / sqrtf_fast(ss / hd + c->eps);
        for (int i = 0; i < hd; i++) kh[i] = kh[i] * inv * bf16_to_f32(wn[i]);
    }
    }
    /* RoPE */
    const float* cos = W->cos + (size_t)pos * hd;
    const float* sin = W->sin + (size_t)pos * hd;
    for (int h = 0; h < H; h++) rope_apply(q + (size_t)h * hd, cos, sin, hd);
    for (int h = 0; h < KH; h++) rope_apply(k + (size_t)h * hd, cos, sin, hd);
    /* KV cache append */
    float* kc = kv[l].k + (size_t)pos * KH * hd;
    float* vc = kv[l].v + (size_t)pos * KH * hd;
    for (int i = 0; i < KH * hd; i++) { kc[i] = k[i]; vc[i] = v[i]; }
    int T = pos + 1;
    /* SDPA: per q-head (scores allocated ONCE per call) */
    static float scores[65536];   /* static: no stack/heap churn (T<=64k) */
    for (int h = 0; h < H; h++) {
        const float* qh = q + (size_t)h * hd;
        int kh = h / (H / KH);
        const float* kk = kv[l].k + (size_t)kh * hd;
        float smax = -1e30f;
        for (int t = 0; t < T; t++) {
            const float* kt = kk + (size_t)t * KH * hd;
            float s = 0; for (int i = 0; i < hd; i++) s += qh[i] * kt[i];
            scores[t] = s / sqrtf_fast((float)hd);
            if (scores[t] > smax) smax = scores[t];
        }
        float sum = 0;
        for (int t = 0; t < T; t++) { scores[t] = expf_fast(scores[t] - smax); sum += scores[t]; }
        float* oh = att + (size_t)h * hd;
        for (int i = 0; i < hd; i++) oh[i] = 0;
        for (int t = 0; t < T; t++) {
            const float* vt = kv[l].v + (size_t)t * KH * hd + (size_t)kh * hd;
            float p = scores[t] / sum;
            for (int i = 0; i < hd; i++) oh[i] += p * vt[i];
        }
    }
    /* o_proj: [d, H*hd] */
#if defined(__AVX512VNNI__)
    if (W->use_i8) {
        matvec_i8_vnni(out, W->wo_i8 + (size_t)l * d * (H * hd),
                       W->wo_i8s + (size_t)l * d, att, d, H * hd);
    } else
#endif
        matvec_f32_B(out, W->wo + (size_t)l * (size_t)d * (H * hd), att, d, H * hd, 1);
    g_trace = 3;
}

/* ================================================================
 * Persistent thread pool + ggml-exact spin barrier
 *
 * Ported from ggml (src/ggml-cpu/ggml-cpu.c, ggml_barrier):
 *   - workers get FIXED ids at init (loop index) — never arrival order
 *   - main thread participates in every barrier (it is participant #n_workers)
 *   - barrier: fetch_add(n_barrier); last arriver resets n_barrier=0 and
 *     bumps n_barrier_passed; everyone else spins on n_barrier_passed change
 *   - n_barrier_passed is monotonic (no reset race)
 * Work is published before bumping `gen` (seq-cst), so workers that see the
 * new gen are guaranteed to see the published fields (release/acquire pair).
 * Workers spin briefly, then WaitOnAddress (futex) so idle cores sleep.
 * ================================================================ */

static inline void cpu_pause(void) { __asm__ volatile("pause" ::: "memory"); }

typedef unsigned long (__stdcall *WaitOnAddressFn)(volatile void*, void*, size_t, unsigned long);
typedef unsigned long (__stdcall *WakeByAddressAllFn)(volatile void*);

typedef struct {
    volatile long n_barrier;          /* barrier generation (ggml: fetch_add) */
    volatile long n_barrier_passed;   /* monotonic arrival count */
    volatile long gen;                /* work generation (ggml: n_graph) */
    volatile long shutdown;
    long n_workers;                   /* worker threads (main is +1 participant) */
    long n_participants;              /* n_workers + 1 */
    void (*work_fn)(void*, long, long, long);
    void* work_ctx;
    long work_lo, work_hi;
    volatile int initialized;
    WaitOnAddressFn wait_on_addr;
    WakeByAddressAllFn wake_by_addr;
} SpinPool;

#define MAX_POOL_WORKERS 16
typedef struct {
    SpinPool* p;
    long id;                    /* fixed at init: 0..n_workers-1 */
    long last_gen;
} WorkerCtx;

static SpinPool g_pool;
static WorkerCtx g_workers[MAX_POOL_WORKERS];
static void* g_handles[MAX_POOL_WORKERS];

/* ggml_barrier port — called by workers AND main (n_threads = participants) */
static inline void pool_barrier(SpinPool* p, int n_threads) {
    if (n_threads <= 1) return;
    long n_passed = __atomic_load_n(&p->n_barrier_passed, __ATOMIC_RELAXED);
    /* enter barrier (full seq-cst fence) */
    long nb = __atomic_fetch_add(&p->n_barrier, 1, __ATOMIC_SEQ_CST);
    if (nb == n_threads - 1) {
        /* last thread to arrive */
        __atomic_store_n(&p->n_barrier, 0, __ATOMIC_RELAXED);
        /* exit barrier (full seq-cst fence) */
        __atomic_fetch_add(&p->n_barrier_passed, 1, __ATOMIC_SEQ_CST);
        return;
    }
    /* wait for other threads */
    while (__atomic_load_n(&p->n_barrier_passed, __ATOMIC_RELAXED) == n_passed) {
        cpu_pause();
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static unsigned long __stdcall pool_worker(void* arg) {
    WorkerCtx* wc = (WorkerCtx*)arg;
    SpinPool* p = wc->p;
    const long id = wc->id;
    for (;;) {
        /* wait for new work (gen change), spin then futex-sleep */
        long g;
        for (int spins = 0; ; spins++) {
            if (__atomic_load_n(&p->shutdown, __ATOMIC_SEQ_CST)) return 0;
            g = __atomic_load_n(&p->gen, __ATOMIC_SEQ_CST);
            if (g != __atomic_load_n(&wc->last_gen, __ATOMIC_SEQ_CST)) break;
            if (spins < 2000) { cpu_pause(); continue; }
            /* sleep until gen changes (no lost wakeup: WaitOnAddress
             * re-reads the value; if gen already changed it returns) */
            if (p->wait_on_addr) {
                long waitval = g;
                p->wait_on_addr(&p->gen, &waitval, sizeof(long), 0xFFFFFFFFu);
            } else {
                cpu_pause();
            }
            spins = 0;
        }
        __atomic_store_n(&wc->last_gen, g, __ATOMIC_SEQ_CST);
        if (__atomic_load_n(&p->shutdown, __ATOMIC_SEQ_CST)) return 0;
        /* compute my chunk: fixed id, participants = n_workers + 1 */
        long span = p->work_hi - p->work_lo;
        long lo = p->work_lo + span * id / p->n_participants;
        long hi = p->work_lo + span * (id + 1) / p->n_participants;
        p->work_fn(p->work_ctx, id, lo, hi);
        pool_barrier(p, (int)p->n_participants);
    }
    return 0;
}

int spin_pool_init(int n_workers) {
    if (g_pool.initialized) return 1;
    if (n_workers < 1) n_workers = 1;
    if (n_workers > MAX_POOL_WORKERS) n_workers = MAX_POOL_WORKERS;
    void* mod = GetModuleHandleA("kernel32.dll");
    if (!mod) return 0;
    g_pool.n_workers = n_workers;
    g_pool.n_participants = n_workers + 1;
    g_pool.n_barrier = 0;
    g_pool.n_barrier_passed = 0;
    g_pool.gen = 0;
    g_pool.shutdown = 0;
    g_pool.initialized = 0;
    g_pool.wait_on_addr = (WaitOnAddressFn)GetProcAddress(mod, "WaitOnAddress");
    g_pool.wake_by_addr = (WakeByAddressAllFn)GetProcAddress(mod, "WakeByAddressAll");
    for (long i = 0; i < n_workers; i++) {
        g_workers[i].p = &g_pool;
        g_workers[i].id = i;
        g_workers[i].last_gen = 0;
        void* h = CreateThread(0, 0, pool_worker, &g_workers[i], 0, 0);
        if (!h) return 0;
        g_handles[i] = h;
    }
    /* ensure all workers have started and are waiting on gen=0 */
    __asm__ volatile("" ::: "memory");
    g_pool.initialized = 1;
    return 1;
}

void spin_pool_run(void (*fn)(void*, long, long, long), void* ctx, long lo, long hi) {
    if (!g_pool.initialized || g_pool.n_participants <= 1) { fn(ctx, 0, lo, hi); return; }
    /* publish work BEFORE bumping gen (seq-cst store = release) */
    g_pool.work_fn = fn;
    g_pool.work_ctx = ctx;
    g_pool.work_lo = lo;
    g_pool.work_hi = hi;
    __atomic_fetch_add(&g_pool.gen, 1, __ATOMIC_SEQ_CST);
    if (g_pool.wake_by_addr) g_pool.wake_by_addr(&g_pool.gen);
    /* main thread participates (participant id = n_workers) */
    long span = hi - lo;
    long mlo = lo + span * g_pool.n_workers / g_pool.n_participants;
    long mhi = hi;
    fn(ctx, g_pool.n_workers, mlo, mhi);
    pool_barrier(&g_pool, (int)g_pool.n_participants);
}

void spin_pool_shutdown(void) {
    g_pool.shutdown = 1;
    __atomic_fetch_add(&g_pool.gen, 1, __ATOMIC_SEQ_CST);
    if (g_pool.wake_by_addr) g_pool.wake_by_addr(&g_pool.gen);
}

/* ================================================================
 * BATCHED ops — every weight row is read ONCE, dotted against B x's.
 * ================================================================ */

/* fp32 batched matvec: out[B*m] = W[m,d] @ x[B*d]^T (per row, per b) */
typedef struct {
    const unsigned short* W; const float* x; float* out;
    int m, d, B;
} MV_BJob;

static void matvec_B_worker(void* ctx, long id, long lo, long hi) {
    (void)id;
    MV_BJob* j = (MV_BJob*)ctx;
    for (long i = lo; i < hi; i++) {
        const unsigned short* row = j->W + i * (size_t)j->d;
        for (int b = 0; b < j->B; b++) {
            const float* xb = j->x + b * (size_t)j->d;
            float acc = 0.0f;
#if defined(__AVX2__)
            __m256 vacc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= j->d; k += 8)
                vacc = _mm256_fmadd_ps(load_bf16_ps(row + k),
                                       _mm256_loadu_ps(xb + k), vacc);
            __m128 sh = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
            sh = _mm_add_ps(sh, _mm_movehl_ps(sh, sh));
            sh = _mm_add_ss(sh, _mm_shuffle_ps(sh, sh, 1));
            acc = _mm_cvtss_f32(sh);
            for (; k < j->d; k++) acc += bf16_to_f32(row[k]) * xb[k];
#else
            for (int k = 0; k < j->d; k++) acc += bf16_to_f32(row[k]) * xb[k];
#endif
            j->out[b * (size_t)j->m + i] = acc;
        }
    }
}

static void matvec_f32_B(float* out, const unsigned short* W, const float* x,
                         int m, int d, int B) {
    if (B <= 1) { matvec_f32(out, W, x, m, d); return; }
    MV_BJob j = { W, x, out, m, d, B };
    spin_pool_run(matvec_B_worker, &j, 0, m);
}

/* batched int8 matvec (AVX-512 VNNI): out[B*m] = dequant_i8(W[m,d]) @ x[B,d]^T.
 * Each sequence's x is quantized once (per-tensor scale), then all m rows are
 * dotted with vpmaddubsw. */
#if defined(__AVX512VNNI__)
typedef struct {
    const unsigned char* W; const float* Ws; const float* x; float* out;
    int m, d, B;
} MVI8BJob;

static void matvec_i8_B_worker(void* ctx, long id, long lo, long hi) {
    (void)id;
    MVI8BJob* j = (MVI8BJob*)ctx;
    const int d8 = (j->d + 63) / 64;
    signed char xq[16384];
    for (int b = 0; b < j->B; b++) {
        float sx;
        quantize_x_i8(j->x + b * (size_t)j->d, xq, j->d, &sx);
        int32_t xsum = 0;
        for (int t = 0; t < j->d; t++) xsum += xq[t];
        const __m512i* a = (const __m512i*)xq;
        for (long i = lo; i < hi; i++) {
            const __m512i* w = (const __m512i*)(j->W + i * (size_t)j->d);
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < d8; k++) acc = _mm512_dpbusd_epi32(acc, w[k], a[k]);
            int32_t dot = _mm512_reduce_add_epi32(acc);
            dot -= 128 * xsum;
            j->out[b * (size_t)j->m + i] = (float)dot * sx * j->Ws[i];
        }
    }
}

static void matvec_i8_B(float* out, const unsigned char* W, const float* Ws,
                        const float* x, int m, int d, int B) {
    MVI8BJob j = { W, Ws, x, out, m, d, B };
    spin_pool_run(matvec_i8_B_worker, &j, 0, m);
}
#endif

/* Q4 batched row dot: unpack the q4 row ONCE (per group), then dot with
 * each of B activation vectors. acc[B] filled. `stride` = x row stride
 * (d for gate/up, intermediate for down). Matches q4_row_dot math exactly
 * (same group order, same bias/scale application) so results are
 * bit-identical to the single-stream path. */
static inline void q4_row_dot_B(const unsigned char* row, const unsigned short* sc,
                                const float* x, int B, float* acc,
                                int nb, int gbytes, int ngroups, int stride) {
    for (int b = 0; b < B; b++) acc[b] = 0.0f;
#if defined(__AVX2__)
    const __m256i bias8 = _mm256_set1_epi16(-8);
    const __m128i mask0F = _mm_set1_epi8(0x0F);
    for (int gb = 0, g = 0; gb + gbytes <= nb; gb += gbytes, g++) {
        const float s = fp16_to_f32(sc[g]);
        int32_t vals[64];
        /* unpack group bytes -> int32 values, ONCE per group */
        for (int chunk = 0; chunk < gbytes; chunk += 16) {
            __m128i Bv = _mm_loadu_si128((const __m128i*)(row + gb + chunk));
            __m128i lonib = _mm_and_si128(Bv, mask0F);
            __m128i hinib = _mm_and_si128(_mm_srli_epi16(Bv, 4), mask0F);
            __m128i seq0 = _mm_unpacklo_epi8(lonib, hinib);
            __m128i seq1 = _mm_unpackhi_epi8(lonib, hinib);
            __m256i w0 = _mm256_add_epi16(_mm256_cvtepu8_epi16(seq0), bias8);
            __m256i w1 = _mm256_add_epi16(_mm256_cvtepu8_epi16(seq1), bias8);
            __m256i i0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w0));
            __m256i i1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w0, 1));
            __m256i i2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w1));
            __m256i i3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w1, 1));
            int32_t* dst = vals + (chunk / 16) * 32;
            _mm256_storeu_si256((__m256i*)(dst + 0), i0);
            _mm256_storeu_si256((__m256i*)(dst + 8), i1);
            _mm256_storeu_si256((__m256i*)(dst + 16), i2);
            _mm256_storeu_si256((__m256i*)(dst + 24), i3);
        }
        /* convert int32 -> float ONCE per group (not per b) */
        __m256 a0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(vals + 0)));
        __m256 a1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(vals + 8)));
        __m256 a2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(vals + 16)));
        __m256 a3 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(vals + 24)));
        __m256 a4 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(vals + 32)));
        __m256 a5 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(vals + 40)));
        __m256 a6 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(vals + 48)));
        __m256 a7 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(vals + 56)));
        for (int b = 0; b < B; b++) {
            const float* xg = x + (size_t)b * stride + gb * 2;
            __m256 vacc = _mm256_setzero_ps();
            vacc = _mm256_fmadd_ps(a0, _mm256_loadu_ps(xg + 0), vacc);
            vacc = _mm256_fmadd_ps(a1, _mm256_loadu_ps(xg + 8), vacc);
            vacc = _mm256_fmadd_ps(a2, _mm256_loadu_ps(xg + 16), vacc);
            vacc = _mm256_fmadd_ps(a3, _mm256_loadu_ps(xg + 24), vacc);
            vacc = _mm256_fmadd_ps(a4, _mm256_loadu_ps(xg + 32), vacc);
            vacc = _mm256_fmadd_ps(a5, _mm256_loadu_ps(xg + 40), vacc);
            vacc = _mm256_fmadd_ps(a6, _mm256_loadu_ps(xg + 48), vacc);
            vacc = _mm256_fmadd_ps(a7, _mm256_loadu_ps(xg + 56), vacc);
            __m128 sh = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
            sh = _mm_add_ps(sh, _mm_movehl_ps(sh, sh));
            sh = _mm_add_ss(sh, _mm_shuffle_ps(sh, sh, 1));
            acc[b] += _mm_cvtss_f32(sh) * s;
        }
    }
    /* tail groups (nb not a multiple of gbytes — rare) */
    for (int gb = (nb / gbytes) * gbytes, g = nb / gbytes; gb < nb; gb += gbytes, g++) {
        const float s = fp16_to_f32(sc[g]);
        const int end = gb + gbytes;
        for (int b = 0; b < B; b++) {
            const float* xb = x + (size_t)b * stride;
            float ga = 0.0f;
            for (int by = gb; by < end && by < nb; by++) {
                const unsigned char byte = row[by];
                const int cc = by << 1;
                ga += (float)((int)(byte & 0x0Fu) - 8) * xb[cc]
                    + (float)((int)(byte >> 4) - 8) * xb[cc + 1];
            }
            acc[b] += ga * s;
        }
    }
#else
    for (int gb = 0, g = 0; gb < nb; gb += gbytes, g++) {
        const float s = fp16_to_f32(sc[g]);
        const int end = gb + gbytes;
        for (int b = 0; b < B; b++) {
            const float* xb = x + (size_t)b * stride;
            float ga = 0.0f;
            for (int by = gb; by < end; by++) {
                const unsigned char byte = row[by];
                const int cc = by << 1;
                ga += (float)((int)(byte & 0x0Fu) - 8) * xb[cc]
                    + (float)((int)(byte >> 4) - 8) * xb[cc + 1];
            }
            acc[b] += ga * s;
        }
    }
#endif
}

/* ---------------- Q4 row dot, BATCHED, AVX-512 VNNI ----------------
 * The Q4 store holds nibbles already biased (+8): byte b = [v[2b] | v[2b+1]]
 * each 0..15 representing int4 -8..7. vpmaddubsw does u8 x s8, so:
 *   dot_q4 = dpbusd(q4_u8, xq) - 8 * sum(xq)   (per group, then * group scale)
 * where xq is the fp32 activation quantized to int8 (per-tensor scale).
 * xq/sx are PRE-QUANTIZED by the caller ONCE per expert (all rows share the
 * same activation) — no per-row re-quantization.
 */
#if defined(__AVX512VNNI__)
static inline void q4_row_dot_B_q(const signed char* xq, const float* sx,
                                  const int32_t* xsum_group,  /* [ngroups] per b? no: [B][ngroups] */
                                  const unsigned char* row, const unsigned short* sc,
                                  int B, float* acc,
                                  int nb, int gbytes, int ngroups, int stride) {    for (int b = 0; b < B; b++) acc[b] = 0.0f;
    const __m512i mask0F = _mm512_set1_epi32(0x0F0F0F0Fu);
    for (int gb = 0, g = 0; gb + gbytes <= nb; gb += gbytes, g++) {
        const float s = fp16_to_f32(sc[g]);
        __m256i Bv = _mm256_loadu_si256((const __m256i*)(row + gb));
        __m256i lo = _mm256_and_si256(Bv, _mm512_castsi512_si256(mask0F));
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(Bv, 4),
                                      _mm512_castsi512_si256(mask0F));
        __m256i u0 = _mm256_unpacklo_epi8(lo, hi);
        __m256i u1 = _mm256_unpackhi_epi8(lo, hi);
        __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(u0), u1, 1);
        for (int b = 0; b < B; b++) {
            const __m512i* ab = (const __m512i*)(xq + b * stride + gb * 2);
            __m512i accv = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, ab[0]);
            int32_t dot = _mm512_reduce_add_epi32(accv);
            dot -= 8 * xsum_group[b * ngroups + g];   /* precomputed bias */
            acc[b] += (float)dot * sx[b] * s;
        }
    }
}
#endif

/* ---------------- int8 expert row-dot (Q8_0 experts, VNNI) ----------------
 * Expert rows are int8 (unsigned +128 bias), one fp32 scale PER ROW (not per
 * group). The activation is quantized once per sequence (per-tensor scale),
 * then vpmaddubsw (u8 x s8 -> i32). The per-row scale makes the Q8 store's
 * quantization error ~1.5% vs Q4's ~11% (rms) — the Entry 61 long-gen drift
 * fix. At B>=2 the per-row unpack is amortized (the FFN's speed win). */
static inline void q8_row_dot_B(const unsigned char* row, const float sc,
                                const signed char* xq, const float* sx,
                                const int32_t* xsum,
                                int B, float* acc, int d, int stride) {
#if defined(__AVX512VNNI__)
    const int d8 = (d + 63) / 64;
    for (int b = 0; b < B; b++) {
        const __m512i* w = (const __m512i*)row;
        const __m512i* a = (const __m512i*)(xq + (size_t)b * stride);
        __m512i accv = _mm512_setzero_si512();
        for (int k = 0; k < d8; k++)
            accv = _mm512_dpbusd_epi32(accv,
                                       _mm512_loadu_si512((const void*)(w + k)),
                                       _mm512_loadu_si512((const void*)(a + k)));
        int32_t dot = _mm512_reduce_add_epi32(accv);
        acc[b] = (float)(dot - 128 * xsum[b]) * sx[b] * sc;
    }
#else
    (void)xsum;
    for (int b = 0; b < B; b++) {
        const signed char* xb = xq + (size_t)b * stride;
        int32_t dot = 0;
        for (int kk = 0; kk < d; kk++)
            dot += (int32_t)((int)row[kk] - 128) * (int32_t)xb[kk];
        acc[b] = (float)dot * sx[b] * sc;
    }
#endif
}


/* batched bf16 row dot: out[b] = row[d] @ x[b*stride ..]. Exact tier (no
 * quantization) — the 64-token-identity path. Uses the same AVX2 bf16 fma
 * as matvec_B_worker, so results match the fp32 reference to bf16 rounding. */
static inline void bf16_row_dot_B(const unsigned short* row,
                                  const float* x, int B, float* acc,
                                  int d, int stride) {
    for (int b = 0; b < B; b++) acc[b] = 0.0f;
#if defined(__AVX2__)
    const __m256i* r = (const __m256i*)row;
    for (int b = 0; b < B; b++) {
        const float* xb = x + (size_t)b * stride;
        __m256 vacc = _mm256_setzero_ps();
        int k = 0;
        for (; k + 8 <= d; k += 8)
            vacc = _mm256_fmadd_ps(load_bf16_ps(row + k),
                                   _mm256_loadu_ps(xb + k), vacc);
        __m128 sh = _mm_add_ps(_mm256_castps256_ps128(vacc),
                               _mm256_extractf128_ps(vacc, 1));
        sh = _mm_add_ps(sh, _mm_movehl_ps(sh, sh));
        sh = _mm_add_ss(sh, _mm_shuffle_ps(sh, sh, 1));
        acc[b] = _mm_cvtss_f32(sh);
        for (; k < d; k++) acc[b] += bf16_to_f32(row[k]) * xb[k];
    }
#else
    for (int b = 0; b < B; b++) {
        const float* xb = x + (size_t)b * stride;
        float a = 0.0f;
        for (int k = 0; k < d; k++) a += bf16_to_f32(row[k]) * xb[k];
        acc[b] = a;
    }
#endif
}

/* ---------------- batched lm_head: logits[B*V] = lm_head[V,d] @ x[B,d]^T -- */
typedef struct {
    const float* W; const float* x; float* out;
    int V, d, B;
} LMBJob;

static void lm_B_worker(void* ctx, long id, long lo, long hi) {
    (void)id;
    LMBJob* j = (LMBJob*)ctx;
    for (long i = lo; i < hi; i++) {
        const float* row = j->W + i * (size_t)j->d;
        for (int b = 0; b < j->B; b++) {
            const float* xb = j->x + b * (size_t)j->d;
            float acc = 0.0f;
#if defined(__AVX2__)
            __m256 vacc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= j->d; k += 8)
                vacc = _mm256_fmadd_ps(_mm256_loadu_ps(row + k),
                                       _mm256_loadu_ps(xb + k), vacc);
            __m128 sh = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
            sh = _mm_add_ps(sh, _mm_movehl_ps(sh, sh));
            sh = _mm_add_ss(sh, _mm_shuffle_ps(sh, sh, 1));
            acc = _mm_cvtss_f32(sh);
            for (; k < j->d; k++) acc += row[k] * xb[k];
#else
            for (int k = 0; k < j->d; k++) acc += row[k] * xb[k];
#endif
            j->out[b * (size_t)j->V + i] = acc;
        }
    }
}

static void lm_head_B(const float* W, const float* x, float* logits, int V, int d, int B) {
    LMBJob j = { W, x, logits, V, d, B };
    spin_pool_run(lm_B_worker, &j, 0, V);
}

/* argmax-only lm_head: out_tok[b] = argmax_v (W[v,d] @ x[b]) — the B*V
 * logits are never materialized (Python side skips the 4.8MB numpy alloc +
 * argmax per batch step — the serve decode overhead).
 *
 * Race-free parallel argmax: each pool participant (fixed id, includes main)
 * tracks the best (index, value) over ITS row range into per-worker locals;
 * after the pool barrier a single thread merges the workers' bests. The
 * shared out[] is NEVER written concurrently. */
typedef struct {
    const float* W; const float* x; int* out;
    int V, d, B;
    float* out_v;    /* [n_participants][B] best value per worker */
    int* out_i;      /* [n_participants][B] best index per worker */
} LMAJob;

#define LMA_MAX_W 17   /* MAX_POOL_WORKERS + main */

static void lm_A_worker(void* ctx, long id, long lo, long hi) {
    (void)id;
    LMAJob* j = (LMAJob*)ctx;
    float* ov = j->out_v + (size_t)id * j->B;
    int* oi = j->out_i + (size_t)id * j->B;
    for (int b = 0; b < j->B; b++) { ov[b] = -1e30f; oi[b] = 0; }
    for (long i = lo; i < hi; i++) {
        const float* row = j->W + i * (size_t)j->d;
        for (int b = 0; b < j->B; b++) {
            const float* xb = j->x + b * (size_t)j->d;
            float acc = 0.0f;
#if defined(__AVX2__)
            __m256 vacc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= j->d; k += 8)
                vacc = _mm256_fmadd_ps(_mm256_loadu_ps(row + k),
                                       _mm256_loadu_ps(xb + k), vacc);
            __m128 sh = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
            sh = _mm_add_ps(sh, _mm_movehl_ps(sh, sh));
            sh = _mm_add_ss(sh, _mm_shuffle_ps(sh, sh, 1));
            acc = _mm_cvtss_f32(sh);
            for (; k < j->d; k++) acc += row[k] * xb[k];
#else
            for (int k = 0; k < j->d; k++) acc += row[k] * xb[k];
#endif
            if (acc > ov[b]) { ov[b] = acc; oi[b] = (int)i; }
        }
    }
}

static void lm_head_A(const float* W, const float* x, int* out_tok, int V, int d, int B) {
    static float out_v[LMA_MAX_W][16];
    static int out_i[LMA_MAX_W][16];
    LMAJob j = { W, x, out_tok, V, d, B, &out_v[0][0], &out_i[0][0] };
    spin_pool_run(lm_A_worker, &j, 0, V);
    /* merge the participants' bests (single-threaded, deterministic) */
    const int np = (int)(g_pool.n_participants);
    for (int b = 0; b < B; b++) {
        int bi = 0; float bv = -1e30f;
        for (int w = 0; w < np; w++)
            if (out_v[w][b] > bv) { bv = out_v[w][b]; bi = out_i[w][b]; }
        out_tok[b] = bi;
    }
}

/* int8 lm_head (AVX-512 VNNI): logits[B*V] = dequant_i8(lm_head) @ x[B]^T
 * Quantizes each x[b] ONCE (per-tensor scale), then dots all V rows —
 * no per-row re-quantization. */
#if defined(__AVX512VNNI__)
typedef struct {
    const unsigned char* W; const float* Ws; const float* x; float* out;
    int V, d, B;
} LMI8Job;

static inline void quantize_x_i8(const float* x, signed char* xq, int d, float* sx_out) {
    float xmax = 1e-12f;
    for (int j = 0; j < d; j++) { float a = x[j] < 0 ? -x[j] : x[j]; if (a > xmax) xmax = a; }
    const float sx = xmax / 127.0f;
    *sx_out = sx;
    __m512 xs = _mm512_set1_ps(sx);
    int j = 0;
    for (; j + 16 <= d; j += 16) {
        __m512 xv = _mm512_loadu_ps(x + j);
        __m512 xr = _mm512_roundscale_ps(_mm512_div_ps(xv, xs), 0);
        __m512i xi = _mm512_cvtps_epi32(xr);
        _mm_storeu_si128((__m128i*)(xq + j), _mm512_cvtepi32_epi8(xi));
    }
    for (; j < d; j++) xq[j] = (signed char)(x[j] / sx);
}

static void lm_i8_worker(void* ctx, long id, long lo, long hi) {
    (void)id;
    LMI8Job* j = (LMI8Job*)ctx;
    const int d8 = (j->d + 63) / 64;
    for (int b = 0; b < j->B; b++) {
        /* quantize this sequence's x once */
        signed char xq[16384];
        float sx;
        quantize_x_i8(j->x + b * (size_t)j->d, xq, j->d, &sx);
        int32_t xsum = 0;
        for (int t = 0; t < j->d; t++) xsum += xq[t];
        const __m512i* a = (const __m512i*)xq;
        for (long i = lo; i < hi; i++) {
            const __m512i* w = (const __m512i*)(j->W + i * (size_t)j->d);
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < d8; k++) acc = _mm512_dpbusd_epi32(acc, w[k], a[k]);
            int32_t dot = _mm512_reduce_add_epi32(acc);
            dot -= 128 * xsum;
            j->out[b * (size_t)j->V + i] = (float)dot * sx * j->Ws[i];
        }
    }
}

static void lm_head_B_i8(const unsigned char* W, const float* Ws, const float* x,
                         float* logits, int V, int d, int B) {
    LMI8Job j = { W, Ws, x, logits, V, d, B };
    spin_pool_run(lm_i8_worker, &j, 0, V);
}
#endif

/* ---------------- batched attention: B seqs, shared W reads ------------- */
static void layer_attn_batch(const KernelCfg* c, const KernelW* W, KVCache* kv,
                             int l, const int* positions, const float* x,
                             int B, float* out, BatchWS* ws) {
    const int d = c->d, hd = c->hd, H = c->n_heads, KH = c->n_kv;
    ws_init(c, B);
    /* QKV projections (shared weights, B rows) */
#if defined(__AVX512VNNI__)
    if (W->use_i8) {
        matvec_i8_B(ws->q, W->wq_i8 + (size_t)l * (H * hd) * d,
                    W->wq_i8s + (size_t)l * (H * hd), x, H * hd, d, B);
        matvec_i8_B(ws->k, W->wk_i8 + (size_t)l * (KH * hd) * d,
                    W->wk_i8s + (size_t)l * (KH * hd), x, KH * hd, d, B);
        matvec_i8_B(ws->v, W->wv_i8 + (size_t)l * (KH * hd) * d,
                    W->wv_i8s + (size_t)l * (KH * hd), x, KH * hd, d, B);
    } else
#endif
    {
        matvec_f32_B(ws->q, W->wq + (size_t)l * (size_t)(H * hd) * d, x, H * hd, d, B);
        matvec_f32_B(ws->k, W->wk + (size_t)l * (size_t)(KH * hd) * d, x, KH * hd, d, B);
        matvec_f32_B(ws->v, W->wv + (size_t)l * (size_t)(KH * hd) * d, x, KH * hd, d, B);
    }
    /* QKV bias (qwen2 family) */
    if (c->bias_qkv) {
        const float* bq = W->bq + (size_t)l * (H * hd);
        const float* bk = W->bk + (size_t)l * (KH * hd);
        const float* bv = W->bv + (size_t)l * (KH * hd);
        for (int b = 0; b < B; b++) {
            float* qb = ws->q + (size_t)b * (H * hd);
            float* kb = ws->k + (size_t)b * (KH * hd);
            float* vb = ws->v + (size_t)b * (KH * hd);
            for (int i = 0; i < H * hd; i++) qb[i] += bq[i];
            for (int i = 0; i < KH * hd; i++) { kb[i] += bk[i]; vb[i] += bv[i]; }
        }
    }
    /* whole-vector QK-norm (olmoe) — before head view, weights [L, vec] */
    if (c->qk_norm_type == 2) {
        const unsigned short* wnq = W->qn + (size_t)l * (H * hd);
        const unsigned short* wnk = W->kn + (size_t)l * (KH * hd);
        for (int b = 0; b < B; b++) {
            float* qb = ws->q + (size_t)b * (H * hd);
            float* kb = ws->k + (size_t)b * (KH * hd);
            float ss = 0; for (int i = 0; i < H * hd; i++) ss += qb[i] * qb[i];
            float inv = 1.0f / sqrtf_fast(ss / (H * hd) + c->eps);
            for (int i = 0; i < H * hd; i++) qb[i] = qb[i] * inv * bf16_to_f32(wnq[i]);
            ss = 0; for (int i = 0; i < KH * hd; i++) ss += kb[i] * kb[i];
            inv = 1.0f / sqrtf_fast(ss / (KH * hd) + c->eps);
            for (int i = 0; i < KH * hd; i++) kb[i] = kb[i] * inv * bf16_to_f32(wnk[i]);
        }
    }
    for (int b = 0; b < B; b++) {
        float* qb = ws->q + (size_t)b * (H * hd);
        float* kb = ws->k + (size_t)b * (KH * hd);
        float* vb = ws->v + (size_t)b * (KH * hd);
        /* per-head QK-norm — only for per_head models (qwen3) */
        if (c->qk_norm_type == 1) {
        for (int h = 0; h < H; h++) {
            float* qh = qb + (size_t)h * hd;
            const unsigned short* wn = W->qn + (size_t)l * hd;
            float ss = 0; for (int i = 0; i < hd; i++) ss += qh[i] * qh[i];
            float inv = 1.0f / sqrtf_fast(ss / hd + c->eps);
            for (int i = 0; i < hd; i++) qh[i] = qh[i] * inv * bf16_to_f32(wn[i]);
        }
        for (int h = 0; h < KH; h++) {
            float* kh = kb + (size_t)h * hd;
            const unsigned short* wn = W->kn + (size_t)l * hd;
            float ss = 0; for (int i = 0; i < hd; i++) ss += kh[i] * kh[i];
            float inv = 1.0f / sqrtf_fast(ss / hd + c->eps);
            for (int i = 0; i < hd; i++) kh[i] = kh[i] * inv * bf16_to_f32(wn[i]);
        }
        }
        /* RoPE */
        const float* cos = W->cos + (size_t)positions[b] * hd;
        const float* sin = W->sin + (size_t)positions[b] * hd;
        for (int h = 0; h < H; h++) rope_apply(qb + (size_t)h * hd, cos, sin, hd);
        for (int h = 0; h < KH; h++) rope_apply(kb + (size_t)h * hd, cos, sin, hd);
        /* KV append */
        KVCache* kvb = &kv[b * (size_t)c->L + l];
        float* kc = kvb->k + (size_t)positions[b] * KH * hd;
        float* vc = kvb->v + (size_t)positions[b] * KH * hd;
        for (int i = 0; i < KH * hd; i++) { kc[i] = kb[i]; vc[i] = vb[i]; }
        /* SDPA */
        int T = positions[b] + 1;
        float* attb = ws->att + (size_t)b * (H * hd);
        static float scores[65536];   /* static: no stack pressure (T<=64k) */
        for (int h = 0; h < H; h++) {
            const float* qh = qb + (size_t)h * hd;
            int kh = h / (H / KH);
            const float* kk = kvb->k + (size_t)kh * hd;
            float smax = -1e30f;
            for (int t = 0; t < T; t++) {
                const float* kt = kk + (size_t)t * KH * hd;
                float s = 0; for (int i = 0; i < hd; i++) s += qh[i] * kt[i];
                scores[t] = s / sqrtf_fast((float)hd);
                if (scores[t] > smax) smax = scores[t];
            }
            float sum = 0;
            for (int t = 0; t < T; t++) { scores[t] = expf_fast(scores[t] - smax); sum += scores[t]; }
            float* oh = attb + (size_t)h * hd;
            for (int i = 0; i < hd; i++) oh[i] = 0;
            for (int t = 0; t < T; t++) {
                const float* vt = kvb->v + (size_t)t * KH * hd + (size_t)kh * hd;
                float p = scores[t] / sum;
                for (int i = 0; i < hd; i++) oh[i] += p * vt[i];
            }
        }
    }
    /* o_proj: [d, H*hd], B rows */
#if defined(__AVX512VNNI__)
    if (W->use_i8) {
        matvec_i8_B(out, W->wo_i8 + (size_t)l * d * (H * hd),
                    W->wo_i8s + (size_t)l * d, ws->att, d, H * hd, B);
    } else
#endif
        matvec_f32_B(out, W->wo + (size_t)l * (size_t)d * (H * hd), ws->att, d, H * hd, B);
}

/* ---------------- batched MoE FFN: union experts, shared weight reads ---- */
typedef struct {
    const KernelCfg* c; const KernelW* W;
    int l;               /* layer index (expert key = l*E + e) */
    const float* x;      /* [B, d] input */
    float* out;          /* [B, d] accumulator */
    int B;
    int U;               /* unique experts across batch */
    const int* sel;      /* [U] expert ids */
    const int* uidx;     /* [B*topk] slot ids (b*topk+i) */
    const float* uw;     /* [B*topk] routing weights */
    const int* uh;       /* [U+1] prefix offsets into uidx/uw */
    BatchWS* ws;
} FFNJob;

static void ffn_expert_worker(void* ctx, long id, long lo, long hi) {
    (void)id;
    FFNJob* j = (FFNJob*)ctx;
    const KernelCfg* c = j->c;
    const KernelW* W = j->W;
    const int d = c->d, m = c->intermediate, topk = c->topk, E = c->E;
    const int nb = d >> 1, gbytes = 32, ng = d / 64;
    const int nb_d = m >> 1, ng_d = m / 64;
    BatchWS* ws = j->ws;
    const size_t base = (size_t)j->l * E;
    for (long ue = lo; ue < hi; ue++) {
        const int e = j->sel[ue];
        const int s0 = j->uh[ue], s1 = j->uh[ue + 1];
        if (s1 <= s0) continue;
        const unsigned char* pkg = W->expert_pk[0][base + e];
        const unsigned short* scg = W->expert_sc[0][base + e];
        const unsigned char* pku = W->expert_pk[1][base + e];
        const unsigned short* scu = W->expert_sc[1][base + e];
        const unsigned char* pkd = W->expert_pk[2][base + e];
        const unsigned short* scd = W->expert_sc[2][base + e];
        float gacc[64], uacc[64], dacc[64];  /* B <= 64 */
        /* per-expert act: [B][m], indexed by union position ue (disjoint
         * worker ranges -> no two participants touch the same region) */
        float* acte = ws->act + ((size_t)ue * j->B) * m;
        float* ye = ws->y + ((size_t)ue * j->B) * d;
        if (c->expert_bf16) {
            /* bf16 experts (exact tier): rows are bf16 uint16, no scales.
             * expert_pk[p][base+e] points at the [rows][cols] bf16 matrix.
             * The gate/up/down dots are exact bf16 matvecs — no quantization
             * error, so long generations stay token-identical to HF. */
            const unsigned short* bg = (const unsigned short*)pkg;
            const unsigned short* bu = (const unsigned short*)pku;
            const unsigned short* bd = (const unsigned short*)pkd;
            for (int i = 0; i < m; i++) {
                bf16_row_dot_B(bg + (size_t)i * d, j->x, j->B, gacc, d, d);
                bf16_row_dot_B(bu + (size_t)i * d, j->x, j->B, uacc, d, d);
                for (int s = s0; s < s1; s++) {
                    const int b = j->uidx[s] / topk;
                    const float g = gacc[b], u = uacc[b];
                    acte[b * m + i] = (g / (1.0f + expf_fast(-g))) * u;
                }
            }
            for (int i = 0; i < d; i++) {
                bf16_row_dot_B(bd + (size_t)i * m, acte, j->B, dacc, m, m);
                for (int s = s0; s < s1; s++) {
                    const int b = j->uidx[s] / topk;
                    ye[b * d + i] += j->uw[s] * dacc[b];
                }
            }
            continue;
        }
        if (c->expert_i8) {
            /* int8 experts (Q8_0: per-row fp32 scale, +128 bias): quantize
             * the activation ONCE per sequence, then vpmaddubsw per row.
             * scg/scu/scd point at fp32 per-row scales (not fp16 groups). */
            float sx[64], asx[64];
            int32_t xsum[64], asum[64];
            /* i8 store scales are fp32 PER ROW — cast once (the pointer is
             * unsigned short*, so `scg + i` would stride 2 bytes and read
             * garbage/NaN scales — must index the fp32 view) */
            const float* scgf = (const float*)scg;
            const float* scuf = (const float*)scu;
            const float* scdf = (const float*)scd;
            /* Per-participant aq slice: ws->aq is SHARED, and each union
             * expert quantizes its OWN acte into it before the down dot. With
             * the pool (B>1) multiple workers write ws->aq concurrently and
             * overwrite each other's rows -> garbage down outputs (i8-only;
             * the q4 path reads acte directly and is immune). Slice by worker
             * id so each participant's quantize/read stays private. */
            signed char* aqs = ws->aq + (size_t)id * (size_t)BMAX * (d > m ? d : m);
            for (int b = 0; b < j->B; b++) {
                const float* xb = j->x + (size_t)b * d;
                float xmax = 1e-12f;
                for (int k = 0; k < d; k++) { float a = xb[k] < 0 ? -xb[k] : xb[k]; if (a > xmax) xmax = a; }
                sx[b] = xmax / 127.0f;
                signed char* xqb = ws->xq + (size_t)b * d;
                /* round-to-nearest (C cast truncates toward 0 — a 0.5-1.0
                 * bias per element that compounds into garbage; the Q8_0
                 * attention path rounds too) */
                for (int k = 0; k < d; k++)
                    xqb[k] = (signed char)(xb[k] / sx[b] + (xb[k] >= 0 ? 0.5f : -0.5f));
                xsum[b] = 0; for (int k = 0; k < d; k++) xsum[b] += xqb[k];
            }
            for (int i = 0; i < m; i++) {
                q8_row_dot_B(pkg + (size_t)i * d, scgf[i],
                             ws->xq, sx, xsum, j->B, gacc, d, d);
                q8_row_dot_B(pku + (size_t)i * d, scuf[i],
                             ws->xq, sx, xsum, j->B, uacc, d, d);
                for (int s = s0; s < s1; s++) {
                    const int b = j->uidx[s] / topk;
                    const float g = gacc[b], u = uacc[b];
                    acte[b * m + i] = (g / (1.0f + expf_fast(-g))) * u;
                }
            }
            for (int b = 0; b < j->B; b++) {
                const float* ab = acte + (size_t)b * m;
                float amax = 1e-12f;
                for (int k = 0; k < m; k++) { float a = ab[k] < 0 ? -ab[k] : ab[k]; if (a > amax) amax = a; }
                asx[b] = amax / 127.0f;
                signed char* aqb = aqs + (size_t)b * m;
                for (int k = 0; k < m; k++)
                    aqb[k] = (signed char)(ab[k] / asx[b] + (ab[k] >= 0 ? 0.5f : -0.5f));
                asum[b] = 0; for (int k = 0; k < m; k++) asum[b] += aqb[k];
            }
            for (int i = 0; i < d; i++) {
                q8_row_dot_B(pkd + (size_t)i * m, scdf[i],
                             aqs, asx, asum, j->B, dacc, m, m);
                for (int s = s0; s < s1; s++) {
                    const int b = j->uidx[s] / topk;
                    ye[b * d + i] += j->uw[s] * dacc[b];
                }
            }
            continue;
        }
        for (int i = 0; i < m; i++) {
            q4_row_dot_B(pkg + (size_t)i * nb, scg + (size_t)i * ng, j->x, j->B, gacc, nb, gbytes, ng, d);
            q4_row_dot_B(pku + (size_t)i * nb, scu + (size_t)i * ng, j->x, j->B, uacc, nb, gbytes, ng, d);
            for (int s = s0; s < s1; s++) {
                const int b = j->uidx[s] / topk;
                const float g = gacc[b], u = uacc[b];
                acte[b * m + i] = (g / (1.0f + expf_fast(-g))) * u;
            }
        }
        /* down: y_e[b][i] += w * (act[b] @ down row^T) — per-expert partial,
         * combined into out AFTER the pool barrier (no cross-thread RMW) */
        for (int i = 0; i < d; i++) {
            q4_row_dot_B(pkd + (size_t)i * nb_d, scd + (size_t)i * ng_d,
                         acte, j->B, dacc, nb_d, gbytes, ng_d, m);
            for (int s = s0; s < s1; s++) {
                const int b = j->uidx[s] / topk;
                ye[b * d + i] += j->uw[s] * dacc[b];
            }
        }
    }
}

static void layer_ffn_batch(const KernelCfg* c, const KernelW* W, int l,
                            const float* x, int B, float* out) {
    const int d = c->d, E = c->E, topk = c->topk, m = c->intermediate;
    ws_init(c, B);
    BatchWS* ws = &g_ws;
    /* dense FFN (E==0): silu(x@w1.T) * (x@w2.T) @ w3.T — batched, shared weights */
    if (E == 0) {
        const unsigned short* w1 = W->w1 + (size_t)l * m * d;
        const unsigned short* w2 = W->w2 + (size_t)l * m * d;
        const unsigned short* w3 = W->w3 + (size_t)l * d * m;
        for (int i = 0; i < m; i++) {
            float gb[64], ub[64];
            for (int b = 0; b < B; b++) {
                const float* xb = x + (size_t)b * d;
                float ga = 0, ua = 0;
                for (int j = 0; j < d; j++) {
                    ga += bf16_to_f32(w1[(size_t)i * d + j]) * xb[j];
                    ua += bf16_to_f32(w2[(size_t)i * d + j]) * xb[j];
                }
                gb[b] = ga; ub[b] = ua;
            }
            for (int b = 0; b < B; b++) {
                float act = (gb[b] / (1.0f + expf_fast(-gb[b]))) * ub[b];
                ws->act[(size_t)b * m + i] = act;
            }
        }
        for (int i = 0; i < d; i++) {
            for (int b = 0; b < B; b++) {
                const float* actb = ws->act + (size_t)b * m;
                float acc = 0;
                for (int j = 0; j < m; j++) acc += bf16_to_f32(w3[(size_t)i * m + j]) * actb[j];
                out[(size_t)b * d + i] = acc;
            }
        }
        return;
    }
    /* routing: scores[B*E] = gate_w[E,d] @ x[B,d]^T (shared weight read) */
    matvec_f32_B(ws->scores, W->gate_w + (size_t)l * E * d, x, E, d, B);
    /* per-seq softmax + topk + renormalize */
    for (int b = 0; b < B; b++) {
        const float* sc = ws->scores + (size_t)b * E;
        float smax = -1e30f;
        for (int e = 0; e < E; e++) if (sc[e] > smax) smax = sc[e];
        float sum = 0;
        for (int e = 0; e < E; e++) { ws->scores[(size_t)b * E + e] = expf_fast(sc[e] - smax); sum += ws->scores[(size_t)b * E + e]; }
        for (int e = 0; e < E; e++) ws->scores[(size_t)b * E + e] /= sum;
        for (int i = 0; i < topk; i++) {
            int bi = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int used = 0;
                for (int j = 0; j < i; j++) if (ws->top[(size_t)b * topk + j] == e) { used = 1; break; }
                if (!used && ws->scores[(size_t)b * E + e] > bv) { bv = ws->scores[(size_t)b * E + e]; bi = e; }
            }
            ws->top[(size_t)b * topk + i] = bi; ws->tw[(size_t)b * topk + i] = bv;
        }
        /* renormalize only when the arch does (qwen3_moe); olmoe/mixtral keep
         * the raw softmax weights — HF gates on norm_topk_prob */
        if (c->norm_topk_prob) {
            float ssum = 0; for (int i = 0; i < topk; i++) ssum += ws->tw[(size_t)b * topk + i];
            for (int i = 0; i < topk; i++) ws->tw[(size_t)b * topk + i] /= ssum;
        }
    }
    /* union of experts (max B*topk, capped at 128 = E) */
    int U = 0;
    for (int b = 0; b < B; b++)
        for (int i = 0; i < topk; i++) {
            const int e = ws->top[(size_t)b * topk + i];
            int found = 0;
            for (int u = 0; u < U; u++) if (ws->sel[u] == e) { found = 1; break; }
            if (!found && U < E) ws->sel[U++] = e;
        }
    /* count slots per union expert, build prefix offsets + ordered user map */
    int cnt[512] = {0};
    for (int b = 0; b < B; b++)
        for (int i = 0; i < topk; i++) {
            const int e = ws->top[(size_t)b * topk + i];
            int u = 0;
            for (; u < U; u++) if (ws->sel[u] == e) break;
            cnt[u]++;
        }
    int acc = 0;
    for (int u = 0; u < U; u++) { ws->uh[u] = acc; acc += cnt[u]; }
    ws->uh[U] = acc;
    {
        int pos[512];
        for (int u = 0; u < U; u++) pos[u] = ws->uh[u];
        int uid_tmp[512]; float uw_tmp[512];
        for (int b = 0; b < B; b++)
            for (int i = 0; i < topk; i++) {
                const int e = ws->top[(size_t)b * topk + i];
                int u = 0;
                for (; u < U; u++) if (ws->sel[u] == e) break;
                const int slot = b * topk + i;
                uid_tmp[pos[u]] = slot;
                uw_tmp[pos[u]] = ws->tw[(size_t)b * topk + i];
                pos[u]++;
            }
        for (int s = 0; s < B * topk; s++) { ws->uidx[s] = uid_tmp[s]; ws->uw[s] = uw_tmp[s]; }
    }
    /* zero accumulators, run experts on the pool (each expert's weight rows
     * are read once across all of its users — the batch amortization).
     * Each participant writes per-expert partials into ws->y (disjoint ue
     * ranges); the combine below is single-threaded after the pool barrier. */
    for (int b = 0; b < B; b++)
        for (int i = 0; i < d; i++) out[b * (size_t)d + i] = 0.0f;
    for (int ue = 0; ue < U; ue++)
        for (int s = 0; s < B * d; s++) ws->y[(size_t)ue * (B * d) + s] = 0.0f;
    FFNJob job = { c, W, l, x, out, B, U, ws->sel, ws->uidx, ws->uw, ws->uh, ws };
    /* DEBUG: inline (no pool) for B=1 */
    if (B == 1) {
        ffn_expert_worker(&job, 0, 0, U);
    } else {
        spin_pool_run(ffn_expert_worker, &job, 0, U);
    }
    /* combine per-expert partials (single-threaded, deterministic) */
    for (int ue = 0; ue < U; ue++) {
        const float* ye = ws->y + ((size_t)ue * B) * d;
        for (int b = 0; b < B; b++)
            for (int i = 0; i < d; i++)
                out[b * (size_t)d + i] += ye[b * d + i];
    }
}

static void layer_ffn(const KernelCfg* c, const KernelW* W, int l,
                      const float* x, float* out) {
    layer_ffn_batch(c, W, l, x, 1, out);
}

/* ---------------- main: batch decode B tokens ---------------- */
/* One shared weight pass: per layer, batched attention + union-expert FFN;
 * per-seq KV positions; pooled lm_head at the end. */
void decode_layers_batch(const KernelCfg* c, const KernelW* W, KVCache* kv,
                         const int* positions, const float* xin, int B,
                         float* logits) {
    const int d = c->d, L = c->L;
    if (B > BMAX) B = BMAX;  /* workspace is sized for BMAX */
    BatchWS* ws = &g_ws;
    ws_init(c, B);
    for (int b = 0; b < B; b++)
        for (int i = 0; i < d; i++) ws->h[(size_t)b * d + i] = xin[(size_t)b * d + i];
    for (int l = 0; l < L && (g_max_layers < 0 || l < g_max_layers); l++) {
        for (int b = 0; b < B; b++)
            rms_norm(ws->tmp + (size_t)b * d, ws->h + (size_t)b * d, W->norm1 + (size_t)l * d, d, c->eps);
        layer_attn_batch(c, W, kv, l, positions, ws->tmp, B, ws->h2, ws);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < d; i++) ws->h[(size_t)b * d + i] += ws->h2[(size_t)b * d + i];
        for (int b = 0; b < B; b++)
            rms_norm(ws->tmp + (size_t)b * d, ws->h + (size_t)b * d, W->norm2 + (size_t)l * d, d, c->eps);
        layer_ffn_batch(c, W, l, ws->tmp, B, ws->h2);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < d; i++) ws->h[(size_t)b * d + i] += ws->h2[(size_t)b * d + i];
        if (l == L - 1) {
            float s = 0;
            for (int i = 0; i < d; i++) s += ws->h[i] * ws->h[i];
            g_h_norm = s;   /* capture pre-final-norm h^2 at the last layer */
        }
    }
    for (int b = 0; b < B; b++)
        rms_norm(ws->tmp + (size_t)b * d, ws->h + (size_t)b * d, W->final_norm, d, c->eps);
    /* lm_head is the most expensive single read (V x d); prefill passes
     * logits=NULL for all but the last token to skip it there */
    if (logits) {
#if defined(__AVX512VNNI__)
        if (W->use_i8 && W->lm_i8) {
            lm_head_B_i8(W->lm_i8, W->lm_i8s, ws->tmp, logits, c->V, d, B);
        } else
#endif
            lm_head_B(W->lm_head, ws->tmp, logits, c->V, d, B);
    }
}

/* decode_layers_batch, argmax-only: out_tok[b] = the argmax token per seq.
 * No B*V logits buffer — the hot serve decode path (one argmax per step). */
void decode_layers_batch_argmax(const KernelCfg* c, const KernelW* W, KVCache* kv,
                                const int* positions, const float* xin, int B,
                                int* out_tok) {
    const int d = c->d, L = c->L;
    if (B > BMAX) B = BMAX;
    BatchWS* ws = &g_ws;
    ws_init(c, B);
    for (int b = 0; b < B; b++)
        for (int i = 0; i < d; i++) ws->h[(size_t)b * d + i] = xin[(size_t)b * d + i];
    for (int l = 0; l < L && (g_max_layers < 0 || l < g_max_layers); l++) {
        for (int b = 0; b < B; b++)
            rms_norm(ws->tmp + (size_t)b * d, ws->h + (size_t)b * d, W->norm1 + (size_t)l * d, d, c->eps);
        layer_attn_batch(c, W, kv, l, positions, ws->tmp, B, ws->h2, ws);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < d; i++) ws->h[(size_t)b * d + i] += ws->h2[(size_t)b * d + i];
        for (int b = 0; b < B; b++)
            rms_norm(ws->tmp + (size_t)b * d, ws->h + (size_t)b * d, W->norm2 + (size_t)l * d, d, c->eps);
        layer_ffn_batch(c, W, l, ws->tmp, B, ws->h2);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < d; i++) ws->h[(size_t)b * d + i] += ws->h2[(size_t)b * d + i];
    }
    for (int b = 0; b < B; b++)
        rms_norm(ws->tmp + (size_t)b * d, ws->h + (size_t)b * d, W->final_norm, d, c->eps);
    /* argmax decode uses the fp32 lm_head (exact; the int8 lm_head argmax
     * needs a per-row-scale compare — fp32 is simple and correct) */
    lm_head_A(W->lm_head, ws->tmp, out_tok, c->V, d, B);
}

/* ---------------- main: decode one token ---------------- */
void decode_layers(const KernelCfg* c, const KernelW* W, KVCache* kv,
                   int pos, const float* xin, float* logits) {
    /* single decode = decode_layers_batch with B=1 (the verified path) */
    int pos_arr[1] = {pos};
    decode_layers_batch(c, W, kv, pos_arr, xin, 1, logits);
}

/* Embedding lookup: out[d] = embed[token] (embed is bf16 uint16) */
void embed_lookup(const KernelW* W, const KernelCfg* c, int token, float* out) {
    const unsigned short* row = W->embed + (size_t)token * c->d;
    for (int i = 0; i < c->d; i++) out[i] = bf16_to_f32(row[i]);
}

/* Prefill: run a full prompt [tokens, T] through all layers, storing KV at
 * positions 0..T-1. The lm_head is applied ONLY at the last token (the
 * intermediate tokens don't need logits — saves T-1 lm_head reads, the
 * 1.2GB-per-token cost). Returns the last-token logits in `logits`. */
void prefill_layers(const KernelCfg* c, const KernelW* W, KVCache* kv,
                    const int* tokens, int T, float* logits) {
    static float xin[65536];   /* static: avoid stack pressure in the loop */
    for (int t = 0; t < T; t++) {
        int pos_arr[1] = {t};
        embed_lookup(W, c, tokens[t], xin);
        decode_layers_batch(c, W, kv, pos_arr, xin, 1,
                            (t == T - 1) ? logits : NULL);
    }
}

void set_max_layers(int n) { g_max_layers = n; }

/* debug: capture ws->act[0..n) after running the dense FFN first loop */
void dense_act_test(const KernelCfg* c, const KernelW* W, int l, const float* x, float* out, int n) {
    const int d = c->d, m = c->intermediate;
    ws_init(c, 1);
    BatchWS* ws = &g_ws;
    const unsigned short* w1 = W->w1 + (size_t)l * m * d;
    const unsigned short* w2 = W->w2 + (size_t)l * m * d;
    for (int i = 0; i < m; i++) {
        float gb = 0, ub = 0;
        for (int j = 0; j < d; j++) {
            gb += bf16_to_f32(w1[(size_t)i * d + j]) * x[j];
            ub += bf16_to_f32(w2[(size_t)i * d + j]) * x[j];
        }
        ws->act[i] = (gb / (1.0f + expf_fast(-gb))) * ub;
    }
    for (int i = 0; i < n && i < m; i++) out[i] = ws->act[i];
}

/* debug: dense down row-0 dot for (layer, d_idx) */
void dense_down_test(const KernelW* W, const KernelCfg* c, int l, int di,
                     const float* x, float* out) {
    const unsigned short* w3 = W->w3 + (size_t)l * c->d * c->intermediate;
    float acc = 0;
    for (int j = 0; j < c->intermediate; j++) acc += bf16_to_f32(w3[(size_t)di * c->intermediate + j]) * x[j];
    out[0] = acc;
}

/* debug: dense gate row-0 dot for (layer, m_idx) */
void dense_gate_test(const KernelW* W, const KernelCfg* c, int l, int mi,
                     const float* x, float* out) {
    const unsigned short* w1 = W->w1 + (size_t)l * c->intermediate * c->d;
    float acc = 0;
    for (int j = 0; j < c->d; j++) acc += bf16_to_f32(w1[(size_t)mi * c->d + j]) * x[j];
    out[0] = acc;
}

/* debug: attention output at layer l (after layers 0..l-1 fully) */
void debug_attn_n(const KernelCfg* c, const KernelW* W, KVCache* kv,
                  const float* xin, int l, float* out) {
    const int d = c->d;
    ws_init(c, 1);
    BatchWS* ws = &g_ws;
    int pos_arr[1] = {0};
    for (int i = 0; i < d; i++) ws->h[i] = xin[i];
    for (int ll = 0; ll < l; ll++) {
        rms_norm(ws->tmp, ws->h, W->norm1 + (size_t)ll * d, d, c->eps);
        layer_attn_batch(c, W, kv, ll, pos_arr, ws->tmp, 1, ws->h2, ws);
        for (int i = 0; i < d; i++) ws->h[i] += ws->h2[i];
        rms_norm(ws->tmp, ws->h, W->norm2 + (size_t)ll * d, d, c->eps);
        layer_ffn_batch(c, W, ll, ws->tmp, 1, ws->h2);
        for (int i = 0; i < d; i++) ws->h[i] += ws->h2[i];
    }
    rms_norm(ws->tmp, ws->h, W->norm1 + (size_t)l * d, d, c->eps);
    layer_attn_batch(c, W, kv, l, pos_arr, ws->tmp, 1, ws->h2, ws);
    for (int i = 0; i < d; i++) out[i] = ws->h2[i];
}

/* debug: return the pre-final-norm hidden after n_layers (batch B=1) */
void debug_hidden_n(const KernelCfg* c, const KernelW* W, KVCache* kv,
                    const float* xin, int n, float* out) {
    const int d = c->d;
    ws_init(c, 1);
    BatchWS* ws = &g_ws;
    int pos_arr[1] = {0};
    for (int i = 0; i < d; i++) ws->h[i] = xin[i];
    for (int ll = 0; ll < n; ll++) {
        rms_norm(ws->tmp, ws->h, W->norm1 + (size_t)ll * d, d, c->eps);
        layer_attn_batch(c, W, kv, ll, pos_arr, ws->tmp, 1, ws->h2, ws);
        for (int i = 0; i < d; i++) ws->h[i] += ws->h2[i];
        rms_norm(ws->tmp, ws->h, W->norm2 + (size_t)ll * d, d, c->eps);
        layer_ffn_batch(c, W, ll, ws->tmp, 1, ws->h2);
        for (int i = 0; i < d; i++) ws->h[i] += ws->h2[i];
        if (ll == n - 1) {
            float s = 0;
            for (int i = 0; i < d; i++) s += ws->tmp[i] * ws->tmp[i];
            g_ffn_in_norm = s;
        }
    }
    for (int i = 0; i < d; i++) out[i] = ws->h[i];
}

/* debug: run layers 0..l-1, then layer l attention, then layer l FFN.
 * Returns the FFN output (pre-residual) in `out`. */
void debug_ffn_out(const KernelCfg* c, const KernelW* W, KVCache* kv,
                   const float* xin, int l, float* out) {
    const int d = c->d;
    ws_init(c, 1);
    BatchWS* ws = &g_ws;
    int pos_arr[1] = {0};
    for (int i = 0; i < d; i++) ws->h[i] = xin[i];
    for (int ll = 0; ll < l; ll++) {
        rms_norm(ws->tmp, ws->h, W->norm1 + (size_t)ll * d, d, c->eps);
        layer_attn_batch(c, W, kv, ll, pos_arr, ws->tmp, 1, ws->h2, ws);
        for (int i = 0; i < d; i++) ws->h[i] += ws->h2[i];
        rms_norm(ws->tmp, ws->h, W->norm2 + (size_t)ll * d, d, c->eps);
        layer_ffn_batch(c, W, ll, ws->tmp, 1, ws->h2);
        for (int i = 0; i < d; i++) ws->h[i] += ws->h2[i];
    }
    /* layer l: attention then FFN (matching decode_layers_batch) */
    rms_norm(ws->tmp, ws->h, W->norm1 + (size_t)l * d, d, c->eps);
    layer_attn_batch(c, W, kv, l, pos_arr, ws->tmp, 1, ws->h2, ws);
    for (int i = 0; i < d; i++) ws->h[i] += ws->h2[i];
    rms_norm(ws->tmp, ws->h, W->norm2 + (size_t)l * d, d, c->eps);
    layer_ffn_batch(c, W, l, ws->tmp, 1, ws->h2);
    for (int i = 0; i < d; i++) out[i] = ws->h2[i];
}

/* debug: down row-0 dot for (layer, expert) — dims from the runtime cfg */
void q4_dot_test(const KernelW* W, int l, int e, const float* x, float* out) {
    const size_t base = (size_t)l * (size_t)g_E + e;
    const unsigned char* pkd = W->expert_pk[2][base];
    const unsigned short* scd = W->expert_sc[2][base];
    float acc[16];
    q4_row_dot_B(pkd, scd, x, 1, acc, g_m >> 1, 32, g_m / 64, g_m);
    out[0] = acc[0];
}

/* debug: run `layers` layers (batch B=1) and write the pre-lm_head hidden */
void debug_hidden_after(const KernelCfg* c, const KernelW* W, KVCache* kv,
                        const float* xin, int layers, float* out) {
    int pos_arr[1] = {0};
    float lg[4096];   /* dummy logits */
    int saved = g_max_layers;
    g_max_layers = layers;
    decode_layers_batch(c, W, kv, pos_arr, xin, 1, lg);
    /* capture hidden: re-run and grab pre-lm_head via the workspace */
    BatchWS* ws = &g_ws;
    const int d = c->d;
    /* decode_layers_batch ends with final_norm in ws->tmp */
    for (int i = 0; i < d; i++) out[i] = ws->tmp[i];
    g_max_layers = saved;
}

/* debug: decode at `pos` (KV must already hold positions < pos from prefill/
 * replay) and capture per-layer intermediates for a component-level bisect:
 * for each layer l, a row of 7 * rowlen floats (rowlen = max(d, E)):
 *   [0] attn_in   (post-norm1, the attention input)
 *   [1] attn_out  (post-o_proj, pre-residual)
 *   [2] h_after_attn
 *   [3] ffn_in    (post-norm2)
 *   [4] router_logits [E] (pre-softmax; dense layers leave 0)
 *   [5] ffn_out   (pre-residual)
 *   [6] h_after_ffn
 * KV at `pos` is recomputed idempotently (same input as the replay wrote).
 */
void debug_decode_layers(const KernelCfg* c, const KernelW* W, KVCache* kv,
                         int pos, const float* xin, float* out) {
    const int d = c->d, L = c->L, E = c->E;
    const int rowlen = (E > d ? E : d);
    ws_init(c, 1);
    BatchWS* ws = &g_ws;
    int pos_arr[1] = {pos};
    for (int i = 0; i < d; i++) ws->h[i] = xin[i];
    for (int l = 0; l < L; l++) {
        float* row = out + (size_t)l * 7 * rowlen;
        rms_norm(ws->tmp, ws->h, W->norm1 + (size_t)l * d, d, c->eps);
        memcpy(row + 0 * rowlen, ws->tmp, (size_t)d * sizeof(float));
        layer_attn_batch(c, W, kv, l, pos_arr, ws->tmp, 1, ws->h2, ws);
        memcpy(row + 1 * rowlen, ws->h2, (size_t)d * sizeof(float));
        for (int i = 0; i < d; i++) { ws->h[i] += ws->h2[i]; }
        memcpy(row + 2 * rowlen, ws->h, (size_t)d * sizeof(float));
        rms_norm(ws->tmp, ws->h, W->norm2 + (size_t)l * d, d, c->eps);
        memcpy(row + 3 * rowlen, ws->tmp, (size_t)d * sizeof(float));
        if (E > 0) {
            matvec_f32(row + 4 * rowlen, W->gate_w + (size_t)l * E * d, ws->tmp, E, d);
        }
        layer_ffn_batch(c, W, l, ws->tmp, 1, ws->h2);
        memcpy(row + 5 * rowlen, ws->h2, (size_t)d * sizeof(float));
        for (int i = 0; i < d; i++) { ws->h[i] += ws->h2[i]; }
        memcpy(row + 6 * rowlen, ws->h, (size_t)d * sizeof(float));
    }
}

/* debug: return the workspace base pointer (0 if malloc failed) */
unsigned long long ws_base_debug(const KernelCfg* c) {
    if (!ws_init(c, 1)) return 0;
    return (unsigned long long)(size_t)g_ws.h;
}

/* ---------------- threaded lm_head (win32, no CRT) ---------------- */
typedef struct {
    const float* W; const float* x; float* out; int m, d, start, end;
} LMJob;
static unsigned long __stdcall lm_worker(void* p) {
    LMJob* j = (LMJob*)p;
    for (int i = j->start; i < j->end; i++) {
        const float* row = j->W + (size_t)i * j->d;
        float acc = 0.0f;
#if defined(__AVX2__)
        __m256 vacc = _mm256_setzero_ps();
        int k = 0;
        for (; k + 8 <= j->d; k += 8)
            vacc = _mm256_fmadd_ps(_mm256_loadu_ps(row + k), _mm256_loadu_ps(j->x + k), vacc);
        __m128 sh = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
        sh = _mm_add_ps(sh, _mm_movehl_ps(sh, sh));
        sh = _mm_add_ss(sh, _mm_shuffle_ps(sh, sh, 1));
        acc = _mm_cvtss_f32(sh);
        for (; k < j->d; k++) acc += row[k] * j->x[k];
#else
        for (int k = 0; k < j->d; k++) acc += row[k] * j->x[k];
#endif
        j->out[i] = acc;
    }
    return 0;
}
/* ---------------- threaded matvec (win32) ---------------- */
static void matvec_f32_par(float* out, const float* W, const float* x,
                           int m, int d, int nthreads) {
    if (nthreads <= 1 || m < 256) { matvec_f32(out, W, x, m, d); return; }
    if (nthreads > 16) nthreads = 16;
    static LMJob jobs[16]; static void* handles[16];
    for (int t = 0; t < nthreads; t++) {
        jobs[t].W = W; jobs[t].x = x; jobs[t].out = out;
        jobs[t].m = m; jobs[t].d = d;
        jobs[t].start = (size_t)t * m / nthreads;
        jobs[t].end = (size_t)(t + 1) * m / nthreads;
    }
    void* mod = GetModuleHandleA("kernel32.dll");
    if (!mod) { matvec_f32(out, W, x, m, d); return; }
    for (int t = 0; t < nthreads; t++) handles[t] = CreateThread(0, 0, lm_worker, &jobs[t], 0, 0);
    WaitForMultipleObjects(nthreads, handles, 1, 0xFFFFFFFFu);
}

static int lm_nthreads = 8;
void set_lm_threads(int n) { lm_nthreads = n; }
void lm_head_parallel(const float* W, const float* x, float* out, int m, int d) {
    int nt = lm_nthreads; if (nt > m) nt = m;
    LMJob jobs[32]; void* handles[32];
    for (int t = 0; t < nt; t++) {
        jobs[t].W = W; jobs[t].x = x; jobs[t].out = out;
        jobs[t].m = m; jobs[t].d = d;
        jobs[t].start = (size_t)t * m / nt;
        jobs[t].end = (size_t)(t + 1) * m / nt;
    }
    void* mod = GetModuleHandleA("kernel32.dll");
    if (!mod) { lm_worker(&jobs[0]); return; }
    for (int t = 0; t < nt; t++) {
        handles[t] = CreateThread(0, 0, lm_worker, &jobs[t], 0, 0);
    }
    WaitForMultipleObjects(nt, handles, 1, 0xFFFFFFFFu);
}

/* debug: run the L-max layers like decode_layers/decode_layers_batch but
 * return the FINAL NORM'D hidden (pre-lm_head) instead of logits. */
void debug_final_hidden(const KernelCfg* c, const KernelW* W, KVCache* kv,
                        int pos, const float* xin, float* h_out, int use_batch) {
    const int d = c->d, L = c->L;
    BatchWS* ws = &g_ws;
    if (use_batch) {
        int positions[1] = {pos};
        for (int i = 0; i < d; i++) ws->h[i] = xin[i];
        for (int l = 0; l < L && (g_max_layers < 0 || l < g_max_layers); l++) {
            rms_norm(ws->tmp, ws->h, W->norm1 + (size_t)l * d, d, c->eps);
            layer_attn_batch(c, W, kv, l, positions, ws->tmp, 1, ws->h2, ws);
            for (int i = 0; i < d; i++) ws->h[i] += ws->h2[i];
            rms_norm(ws->tmp, ws->h, W->norm2 + (size_t)l * d, d, c->eps);
            layer_ffn_batch(c, W, l, ws->tmp, 1, ws->h2);
            for (int i = 0; i < d; i++) ws->h[i] += ws->h2[i];
        }
        rms_norm(ws->tmp, ws->h, W->final_norm, d, c->eps);
        for (int i = 0; i < d; i++) h_out[i] = ws->tmp[i];
    } else {
        float h[2048], h2[2048], tmp[2048];
        for (int i = 0; i < d; i++) h[i] = xin[i];
        for (int l = 0; l < L && (g_max_layers < 0 || l < g_max_layers); l++) {
            rms_norm(tmp, h, W->norm1 + (size_t)l * d, d, c->eps);
            layer_attn(c, W, kv, l, pos, tmp, h2);
            for (int i = 0; i < d; i++) h[i] += h2[i];
            rms_norm(tmp, h, W->norm2 + (size_t)l * d, d, c->eps);
            layer_ffn(c, W, l, tmp, h2);
            for (int i = 0; i < d; i++) h[i] += h2[i];
        }
        rms_norm(tmp, h, W->final_norm, d, c->eps);
        for (int i = 0; i < d; i++) h_out[i] = tmp[i];
    }
}
/* debug: run layer 0 FFN on xin (post-norm), write ffn_out[d] */
void debug_ffn0(const KernelCfg* c, const KernelW* W, const float* xin, float* ffn_out) {
    float tmp[2048];
    rms_norm(tmp, xin, W->norm2, c->d, c->eps);
    layer_ffn(c, W, 0, tmp, ffn_out);
}
/* debug: batch attention B=1 (pos 0) on xin, write attn_out[d] */
void debug_attn0_batch(const KernelCfg* c, const KernelW* W, KVCache* kv,
                       const float* xin, float* attn_out) {
    int pos[1] = {0};
    float tmp[2048];
    rms_norm(tmp, xin, W->norm1, c->d, c->eps);
    layer_attn_batch(c, W, kv, 0, pos, tmp, 1, attn_out, &g_ws);
}

int _DllMainCRTStartup(void* a, unsigned long b, void* c2) { (void)a;(void)b;(void)c2; return 1; }

/* malloc/free via runtime-resolved kernel32 (no CRT, no static import).
 * GetProcAddress is already used for WaitOnAddress — same pattern. */
typedef void* (__stdcall *GetProcessHeapFn)(void);
typedef void* (__stdcall *HeapAllocFn)(void*, unsigned long, unsigned long);
typedef int (__stdcall *HeapFreeFn)(void*, unsigned long, void*);
void* malloc(size_t n) {
    static GetProcessHeapFn f_gph = 0;
    static HeapAllocFn f_ha = 0;
    if (!f_gph) {
        void* m = GetModuleHandleA("kernel32.dll");
        if (!m) return 0;
        f_gph = (GetProcessHeapFn)GetProcAddress(m, "GetProcessHeap");
        f_ha = (HeapAllocFn)GetProcAddress(m, "HeapAlloc");
    }
    if (!f_gph || !f_ha) return 0;
    void* h = f_gph();
    return h ? f_ha(h, 8, n) : 0;   /* HEAP_ZERO_MEMORY=8 */
}
void free(void* p) {
    static HeapFreeFn f_hf = 0;
    if (!p) return;
    if (!f_hf) {
        void* m = GetModuleHandleA("kernel32.dll");
        if (!m) return;
        f_hf = (HeapFreeFn)GetProcAddress(m, "HeapFree");
    }
    if (!f_hf) return;
    void* h = ((GetProcessHeapFn)GetProcAddress(
        GetModuleHandleA("kernel32.dll"), "GetProcessHeap"))();
    if (h) f_hf(h, 0, p);
}

void* memcpy(void* d, const void* s, unsigned long n) {
    unsigned char* dd = d; const unsigned char* ss = s;
    for (unsigned long i = 0; i < n; i++) dd[i] = ss[i];
    return d;
}
void* memset(void* d, int c, unsigned long n) {
    unsigned char* dd = d;
    for (unsigned long i = 0; i < n; i++) dd[i] = (unsigned char)c;
    return d;
}
void ___chkstk_ms(void) {}
