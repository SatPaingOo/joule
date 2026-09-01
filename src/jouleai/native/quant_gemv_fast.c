/* joule_kernel.c — fused Q4-dequant GEMV for MoE expert streaming.
 *
 * Computes out[m] = dequant_q4(packed, scales) @ x[d] without materialising
 * the dequantised weight matrix: packed bytes are unpacked in registers,
 * int4 values are de-biased (-8), group scales (fp16) applied per group of
 * `group` columns. Compile with -O3 -mcpu=native for SIMD auto-vectorisation.
 */

#include <stdint.h>
#include <stddef.h>

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
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

void q4_gemv_f32(float *out, const float *x, const unsigned char *packed,
                 const unsigned short *scales, int m, int d, int group) {
    const int nb = d >> 1;            /* bytes per row (2 nibbles each)    */
    const int gbytes = group >> 1;    /* packed bytes per scale group      */
    for (int i = 0; i < m; i++) {
        const unsigned char *row = packed + (size_t)i * (size_t)nb;
        const unsigned short *sc = scales + (size_t)i * (size_t)(d / group);
        float acc = 0.0f;
        for (int gb = 0; gb < nb; gb += gbytes) {
            const float s = fp16_to_f32(sc[gb / gbytes]);
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
        out[i] = acc;
    }
}
int _DllMainCRTStartup(void* a, unsigned long b, void* c){return 1;}

/* threading: simple win32 threads via kernel32, no CRT */
typedef unsigned long (__attribute__((stdcall)) *LPTHREAD_START_ROUTINE)(void *);
__attribute__((stdcall)) void *kernel32_GetProcAddress_stub(void){return 0;}
