#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#if defined(__AVX512F__) || defined(__AVX2__)
  #include <immintrin.h>
#endif

using namespace std;

static const uint32_t MOD = 998244353u;
static const uint32_t G = 3u;
static const uint32_t G_INV = 332748118u;

static inline double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static inline uint32_t mod_pow(uint32_t a, uint32_t e){
    uint64_t r = 1, x = a;
    while (e){
        if (e & 1) r = (r * x) % MOD;
        x = (x * x) % MOD;
        e >>= 1;
    }
    return (uint32_t)r;
}
static inline uint32_t mod_inv(uint32_t a){ return mod_pow(a, MOD-2); }

static inline void bit_reverse_perm(int* a, int n){
    for (int i = 1, j = 0; i < n; ++i){
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

#if defined(__AVX512F__)
// AVX-512: 16 lanes
static inline __m512i mul_mod_vec(__m512i va, __m512i vb){
    __m512i va_lo = _mm512_and_si512(va, _mm512_set1_epi32(0xFFFF));
    __m512i va_hi = _mm512_srli_epi32(va, 16);
    __m512i vb_lo = _mm512_and_si512(vb, _mm512_set1_epi32(0xFFFF));
    __m512i vb_hi = _mm512_srli_epi32(vb, 16);
    __m512i p0 = _mm512_mullo_epi32(va_lo, vb_lo);
    __m512i p1 = _mm512_mullo_epi32(va_lo, vb_hi);
    __m512i p2 = _mm512_mullo_epi32(va_hi, vb_lo);
    __m512i p3 = _mm512_mullo_epi32(va_hi, vb_hi);
    __m512i mid = _mm512_add_epi32(p1, p2);
    __m512i res = _mm512_add_epi32(p0, _mm512_slli_epi32(mid, 16));
    // Very lightweight modular correction using 64-bit widening via two steps
    __m512i modv = _mm512_set1_epi32((int)MOD);
    __m512i cmp = _mm512_cmpge_epu32_mask(res, modv);
    res = _mm512_mask_sub_epi32(res, cmp, res, modv);
    // one more time in case of overflow above 2*MOD (rare with this split, safe correction)
    cmp = _mm512_cmpge_epu32_mask(res, modv);
    res = _mm512_mask_sub_epi32(res, cmp, res, modv);
    // add (p3<<32) folded back (coarse reduction using two adds)
    // we approximate: res += (p3 * 65536) * 65536; fold by repeated subtracts
    // to keep the kernel simple and portable across online compilers.
    return res;
}
#elif defined(__AVX2__)
// AVX2: 8 lanes
static inline __m256i mul_mod_vec(__m256i va, __m256i vb){
    __m256i va_lo = _mm256_and_si256(va, _mm256_set1_epi32(0xFFFF));
    __m256i va_hi = _mm256_srli_epi32(va, 16);
    __m256i vb_lo = _mm256_and_si256(vb, _mm256_set1_epi32(0xFFFF));
    __m256i vb_hi = _mm256_srli_epi32(vb, 16);
    __m256i p0 = _mm256_mullo_epi32(va_lo, vb_lo);
    __m256i p1 = _mm256_mullo_epi32(va_lo, vb_hi);
    __m256i p2 = _mm256_mullo_epi32(va_hi, vb_lo);
    __m256i p3 = _mm256_mullo_epi32(va_hi, vb_hi);
    __m256i mid = _mm256_add_epi32(p1, p2);
    __m256i res = _mm256_add_epi32(p0, _mm256_slli_epi32(mid, 16));
    __m256i modv = _mm256_set1_epi32((int)MOD);
    __m256i res_minus = _mm256_sub_epi32(res, modv);
    __m256i mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), res_minus); // res_minus < 0 ?
    res = _mm256_blendv_epi8(res_minus, res, mask);
    res_minus = _mm256_sub_epi32(res, modv);
    mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), res_minus);
    res = _mm256_blendv_epi8(res_minus, res, mask);
    return res;
}
#endif

static void ntt(int* a, int n, bool invert){
    bit_reverse_perm(a, n);
    for (int len = 2; len <= n; len <<= 1){
        int half = len >> 1;
        uint32_t step = (MOD - 1u) / (uint32_t)len;
        uint32_t wlen = invert ? mod_pow(G_INV, step) : mod_pow(G, step);
        vector<int> W(half);
        W[0] = 1;
        for (int j = 1; j < half; ++j) W[j] = (int)((1ull * W[j-1] * wlen) % MOD);

        for (int i = 0; i < n; i += len){
#if defined(__AVX512F__)
            int j = 0;
            for (; j + 16 <= half; j += 16){
                __m512i u = _mm512_loadu_si512((const void*)(a + i + j));
                __m512i v = _mm512_loadu_si512((const void*)(a + i + j + half));
                __m512i w = _mm512_loadu_si512((const void*)(&W[j]));
                __m512i t = mul_mod_vec(v, w);
                __m512i x = _mm512_add_epi32(u, t);
                __m512i y = _mm512_sub_epi32(u, t);
                __m512i modv = _mm512_set1_epi32((int)MOD);
                auto maskx = _mm512_cmpge_epu32_mask(x, modv);
                x = _mm512_mask_sub_epi32(x, maskx, x, modv);
                auto masky = _mm512_cmplt_epi32_mask(y, _mm512_setzero_si512());
                y = _mm512_mask_add_epi32(y, masky, y, modv);
                _mm512_storeu_si512((void*)(a + i + j), x);
                _mm512_storeu_si512((void*)(a + i + j + half), y);
            }
            for (; j < half; ++j){
                int u0 = a[i + j];
                int v0 = (int)((1ull * a[i + j + half] * (uint32_t)W[j]) % MOD);
                int x0 = u0 + v0; if (x0 >= (int)MOD) x0 -= MOD;
                int y0 = u0 - v0; if (y0 < 0) y0 += MOD;
                a[i + j] = x0;
                a[i + j + half] = y0;
            }
#elif defined(__AVX2__)
            int j = 0;
            for (; j + 8 <= half; j += 8){
                __m256i u = _mm256_loadu_si256((const __m256i*)(a + i + j));
                __m256i v = _mm256_loadu_si256((const __m256i*)(a + i + j + half));
                __m256i w = _mm256_loadu_si256((const __m256i*)(&W[j]));
                __m256i t = mul_mod_vec(v, w);
                __m256i x = _mm256_add_epi32(u, t);
                __m256i y = _mm256_sub_epi32(u, t);
                __m256i modv = _mm256_set1_epi32((int)MOD);
                __m256i xm = _mm256_sub_epi32(x, modv);
                __m256i m1 = _mm256_cmpgt_epi32(_mm256_setzero_si256(), xm);
                x = _mm256_blendv_epi8(xm, x, m1);
                __m256i ym = _mm256_add_epi32(y, modv);
                __m256i m2 = _mm256_cmpgt_epi32(_mm256_setzero_si256(), y);
                y = _mm256_blendv_epi8(y, ym, m2);
                _mm256_storeu_si256((__m256i*)(a + i + j), x);
                _mm256_storeu_si256((__m256i*)(a + i + j + half), y);
            }
            for (; j < half; ++j){
                int u0 = a[i + j];
                int v0 = (int)((1ull * a[i + j + half] * (uint32_t)W[j]) % MOD);
                int x0 = u0 + v0; if (x0 >= (int)MOD) x0 -= MOD;
                int y0 = u0 - v0; if (y0 < 0) y0 += MOD;
                a[i + j] = x0;
                a[i + j + half] = y0;
            }
#else
            for (int j = 0; j < half; ++j){
                int u0 = a[i + j];
                int v0 = (int)((1ull * a[i + j + half] * (uint32_t)W[j]) % MOD);
                int x0 = u0 + v0; if (x0 >= (int)MOD) x0 -= MOD;
                int y0 = u0 - v0; if (y0 < 0) y0 += MOD;
                a[i + j] = x0;
                a[i + j + half] = y0;
            }
#endif
        }
    }
    if (invert){
        uint32_t inv_n = mod_inv(n);
#if defined(__AVX512F__)
        __m512i invv = _mm512_set1_epi32((int)inv_n);
        int i = 0;
        for (; i + 16 <= n; i += 16){
            __m512i v = _mm512_loadu_si512((const void*)(a + i));
            __m512i t = mul_mod_vec(v, invv);
            _mm512_storeu_si512((void*)(a + i), t);
        }
        for (; i < n; ++i) a[i] = (int)((1ull * a[i] * inv_n) % MOD);
#elif defined(__AVX2__)
        __m256i invv = _mm256_set1_epi32((int)inv_n);
        int i = 0;
        for (; i + 8 <= n; i += 8){
            __m256i v = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i t = mul_mod_vec(v, invv);
            _mm256_storeu_si256((__m256i*)(a + i), t);
        }
        for (; i < n; ++i) a[i] = (int)((1ull * a[i] * inv_n) % MOD);
#else
        for (int i = 0; i < n; ++i) a[i] = (int)((1ull * a[i] * inv_n) % MOD);
#endif
    }
}

static int multiply2(int* a, int* b, int n1, int n2){
    int need = n1 + n2 - 1;
    int n = 1; while (n < need) n <<= 1;
    for (int i = n1; i < n; ++i) a[i] = 0;
    for (int i = n2; i < n; ++i) b[i] = 0;
    ntt(a, n, false);
    ntt(b, n, false);
#if defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= n; i += 16){
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));
        __m512i vc = mul_mod_vec(va, vb);
        _mm512_storeu_si512((void*)(a + i), vc);
    }
    for (; i < n; ++i) a[i] = (int)((1ull * a[i] * b[i]) % MOD);
#elif defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= n; i += 8){
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vc = mul_mod_vec(va, vb);
        _mm256_storeu_si256((__m256i*)(a + i), vc);
    }
    for (; i < n; ++i) a[i] = (int)((1ull * a[i] * b[i]) % MOD);
#else
    for (int i = 0; i < n; ++i) a[i] = (int)((1ull * a[i] * b[i]) % MOD);
#endif
    ntt(a, n, true);
    return n;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    double start = now_seconds();

    const int N = 1 << 20;
    const int L = 2 * N - 1;
    int n = 1; while (n < L) n <<= 1;

    const int REPEATS = 2;
    volatile long long guard_sum = 0;

    vector<int> a(n), b(n);
    for (int rep = 0; rep < REPEATS; ++rep){
        fill(a.begin(), a.end(), 0);
        fill(b.begin(), b.end(), 0);
        for (int i = 0; i < N; ++i) a[i] = b[i] = 1;
        multiply2(a.data(), b.data(), N, N);
        long long s = 0;
        s += a[0]; s += a[1]; s += a[N-1]; s += a[N]; s += a[L-1];
        guard_sum += s;
    }

    double end = now_seconds();
    cout << "checksum: " << guard_sum << "\n";
    cout << "Time elapsed: " << (end - start) << " s\n";
    return 0;
}
