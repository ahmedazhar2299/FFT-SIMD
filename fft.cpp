#include <arm_neon.h>
#include <iostream>
#include <vector>
#include <thread>
#include <cstdint>
#include <chrono>
#include <algorithm>
using namespace std;

static const uint32_t MOD = 998244353u;
static const uint32_t G   = 3u;
static const uint32_t G_INV = 332748118u;

static inline double now_seconds() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    return duration<double>(steady_clock::now() - t0).count();
}
static inline uint32_t mod_pow(uint32_t a, uint32_t e){
    uint64_t r = 1, x = a;
    while (e){ if (e & 1) r = (r * x) % MOD; x = (x * x) % MOD; e >>= 1; }
    return (uint32_t)r;
}
static inline uint32_t mod_inv(uint32_t a){ return mod_pow(a, MOD-2); }

// ---- Montgomery (radix 2^32) ----
static inline uint32_t mont_n0(uint32_t p){
    uint64_t inv = 1;
    for (int i=0;i<5;++i) inv *= (2 - (uint64_t)p * inv);
    return (uint32_t)(0u - (uint32_t)inv);
}
static inline uint32_t mont_R2(uint32_t p){
    __uint128_t R = ((__uint128_t)1 << 32) % p;
    return (uint32_t)((R * R) % p);
}
static inline uint32_t mont_scalar(uint32_t a, uint32_t b, uint32_t p, uint32_t n0){
    uint64_t t = (uint64_t)a * b;
    uint32_t m = (uint32_t)t * n0;
    uint64_t s = t + (uint64_t)m * p;
    uint32_t r = (uint32_t)(s >> 32);
    return (r >= p) ? r - p : r;
}
static inline uint32x4_t mont4(uint32x4_t a, uint32x4_t b, uint32_t p, uint32_t n0){
    uint32x2_t a0 = vget_low_u32(a),  a1 = vget_high_u32(a);
    uint32x2_t b0 = vget_low_u32(b),  b1 = vget_high_u32(b);
    uint64x2_t t0 = vmull_u32(a0, b0);
    uint64x2_t t1 = vmull_u32(a1, b1);
    uint32x2_t m0 = vmovn_u64(t0);
    uint32x2_t m1 = vmovn_u64(t1);
    uint32x2_t n0v = vdup_n_u32(n0);
    uint64x2_t u0 = vmull_u32(m0, n0v);
    uint64x2_t u1 = vmull_u32(m1, n0v);
    uint32x2_t pv = vdup_n_u32(p);
    uint64x2_t k0 = vmull_u32(vmovn_u64(u0), pv);
    uint64x2_t k1 = vmull_u32(vmovn_u64(u1), pv);
    uint64x2_t s0 = vaddq_u64(t0, k0);
    uint64x2_t s1 = vaddq_u64(t1, k1);
    uint32x2_t r0 = vshrn_n_u64(s0, 32);
    uint32x2_t r1 = vshrn_n_u64(s1, 32);
    uint32x4_t r  = vcombine_u32(r0, r1);
    uint32x4_t P  = vdupq_n_u32(p);
    uint32x4_t r_minus = vsubq_u32(r, P);
    uint32x4_t ge = vcgeq_u32(r, P);
    return vbslq_u32(ge, r_minus, r);
}

// ---- bit-reverse ----
static inline void bit_reverse_perm(uint32_t* a, int n){
    for (int i = 1, j = 0; i < n; ++i){
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
}

// ---- domain conversion ----
static inline void to_mont(uint32_t* a, int n, uint32_t p, uint32_t n0, uint32_t R2){
    uint32x4_t R2v = vdupq_n_u32(R2);
    int i = 0;
    for (; i + 4 <= n; i += 4){
        uint32x4_t x = vld1q_u32(a + i);
        vst1q_u32(a + i, mont4(x, R2v, p, n0));
    }
    for (; i < n; ++i) a[i] = mont_scalar(a[i], R2, p, n0);
}
static inline void from_mont(uint32_t* a, int n, uint32_t p, uint32_t n0){
    uint32x4_t ONE = vdupq_n_u32(1);
    int i = 0;
    for (; i + 4 <= n; i += 4){
        uint32x4_t x = vld1q_u32(a + i);
        vst1q_u32(a + i, mont4(x, ONE, p, n0));
    }
    for (; i < n; ++i) a[i] = mont_scalar(a[i], 1, p, n0);
}

// ---- twiddles (Montgomery) ----
static inline void build_twiddles_mont(vector<uint32_t>& Wm, int half, bool invert,
                                       uint32_t p, uint32_t n0, uint32_t R2){
    uint32_t step = (p - 1u) / (uint32_t)(half << 1);
    uint32_t wlen = invert ? mod_pow(G_INV, step) : mod_pow(G, step);
    uint32_t oneR = mont_scalar(1, R2, p, n0);
    uint32_t wR   = mont_scalar(wlen, R2, p, n0);
    Wm.resize(half);
    Wm[0] = oneR;
    for (int j = 1; j < half; ++j) Wm[j] = mont_scalar(Wm[j-1], wR, p, n0);
}

// ---- parallel NTT: threads across blocks per stage ----
static void ntt_parallel(uint32_t* a, int n, bool invert, int num_threads){
    const uint32_t p  = MOD;
    const uint32_t n0 = mont_n0(p);
    const uint32_t R2 = mont_R2(p);

    bit_reverse_perm(a, n);

    // clamp thread count to available work
    num_threads = max(1, num_threads);

    for (int len = 2; len <= n; len <<= 1){
        int half = len >> 1;
        vector<uint32_t> Wm; Wm.reserve(half);
        build_twiddles_mont(Wm, half, invert, p, n0, R2);

        const int blocks = n / len;
        const int T = min(num_threads, blocks);
        if (T <= 1){
            // single-thread fallback
            for (int i = 0; i < n; i += len){
                int j = 0;
                for (; j + 4 <= half; j += 4){
                    uint32x4_t u = vld1q_u32(a + i + j);
                    uint32x4_t v = vld1q_u32(a + i + j + half);
                    uint32x4_t w = vld1q_u32(&Wm[j]);
                    uint32x4_t t = mont4(v, w, p, n0);
                    uint32x4_t P = vdupq_n_u32(p);
                    uint32x4_t x = vaddq_u32(u, t);
                    uint32x4_t xm = vsubq_u32(x, P);
                    x = vbslq_u32(vcgeq_u32(x, P), xm, x);
                    uint32x4_t y = vsubq_u32(u, t);
                    uint32x4_t add = vbslq_u32(vcgtq_u32(t, u), P, vdupq_n_u32(0));
                    y = vaddq_u32(y, add);
                    vst1q_u32(a + i + j, x);
                    vst1q_u32(a + i + j + half, y);
                }
                for (; j < half; ++j){
                    uint32_t u0 = a[i + j];
                    uint32_t t  = mont_scalar(a[i + j + half], Wm[j], p, n0);
                    uint32_t x0 = u0 + t; if (x0 >= p) x0 -= p;
                    uint32_t y0 = (u0 >= t) ? (u0 - t) : (u0 + p - t);
                    a[i + j] = x0; a[i + j + half] = y0;
                }
            }
            continue;
        }

        // multi-thread: split block indices evenly
        vector<thread> workers;
        workers.reserve(T);
        auto worker = [&](int tid){
            int blocks_per = (blocks + T - 1) / T;
            int b0 = tid * blocks_per;
            int b1 = min(blocks, b0 + blocks_per);
            if (b0 >= b1) return;

            for (int b = b0; b < b1; ++b){
                int i = b * len;
                int j = 0;
                for (; j + 4 <= half; j += 4){
                    uint32x4_t u = vld1q_u32(a + i + j);
                    uint32x4_t v = vld1q_u32(a + i + j + half);
                    uint32x4_t w = vld1q_u32(&Wm[j]);
                    uint32x4_t t = mont4(v, w, p, n0);
                    uint32x4_t P = vdupq_n_u32(p);
                    uint32x4_t x = vaddq_u32(u, t);
                    uint32x4_t xm = vsubq_u32(x, P);
                    x = vbslq_u32(vcgeq_u32(x, P), xm, x);
                    uint32x4_t y = vsubq_u32(u, t);
                    uint32x4_t add = vbslq_u32(vcgtq_u32(t, u), P, vdupq_n_u32(0));
                    y = vaddq_u32(y, add);
                    vst1q_u32(a + i + j, x);
                    vst1q_u32(a + i + j + half, y);
                }
                for (; j < half; ++j){
                    uint32_t u0 = a[i + j];
                    uint32_t t  = mont_scalar(a[i + j + half], Wm[j], p, n0);
                    uint32_t x0 = u0 + t; if (x0 >= p) x0 -= p;
                    uint32_t y0 = (u0 >= t) ? (u0 - t) : (u0 + p - t);
                    a[i + j] = x0; a[i + j + half] = y0;
                }
            }
        };

        for (int t = 0; t < T; ++t) workers.emplace_back(worker, t);
        for (auto& th : workers) th.join();
    }

    if (invert){
        const uint32_t p  = MOD;
        const uint32_t n0 = mont_n0(p);
        const uint32_t R2 = mont_R2(p);

        uint32_t inv_n = mod_inv(n);
        uint32_t invR  = mont_scalar(inv_n, R2, p, n0);

        // parallel inverse scaling (embarrassingly parallel)
        int T = max(1, min<int>(num_threads, n / 1024)); // coarse chunks
        vector<thread> workers;
        workers.reserve(T);
        auto worker = [&](int tid){
            int chunk = (n + T - 1) / T;
            int i0 = tid * chunk;
            int i1 = min(n, i0 + chunk);
            uint32x4_t invv = vdupq_n_u32(invR);
            int i = i0;
            for (; i + 4 <= i1; i += 4){
                uint32x4_t x = vld1q_u32(a + i);
                vst1q_u32(a + i, mont4(x, invv, p, n0));
            }
            for (; i < i1; ++i) a[i] = mont_scalar(a[i], invR, p, n0);
        };
        for (int t=0; t<T; ++t) workers.emplace_back(worker, t);
        for (auto& th : workers) th.join();

        from_mont(a, n, p, n0);
    }
}

static int convolve_parallel(uint32_t* a, uint32_t* b, int n1, int n2, int num_threads){
    int need = n1 + n2 - 1, n = 1; while (n < need) n <<= 1;
    // modulus 998244353 supports n <= 2^23
    if (n > (1<<23)) {
        cerr << "NTT length too large for MOD 998244353. Clamping.\n";
        n = 1<<23;
    }
    const uint32_t p = MOD, n0 = mont_n0(p), R2 = mont_R2(p);
    for (int i = n1; i < n; ++i) a[i] = 0;
    for (int i = n2; i < n; ++i) b[i] = 0;
    to_mont(a, n, p, n0, R2);
    to_mont(b, n, p, n0, R2);
    ntt_parallel(a, n, false, num_threads);
    ntt_parallel(b, n, false, num_threads);

    // pointwise mul (parallel)
    int T = max(1, min<int>(num_threads, n / 4096));
    vector<thread> workers;
    workers.reserve(T);
    auto worker = [&](int tid){
        int chunk = (n + T - 1) / T;
        int i0 = tid * chunk, i1 = min(n, i0 + chunk);
        int i = i0;
        for (; i + 4 <= i1; i += 4){
            uint32x4_t va = vld1q_u32(a + i);
            uint32x4_t vb = vld1q_u32(b + i);
            vst1q_u32(a + i, mont4(va, vb, p, n0));
        }
        for (; i < i1; ++i) a[i] = mont_scalar(a[i], b[i], p, n0);
    };
    for (int t=0; t<T; ++t) workers.emplace_back(worker, t);
    for (auto& th : workers) th.join();

    ntt_parallel(a, n, true, num_threads);
    return n;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Choose a VALID N for MOD=998244353: max N = 1<<22 (so that n <= 1<<23)
    int N = 1 << 26;          // try 1<<20 .. 1<<22
    int L = 2 * N - 1;
    int n = 1; while (n < L) n <<= 1; // should be <= 1<<23

    // threads: use performance cores on M2 Pro (e.g., 6 or 8).
    int threads = max(1u, thread::hardware_concurrency());
    // You can tune: threads = 6; // for 6 P-cores on some M2 Pro configs

    vector<uint32_t> A(n), B(n);
    const int REPEATS = 2;
    volatile unsigned long long guard = 0;

    double t0 = now_seconds();
    for (int rep=0; rep<REPEATS; ++rep){
        fill(A.begin(), A.end(), 0);
        fill(B.begin(), B.end(), 0);
        for (int i = 0; i < N; ++i) A[i] = B[i] = 1;
        convolve_parallel(A.data(), B.data(), N, N, threads);
        unsigned long long s = 0;
        s += A[0]; s += A[1]; s += A[N-1]; s += A[N]; s += A[L-1];
        guard += s;
    }
    double t1 = now_seconds();

    cout << "threads: " << threads << "\n";
    cout << "checksum: " << guard << "\n";
    cout << "Time elapsed: " << (t1 - t0) << " s\n";
    return 0;
}
