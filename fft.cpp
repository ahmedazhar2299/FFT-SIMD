#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

static const int MOD = 998244353;
static const int g = 3;
static const int g_inv = 332748118; // 3^{-1} mod MOD

double now_seconds(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);                 // C11 standard
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static inline long long mod_pow(long long a, long long e){
    long long r = 1;
    while (e){
        if (e & 1) r = (r * a) % MOD;
        a = (a * a) % MOD;
        e >>= 1;
    }
    return r;
}
static inline int inverse(int a){ return (int)mod_pow(a, MOD-2); }

static void ntt(int *a, int n, bool invert){
    // bit-reversal
    for (int i = 1, j = 0; i < n; ++i){
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1){
        int val = (MOD - 1) / len;
        int wlen = (int)(invert ? mod_pow(g_inv, val) : mod_pow(g, val));
        for (int i = 0; i < n; i += len){
            int w = 1;
            for (int j = 0; j < len/2; ++j){
                int u = a[i+j];
                int v = (int)(1LL * w * a[i+j+len/2] % MOD);
                int x = u + v; if (x >= MOD) x -= MOD;
                int y = u - v; if (y < 0) y += MOD;
                a[i+j] = x;
                a[i+j+len/2] = y;
                w = (int)(1LL * w * wlen % MOD);
            }
        }
    }
    if (invert){
        int inv_n = inverse(n);
        for (int i = 0; i < n; ++i) a[i] = (int)(1LL * a[i] * inv_n % MOD);
    }
}

static int multiply2(int* a, int* b, int n1, int n2){
    int n = 1;
    while (n < n1 + n2 - 1) n <<= 1;
    for (int i = n1; i < n; ++i) a[i] = 0;
    for (int i = n2; i < n; ++i) b[i] = 0;
    ntt(a, n, false);
    ntt(b, n, false);
    for (int i = 0; i < n; ++i) a[i] = (int)(1LL * a[i] * b[i] % MOD);
    ntt(a, n, true);
    return n;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    double start = now_seconds();

    const int N = 1 << 24;
    const int L = 2 * N - 1;
    int n = 1; while (n < L) n <<= 1;

    // keep REPEATS small during development to avoid long runs
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
    printf("\n###############################\n");
    printf("#                             #\n");
    printf("#                             #\n");
    printf("### Time elapsed: %.3f s ####\n", end - start);
    printf("#                             #\n");
    printf("#                             #\n");
    printf("###############################\n\n");
    
}
