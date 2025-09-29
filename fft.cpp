// fft_neon_double.cpp
#include <arm_neon.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
using namespace std;

static inline double now_seconds() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    return duration<double>(steady_clock::now() - t0).count();
}

static inline void bit_reverse_perm_soa(double* RE, double* IM, int n) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            swap(RE[i], RE[j]);
            swap(IM[i], IM[j]);
        }
    }
}

static inline void build_twiddles(int len, bool invert, vector<double>& WR, vector<double>& WI) {
    const int half = len >> 1;
    WR.resize(half);
    WI.resize(half);
    const double ang = 2.0 * M_PI / len * (invert ? -1.0 : 1.0);
    WR[0] = 1.0;
    WI[0] = 0.0;
    double c = cos(ang), s = sin(ang);
    double wr = 1.0, wi = 0.0;
    for (int j = 1; j < half; ++j) {
        double tr = wr*c - wi*s;
        double ti = wr*s + wi*c;
        WR[j] = tr; WI[j] = ti;
        wr = tr; wi = ti;
    }
}

static void fft_neon_soa(double* RE, double* IM, int n, bool invert) {
    bit_reverse_perm_soa(RE, IM, n);

    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;

        vector<double> WR, WI;
        build_twiddles(len, invert, WR, WI);

        for (int i = 0; i < n; i += len) {
            int j = 0;

            for (; j + 2 <= half; j += 2) {
                float64x2_t wr = vld1q_f64(&WR[j]);  
                float64x2_t wi = vld1q_f64(&WI[j]);  

                float64x2_t ur = vld1q_f64(&RE[i + j]);       
                float64x2_t ui = vld1q_f64(&IM[i + j]);       
                float64x2_t vr = vld1q_f64(&RE[i + j + half]); 
                float64x2_t vi = vld1q_f64(&IM[i + j + half]); 

                float64x2_t tr = vsubq_f64(vmulq_f64(wr, vr), vmulq_f64(wi, vi));
                float64x2_t ti = vaddq_f64(vmulq_f64(wr, vi), vmulq_f64(wi, vr));

                float64x2_t xr = vaddq_f64(ur, tr);
                float64x2_t xi = vaddq_f64(ui, ti);
                float64x2_t yr = vsubq_f64(ur, tr);
                float64x2_t yi = vsubq_f64(ui, ti);

                vst1q_f64(&RE[i + j],        xr);
                vst1q_f64(&IM[i + j],        xi);
                vst1q_f64(&RE[i + j + half], yr);
                vst1q_f64(&IM[i + j + half], yi);
            }

            for (; j < half; ++j) {
                double ur = RE[i + j], ui = IM[i + j];
                double vr = RE[i + j + half], vi = IM[i + j + half];
                double wr = WR[j], wi = WI[j];
                double tr = wr*vr - wi*vi;
                double ti = wr*vi + wi*vr;
                RE[i + j]         = ur + tr;
                IM[i + j]         = ui + ti;
                RE[i + j + half]  = ur - tr;
                IM[i + j + half]  = ui - ti;
            }
        }
    }

    if (invert) {
        const double invn = 1.0 / double(n);
        int i = 0;
        float64x2_t s = vdupq_n_f64(invn);
        for (; i + 2 <= n; i += 2) {
            float64x2_t r = vld1q_f64(&RE[i]);
            float64x2_t m = vld1q_f64(&IM[i]);
            vst1q_f64(&RE[i], vmulq_f64(r, s));
            vst1q_f64(&IM[i], vmulq_f64(m, s));
        }
        for (; i < n; ++i) {
            RE[i] *= invn; IM[i] *= invn;
        }
    }
}

static void fft(vector<complex<double>>& a, bool invert) {
    const int n = (int)a.size();
    vector<double> RE(n), IM(n);
    for (int i = 0; i < n; ++i) { RE[i] = a[i].real(); IM[i] = a[i].imag(); }

    fft_neon_soa(RE.data(), IM.data(), n, invert);

    for (int i = 0; i < n; ++i) a[i] = complex<double>(RE[i], IM[i]);
}

static vector<long long> multiply_fft(const vector<int>& A, const vector<int>& B) {
    int n1 = (int)A.size(), n2 = (int)B.size();
    int n = 1; while (n < n1 + n2 - 1) n <<= 1;

    vector<complex<double>> fa(n), fb(n);
    for (int i = 0; i < n1; ++i) fa[i] = (double)A[i];
    for (int i = 0; i < n2; ++i) fb[i] = (double)B[i];

    double t0 = now_seconds();
    fft(fa, false);
    fft(fb, false);
    for (int i = 0; i < n; ++i) fa[i] *= fb[i];
    fft(fa, true);
    double t1 = now_seconds();

    vector<long long> C(n1 + n2 - 1);
    for (int i = 0; i < (int)C.size(); ++i) C[i] = llround(fa[i].real());

    cout << fixed << setprecision(6)
         << "FFT convolution elapsed: " << (t1 - t0) << " s\n";
    return C;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const int N = 1 << 22; // power of two
    vector<int> a(N, 1), b(N, 1);

    auto c = multiply_fft(a, b);

    cout << "c[0..4]: ";
    for (int i = 0; i < 5 && i < (int)c.size(); ++i) cout << c[i] << (i+1<5?" ":"\n");

    if ((int)c.size() > N) {
        cout << "c[N-2], c[N-1], c[N]: " << c[N-2] << " " << c[N-1] << " " << c[N] << "\n";
    }
    return 0;
}
