#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <iomanip>
using namespace std;

using cd = complex<double>;
const double PI = acos(-1.0);

static inline double now_seconds() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    return duration<double>(steady_clock::now() - t0).count();
}

static void fft(vector<cd>& a, bool invert) {
    const int n = (int)a.size();

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }

    // Iterative FFT
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                cd u = a[i + j];
                cd v = w * a[i + j + len / 2];
                a[i + j]             = u + v;
                a[i + j + len / 2]   = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (int i = 0; i < n; ++i) a[i] /= (double)n;
    }
}

static vector<long long> multiply_fft(const vector<int>& A, const vector<int>& B) {
    int n1 = (int)A.size(), n2 = (int)B.size();
    int n = 1;
    while (n < n1 + n2 - 1) n <<= 1;

    vector<cd> fa(n), fb(n);
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

    const int N = 1 << 22; // 2^X
    vector<int> a(N, 1), b(N, 1);

    auto c = multiply_fft(a, b);

    cout << "c[0..4]: ";
    for (int i = 0; i < 5 && i < (int)c.size(); ++i) cout << c[i] << (i+1<5?" ":"\n");

    if ((int)c.size() > N) {
        cout << "c[N-2], c[N-1], c[N]: " << c[N-2] << " " << c[N-1] << " " << c[N] << "\n";
    }

    return 0;
}
