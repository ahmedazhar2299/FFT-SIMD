# FFT-SIMD

This repository provides an implementation of Fast Fourier Transform (FFT) and Number Theoretic Transform (NTT) for efficient polynomial multiplication. It demonstrates how to construct large integer convolutions using both complex FFT (floating-point) and modular NTT (finite field arithmetic). The project showcases techniques for handling large input sizes, working with NTT-friendly primes (such as 998244353), and benchmarking performance on modern CPUs.

---

## Installation

```bash
brew install imagemagick

# Compile the image processing source code
clang++ -O3 -std=c++17 fft.cpp -o fft

# Run the program
./fft
```

## References
- [geekpradd / Fast-Fourier-Transform](https://github.com/geekpradd/Fast-Fourier-Transform/tree/master)
