# FFT-SIMD

This repository provides an implementation of **Fast Fourier Transform (FFT)** and **Number Theoretic Transform (NTT)** for efficient polynomial multiplication. It demonstrates how to construct large integer convolutions using both complex FFT (floating-point) and modular NTT (finite field arithmetic). The project showcases techniques for handling large input sizes, working with NTT-friendly primes (such as **998244353**), and benchmarking performance on modern CPUs.

In addition, the implementation leverages **SIMD vectorization**:
- **AVX2 / AVX-512** on Intel/AMD  
- **NEON** on Apple Silicon (M1/M2/M3)  

to accelerate butterfly operations inside FFT/NTT. By parallelizing modular arithmetic across multiple data lanes, the SIMD version achieves more than **4× speedup** compared to the scalar baseline.  

This makes the project suitable for high-performance applications in:
- Cryptography  
- Error-correcting codes  
- Signal and image processing  
- Large-scale polynomial computations  

---

## Installation

```bash
# Compile the image processing source code
clang++ -O3 -std=c++17 -mcpu=native fft.cpp -o fft

# Run the program
./fft
```

## References
- [geekpradd / Fast-Fourier-Transform](https://github.com/geekpradd/Fast-Fourier-Transform/tree/master)
