# SIMD-Accelerated C++20 Inference Engine

A lightweight, high-performance C++20 matrix operations engine focused on hardware-level parallel execution, aligned memory management, and zero-copy Python interoperability.

## Key Features

- **32-Byte Aligned Memory Allocation:** Custom memory allocation (`_aligned_malloc`) ensuring vector alignment required for optimal 256-bit AVX2 register loading.
- **AVX2 SIMD Vectorization:** Exploits 256-bit Fused Multiply-Add (`_mm256_fmadd_ps`) intrinsics to process 8 single-precision float operations per cycle.
- **Zero-Copy Pybind11 Bindings:** Exposes C++ tensor memory directly to Python via NumPy's Buffer Protocol without buffer copying or memory reallocation.

## Build & Usage

- C++20 compliant compiler (MSVC, GCC, or Clang)
- CMake 3.20+
- Python 3.8+ with numpy
 
## Benchmark Results

Performance evaluation across float32 matrix multiplication ($N \times N$) comparing naive C++ execution, custom AVX2 SIMD kernel, NumPy, and PyTorch (CPU):

| Matrix Size | C++ Naive | C++ AVX2 (Single-Thread) | NumPy (MKL/OpenBLAS) | PyTorch CPU |
| :--- | :--- | :--- | :--- | :--- |
| **128x128** | 11.77 ms | 1.93 ms | 0.21 ms | 1.59 ms |
| **256x256** | 118.44 ms | 20.27 ms | 0.43 ms | 0.43 ms |
| **512x512** | 1039.10 ms | 130.36 ms | 1.22 ms | 1.22 ms |
| **1024x1024** | *Skipped* | 1150.12 ms | 4.73 ms | 6.44 ms |

*Note: The single-threaded AVX2 kernel achieves ~8x speedup over scalar C++ execution. Industrial frameworks (NumPy/PyTorch) scale further on large matrices by utilizing CPU multi-threading and cache-tiling techniques.*

## Project Structure

```text
├── include/
│   ├── tensor.hpp       # 32-byte aligned Tensor class
│   └── ops.hpp          # Naive & AVX2 MatMul kernels
├── src/
│   ├── main.cpp         # Standalone C++ test runner
│   └── bindings.cpp     # Pybind11 zero-copy interface
├── CMakeLists.txt       # Build configuration
├── test.py              # Integration & zero-copy test
└── benchmark.py         # Comparative performance suite
