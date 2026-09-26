import sys
import os
import time

# derlenen kutuphane yolları
sys.path.append(os.path.abspath("./build/Release"))
sys.path.append(os.path.abspath("./build/Debug"))

import inference_engine_cpp as ie #type: ignore
import numpy as np 

# pytorch varsa ekle , yoksa pas geç
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
def run_benchmark():
    matrix_sizes = [128, 256, 512, 1024]
    
    print("=" * 65)
    print(f"{ 'Boyut (NxN) ': <12} / {'C++ Naive' :<12} / {'C++ AVX2' :<12} / {'Numpy':<12} / {'pytorch cpu':<12}")
    print("=" * 65)
    
    for N in matrix_sizes:
        A = ie.Tensor(N,N)
        B = ie.Tensor(N,N)
        C = ie.Tensor(N,N)
        
        np_A = A.to_numpy()
        np_B = B.to_numpy()
        np_A[:] = np.random.randn(N, N).astype(np.float32)
        np_B[:] = np.random.randn(N, N).astype(np.float32)
        #naive sadece kucuk boyutlar ıcın buyuklerde yavas kalır.
        
        if N <= 512:
            start = time.perf_counter()
            ie.matmul_naive(A, B, C)
            naive_ms = (time.perf_counter() - start) * 1000
            naive_str = f"{naive_ms:.2f} ms"
        else:
            naive_str = "Atlandı"
        
        #avx2 için
        start = time.perf_counter()
        ie.matmul_avx2(A, B, C)
        avx2_ms = (time.perf_counter() - start) * 1000
        #numpy mkl altyapılı
        start = time.perf_counter()
        _ = np.dot(np_A, np_B)
        numpy_ms = (time.perf_counter() - start) * 1000
        #pytorch cpu
        
        if HAS_TORCH:
            t_A = torch.from_numpy(np_A)
            t_B = torch.from_numpy(np_B)
            start = time.perf_counter()
            _ = torch.matmul(t_A, t_B)
            torch_ms = (time.perf_counter() - start) * 1000
            torch_str = f"{torch_ms:.2f} ms"
        else:
            torch_str = "Kurulu Değil"
        
        print(f"{f'{N}x{N}':<12} / {naive_str:<12} / {f'{avx2_ms:.2f} ms':<12} / {f'{numpy_ms:.2f} ms':<12} / {torch_str:<12}")
        
    print("=" * 65)

if __name__ == "__main__":
    run_benchmark()