import sys
import os

# .pyd dosyalarının yolları burada
sys.path.append(os.path.abspath("./build/Release"))
sys.path.append(os.path.abspath("./build/Debug"))

import inference_engine_cpp as ie  # type: ignore
import numpy as np
import time

print("--- Python & C++ SIMD Inference Engine Entegrasyon Testi ---")

N = 512
A = ie.Tensor(N, N)
B = ie.Tensor(N, N)
C = ie.Tensor(N, N)

# C++ de hizlanmıs bellegi kopyalamadan numpy array ile alalım
np_A = A.to_numpy()
np_B = B.to_numpy()

# python da rastgele sayı ile dolduruyoruz dogrudan c++ e yazılır
np_A[:] = np.random.randn(N, N).astype(np.float32)
np_B[:] = np.random.randn(N, N).astype(np.float32)

# C++ AVX2 matmul cagırısı
start = time.perf_counter()
ie.matmul_avx2(A, B, C)
end = time.perf_counter()

print(f"C++ AVX2 MatMul Python Çağrı Süresi: {(end - start) * 1000:.2f} ms")

# numpy ile sonuc dogrulama
np_C = C.to_numpy()
np_ref = np.dot(np_A, np_B)
diff = np.max(np.abs(np_C - np_ref))
print(f"NumPy ve C++ Çıktısı Arasındaki Maksimum Fark: {diff:.6f}")