#pragma once
#include "tensor.hpp"
#include <immintrin.h> // AVX2 fonk için
#include <stdexcept>
#include <algorithm>

namespace ops {

inline void matmul_naive (const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
        throw std::invalid_argument("matris boyutları uyumsuz!");
    }

    std::fill(C.data, C.data + (C.rows * C.cols), 0.0f);

    for (std::size_t i=0; i< A.rows; ++i){
        for(std::size_t k=0; k< A.cols; ++k){
            for(std::size_t j=0; j< B.cols; ++j){
                C(i, j) += A(i,k) * B(k, j);
            }
        }
    }
}

inline void matmul_avx2(const Tensor& A, const Tensor& B, Tensor& C){
    if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols){
        throw std::invalid_argument("matris boyutları uyumsuz");
    }

    std::size_t M= A.rows;
    std::size_t K= A.cols;
    std::size_t N= B.cols;

    //c matirisini sıfırlıuotuz

    std::fill(C.data, C.data + (M * N), 0.0f);

    for (std:: size_t i=0; i< M; ++i){
        for (std::size_t k=0; k< K; ++k){
            __m256 a_ik = _mm256_set1_ps(A(i,k));

            std::size_t j=0;

            for (; j + 7 < N; j +=8) {
                __m256 b_vec = _mm256_load_ps(&B(k,j));

                __m256 c_vec = _mm256_load_ps(&C(i, j));

                c_vec = _mm256_fmadd_ps(a_ik, b_vec, c_vec);

                _mm256_store_ps(&C(i, j), c_vec);
            }

            for (;j < N; ++j) {
                C(i,j) += A(i, k) * B(k, j);
            }
        }
    }
}

} //namespace ops