#include <iostream>
#include <chrono>
#include <random>
#include "../include/tensor.hpp"
#include "../include/ops.hpp"

void fill_random(Tensor& t) {
    std::mt19937 gen(42); // Sabit seed
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (std::size_t i = 0; i < t.rows * t.cols; ++i) {
        t.data[i] = dis(gen);
    }
}

int main() {
    constexpr std::size_t N = 512; // 512x512 Matris
    std::cout << "--- " << N << "x" << N << " Matris Çarpımı Performans Testi ---" << std::endl;

    Tensor A(N, N);
    Tensor B(N, N);
    Tensor C_naive(N, N);
    Tensor C_avx2(N, N);

    fill_random(A);
    fill_random(B);

    // naive standart test
    auto start = std::chrono::high_resolution_clock::now();
    ops::matmul_naive(A, B, C_naive);
    auto end = std::chrono::high_resolution_clock::now();
    double naive_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Standart MatMul Süresi: " << naive_ms << " ms" << std::endl;

    // avx2 sımd test
    start = std::chrono::high_resolution_clock::now();
    ops::matmul_avx2(A, B, C_avx2);
    end = std::chrono::high_resolution_clock::now();
    double avx2_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "AVX2 SIMD MatMul Süresi: " << avx2_ms << " ms" << std::endl;

    // hızlanma oranı
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "AVX2 Hızlanma Faktörü (Speedup): " << (naive_ms / avx2_ms) << "x daha hızlı!" << std::endl;

    // doğrulama aynı sonuc cıktısı alındı mı
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < N * N; ++i) {
        max_diff = std::max(max_diff, std::abs(C_naive.data[i] - C_avx2.data[i]));
    }
    std::cout << "Maksimum Fark (Doğrulama Testi): " << max_diff << std::endl;

    return 0;
}