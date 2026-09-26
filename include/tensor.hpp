#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

class Tensor {
public:
    std::size_t rows;
    std::size_t cols;
    float* data;

    Tensor(std::size_t r, std::size_t c) : rows(r), cols(c) {
#if defined(_MSC_VER)
        data = static_cast<float*>(_aligned_malloc(r * c * sizeof(float), 32));
#else
        if (posix_memalign(reinterpret_cast<void**>(&data), 32, r * c * sizeof(float)) != 0) {
            data = nullptr;
        }
#endif
        if (!data) throw std::bad_alloc();
        std::fill(data, data + (r * c), 0.0f);
    }

    ~Tensor() {
        if (data) {
#if defined(_MSC_VER)
            _aligned_free(data);
#else
            free(data);
#endif
        }
    }

    // kopyalama önle ve bellek cakısmasını önlemek için
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // move kurucusu
    Tensor(Tensor&& other) noexcept 
        : rows(other.rows), cols(other.cols), data(other.data) {
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }

    inline float& operator()(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }

    inline const float& operator()(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }

    float* raw_data() { return data; }
    const float* raw_data() const { return data; }
};