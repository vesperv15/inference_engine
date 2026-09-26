#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../include/tensor.hpp"
#include "../include/ops.hpp"

namespace py=pybind11;

PYBIND11_MODULE(inference_engine_cpp, m) {
    m.doc() = "AVX2 SIMD hızlandırmalı c++ cıkarım motoru";

    // c++ tensır sınıfının python a aktarımı
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::size_t, std::size_t>())
        .def_readonly("rows", &Tensor::rows)
        //numpy buffer protocol ile zero copy veri erişimi

        .def("to_numpy", [](Tensor& t){
            return py::array_t<float>{
                {t.rows, t.cols},
                {t.cols * sizeof(float), sizeof(float)},
                t.raw_data(),
                py::cast(t)
            };
        });
    
    m.def("matmul_avx2", &ops::matmul_avx2, "avx2 destekli matris carpımı");
    m.def("matmul_naive", &ops::matmul_naive, "standart matris carpımu");
}