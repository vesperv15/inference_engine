cmake_minimum_required(VERSION 3.20)
project(InferenceEngine CXX)

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC için AVX2
if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -mfma)
endif()

include_directories(include)

# C++ Executable test
add_executable(inference_engine_test src/main.cpp)

# Pybind11 oto indirme ve modul yapılandırması
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Python c++ dinamik kutuphanesi .pyd
pybind11_add_module(inference_engine_cpp src/bindings.cpp)