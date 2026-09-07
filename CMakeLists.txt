cmake_minimum_required(VERSION 3.18)

project(pwlopt LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    _poisson_disk
    cpp/poisson_disk.cpp
    cpp/poisson_disk_bindings.cpp
)

target_compile_features(
    _poisson_disk
    PRIVATE
    cxx_std_17
)

if(MSVC)
    target_compile_options(
        _poisson_disk
        PRIVATE
        /O2
    )
else()
    target_compile_options(
        _poisson_disk
        PRIVATE
        -O3
        -march=native
    )
endif()


install(
    TARGETS _poisson_disk
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/src/pwlopt/fitting"
    RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/src/pwlopt/fitting"
)