#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "poisson_disk.hpp"

namespace py = pybind11;

namespace {

py::array_t<double> poisson_disk_sample_py(
    py::array_t<double,
                py::array::c_style |
                py::array::forcecast> points,
    double radius,
    double active_darts,
    int max_level,
    std::uint64_t seed
) {
    auto buffer = points.request();

    if (buffer.ndim != 2 ||
        buffer.shape[0] != 3 ||
        buffer.shape[1] != 2) {
        throw std::runtime_error(
            "triangle must have shape (3, 2)"
        );
    }

    const auto* p =
        static_cast<const double*>(buffer.ptr);

    pwlopt::Triangle2 triangle{
        {p[0], p[1]},
        {p[2], p[3]},
        {p[4], p[5]}
    };

    auto result =
        pwlopt::poisson_disk_sample(
            triangle,
            radius,
            active_darts,
            max_level,
            seed
        );

    std::vector<py::ssize_t> shape{
        static_cast<py::ssize_t>(result.size()),
        2
    };

    py::array_t<double> output(shape);

    auto out = output.mutable_unchecked<2>();

    for (py::ssize_t i = 0;
         i < static_cast<py::ssize_t>(result.size());
         ++i) {
        out(i, 0) = result[i].x;
        out(i, 1) = result[i].y;
    }

    return output;
}

}  // namespace


PYBIND11_MODULE(_poisson_disk, m) {

    m.doc() =
        "C++ Poisson-disk sampling";

    m.def(
        "sample",
        &poisson_disk_sample_py,
        py::arg("triangle"),
        py::arg("radius"),
        py::arg("active_darts") = 0.5,
        py::arg("max_level") = 10,
        py::arg("seed") = 0
    );
}