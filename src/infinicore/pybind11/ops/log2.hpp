#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/log2.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_log2(py::module &m) {
    m.def("log2",
          &op::log2,
          py::arg("x"),
          R"doc(Log2 activation function.)doc");

    m.def("log2_",
          &op::log2_,
          py::arg("y"),
          py::arg("x"),
          R"doc(In-place Log2 activation function.)doc");
}

} // namespace infinicore::ops
