#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/mish.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_mish(py::module &m) {
    m.def("mish",
          &op::mish,
          py::arg("x"),
          R"doc(Mish activation function.)doc");

    m.def("mish_",
          &op::mish_,
          py::arg("y"),
          py::arg("x"),
          R"doc(In-place Mish activation function.)doc");
}

} // namespace infinicore::ops
