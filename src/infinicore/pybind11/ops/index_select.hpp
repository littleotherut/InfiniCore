#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/index_select.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_index_select(py::module &m) {
    m.def("index_select",
          &op::index_select,
          py::arg("x"),
          py::arg("dim"),
          py::arg("indices"),
          R"doc(Select elements from a tensor along a given dimension using indices.

Args:
    x: Input tensor
    indices: 1D tensor containing the indices to select
    dim: Dimension along which to select
)doc");

    m.def("index_select_",
          &op::index_select_,
          py::arg("y"),
          py::arg("x"),
          py::arg("dim"),
          py::arg("indices"),
          R"doc(Select elements from a tensor along a given dimension using indices.
          
Args:
    y: Output tensor   
    x: Input tensor
    indices: 1D tensor containing the indices to select
    dim: Dimension along which to select
)doc");
}

} // namespace infinicore::ops