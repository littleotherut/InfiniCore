#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/ops/fold.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_fold(py::module &m) {
    m.def(
        "fold",
        [](Tensor x,
           std::tuple<int64_t, int64_t> output_size,
           std::tuple<int64_t, int64_t> kernel_size,
           std::tuple<int64_t, int64_t> dilation,
           std::tuple<int64_t, int64_t> padding,
           std::tuple<int64_t, int64_t> stride) {
            return op::fold(x,
                            op::Param2(output_size),
                            op::Param2(kernel_size),
                            op::Param2(dilation),
                            op::Param2(padding),
                            op::Param2(stride));
        },
        py::arg("x"),
        py::arg("output_size"),
        py::arg("kernel_size"),
        py::arg("dilation") = std::make_tuple(1, 1),
        py::arg("padding") = std::make_tuple(0, 0),
        py::arg("stride") = std::make_tuple(1, 1),
        R"doc(Combine an array of sliding local blocks into a large containing tensor.

This is the inverse operation of unfold. Also known as col2im.

Args:
    x: Input tensor of shape (N, C * kH * kW, L)
    output_size: Spatial shape of the output (H, W)
    kernel_size: Size of the sliding blocks (kH, kW)
    dilation: Spacing between kernel elements, default (1, 1)
    padding: Implicit zero padding on both sides, default (0, 0)
    stride: Stride of the sliding blocks, default (1, 1)

Returns:
    Folded tensor of shape (N, C, H, W)
)doc");

    m.def(
        "fold_",
        [](Tensor y,
           Tensor x,
           std::tuple<int64_t, int64_t> output_size,
           std::tuple<int64_t, int64_t> kernel_size,
           std::tuple<int64_t, int64_t> dilation,
           std::tuple<int64_t, int64_t> padding,
           std::tuple<int64_t, int64_t> stride) {
            op::fold_(y, x,
                      op::Param2(output_size),
                      op::Param2(kernel_size),
                      op::Param2(dilation),
                      op::Param2(padding),
                      op::Param2(stride));
        },
        py::arg("y"),
        py::arg("x"),
        py::arg("output_size"),
        py::arg("kernel_size"),
        py::arg("dilation") = std::make_tuple(1, 1),
        py::arg("padding") = std::make_tuple(0, 0),
        py::arg("stride") = std::make_tuple(1, 1),
        R"doc(In-place fold (col2im) operation.

Args:
    y: Output tensor of shape (N, C, H, W)
    x: Input tensor of shape (N, C * kH * kW, L)
    output_size: Spatial shape of the output (H, W)
    kernel_size: Size of the sliding blocks (kH, kW)
    dilation: Spacing between kernel elements, default (1, 1)
    padding: Implicit zero padding on both sides, default (0, 0)
    stride: Stride of the sliding blocks, default (1, 1)
)doc");
}

} // namespace infinicore::ops
