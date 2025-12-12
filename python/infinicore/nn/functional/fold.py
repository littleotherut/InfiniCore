from typing import Tuple, Union

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fold(
    input: Tensor,
    output_size: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
    *,
    out: Tensor = None,
) -> Tensor:
    r"""Combine an array of sliding local blocks into a large containing tensor.

    """
    # 将 int 转换为 tuple
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)

    if out is None:
        return Tensor(
            _infinicore.fold(
                input._underlying,
                output_size,
                kernel_size,
                dilation,
                padding,
                stride,
            )
        )

    _infinicore.fold_(
        out._underlying,
        input._underlying,
        output_size,
        kernel_size,
        dilation,
        padding,
        stride,
    )
    return out