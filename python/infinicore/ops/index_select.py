from typing import List

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def index_select(input: Tensor, dim: int, indices: Tensor, *, out=None) -> Tensor:
    """Returns a new tensor which indexes the input tensor along dimension `dim`"""
    if out is None:
        return Tensor(_infinicore.index_select(input._underlying, dim, indices._underlying))
    _infinicore.index_select_(out._underlying, input._underlying, dim, indices._underlying)
    return out