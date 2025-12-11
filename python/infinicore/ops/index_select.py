from typing import List

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def index_select(
    input: Tensor,
    indices: Tensor,
    dim: int = 0,
    *,
    out=None) -> Tensor:
    """Returns a new tensor which indexes the input tensor along dimension `dim`"""
    
    if out is None:
        return _infinicore.index_select(input, indices, dim)
    else:
        _infinicore.index_select_(out, input, indices, dim)
        return out