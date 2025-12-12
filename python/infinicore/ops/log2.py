from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def log2(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.log2(input._underlying))

    _infinicore.log2_(out._underlying, input._underlying)

    return out
