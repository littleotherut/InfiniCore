from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def bitwise_left_shift(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.bitwise_left_shift(input._underlying, other._underlying))

    _infinicore.bitwise_left_shift_(out._underlying, input._underlying, other._underlying)

    return out
