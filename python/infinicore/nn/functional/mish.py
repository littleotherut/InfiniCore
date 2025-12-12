from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mish(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.mish(input._underlying))

    _infinicore.mish_(out._underlying, input._underlying)

    return out
