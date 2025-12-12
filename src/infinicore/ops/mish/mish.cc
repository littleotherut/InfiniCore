#include "infinicore/ops/mish.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Mish::schema> &Mish::dispatcher() {
    static common::OpDispatcher<Mish::schema> dispatcher_;
    return dispatcher_;
};

void Mish::execute(Tensor y, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x);
}

Tensor mish(Tensor x) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    mish_(y, x);
    return y;
}

void mish_(Tensor y, Tensor x) {
    Mish::execute(y, x);
}

} // namespace infinicore::op
