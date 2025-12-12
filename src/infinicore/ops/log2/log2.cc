#include "infinicore/ops/log2.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Log2::schema> &Log2::dispatcher() {
    static common::OpDispatcher<Log2::schema> dispatcher_;
    return dispatcher_;
};

void Log2::execute(Tensor y, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x);
}

Tensor log2(Tensor x) {
    auto y = Tensor::empty(x->shape(), x->dtype(), x->device());
    log2_(y, x);
    return y;
}

void log2_(Tensor y, Tensor x) {
    Log2::execute(y, x);
}

} // namespace infinicore::op
