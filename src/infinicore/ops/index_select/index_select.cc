#include "infinicore/ops/index_select.hpp"

#include "../../utils.hpp"

namespace infinicore::op{

common::OpDispatcher<IndexSelect::schema> &IndexSelect::dispatcher() {
    static common::OpDispatcher<IndexSelect::schema> dispatcher_;
    return dispatcher_;
}

void IndexSelect::execute(Tensor y, Tensor x, Tensor indices, int dim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, indices);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x, indices, dim);
}

Tensor index_select(Tensor x, Tensor indices, int dim) {
    infinicore::Shape y_shape = x->shape();
    y_shape[dim] = indices->shape()[0];
    auto y = Tensor::empty(y_shape, x->dtype(), x->device());
    index_select_(y, x, indices, dim);
    return y;
}

void index_select_(Tensor y, Tensor x, Tensor indices, int dim) {
    IndexSelect::execute(y, x, indices, dim);
}

} // namespace infinicore::op
