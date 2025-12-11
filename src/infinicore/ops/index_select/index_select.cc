#include "infinicore/ops/index_select.hpp"

#include "../../utils.hpp"

namespace infinicore::op{

common::OpDispatcher<IndexSelect::schema> &IndexSelect::dispatcher() {
    static common::OpDispatcher<IndexSelect::schema> dispatcher_;
    return dispatcher_;
}

void IndexSelect::execute(Tensor y, Tensor x, int dim, Tensor indices) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, indices);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x, dim, indices);
}

Tensor index_select(Tensor x, int dim, Tensor indices) {
    infinicore::Shape y_shape = x->shape();
    if(dim < 0){
        dim += x->ndim();
    }
    y_shape[dim] = indices->shape()[0];
    auto y = Tensor::empty(y_shape, x->dtype(), x->device());
    index_select_(y, x, dim, indices);
    return y;
}

void index_select_(Tensor y, Tensor x, int dim, Tensor indices) {
    IndexSelect::execute(y, x, dim, indices);
}

} // namespace infinicore::op
