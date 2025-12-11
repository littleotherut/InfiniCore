#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class IndexSelect {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, int);
    static void execute(Tensor y, Tensor x, Tensor indices, int dim);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor index_select(Tensor x, Tensor indices, int dim);
void index_select_(Tensor y, Tensor x, Tensor indices, int dim);
} // namespace infinicore::op