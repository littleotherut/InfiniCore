#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class IndexSelect {
public:
    using schema = void (*)(Tensor, Tensor, int, Tensor);
    static void execute(Tensor y, Tensor x, int dim, Tensor indices);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor index_select(Tensor x, int dim, Tensor indices);
void index_select_(Tensor y, Tensor x, int dim, Tensor indices);
} // namespace infinicore::op