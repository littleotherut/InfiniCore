#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Mish {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor y, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor mish(Tensor x);
void mish_(Tensor y, Tensor x);
} // namespace infinicore::op
