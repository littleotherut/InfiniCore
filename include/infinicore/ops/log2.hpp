#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Log2 {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor y, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor log2(Tensor x);
void log2_(Tensor y, Tensor x);
} // namespace infinicore::op
