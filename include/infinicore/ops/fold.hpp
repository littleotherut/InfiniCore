#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <cstdint>
#include <tuple>

namespace infinicore::op {

struct Param2 {
    int64_t h;
    int64_t w;

    Param2(int64_t v) : h(v), w(v) {}
    Param2(std::tuple<int64_t, int64_t> t) : h(std::get<0>(t)), w(std::get<1>(t)) {}
};

class Fold {
public:
    using schema = void (*)(Tensor, Tensor, Param2, Param2, Param2, Param2, Param2);
    
        static void execute(Tensor y, Tensor x, Param2 output_size, Param2 kernel_size,
                            Param2 dilation, Param2 padding, Param2 stride);

    static common::OpDispatcher<schema> &dispatcher();
};

Tensor fold (Tensor x, Param2 output_size, Param2 kernel_size,
            Param2 dilation, Param2 padding, Param2 stride);
void fold_ (Tensor y, Tensor x,  Param2 output_size, Param2 kernel_size,
            Param2 dilation, Param2 padding, Param2 stride);
} // namespace infinicore::op