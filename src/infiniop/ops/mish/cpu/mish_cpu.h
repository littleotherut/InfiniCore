#ifndef __MISH_CPU_H__
#define __MISH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(mish, cpu)

namespace op::mish::cpu {
typedef struct MishOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        if (x > T(20)) {
            return x * std::tanh(x);
        } else {
            return x * std::tanh(std::log(T(1) + std::exp(x)));
        }
    }
} MishOp;
} // namespace op::mish::cpu

#endif // ___CPU_H__
