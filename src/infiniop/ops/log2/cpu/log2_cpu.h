#ifndef __LOG2_CPU_H__
#define __LOG2_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(log2, cpu)

namespace op::log2::cpu {
typedef struct Log2Op {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return std::log2(x);
    }
} Log2Op;
} // namespace op::log2::cpu

#endif // ___CPU_H__
