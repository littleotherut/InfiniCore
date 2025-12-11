#ifndef __BITWISE_LEFT_SHIFT_CPU_H__
#define __BITWISE_LEFT_SHIFT_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <type_traits>

ELEMENTWISE_DESCRIPTOR(bitwise_left_shift, cpu)

namespace op::bitwise_left_shift::cpu {
typedef struct BitwiseLeftShiftOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (std::is_signed_v<T>) {
            if (b < 0) {
                return 0;
            }
        }
        if (b >= static_cast<T>(sizeof(T) * 8)) {
            return 0;
        }
        return a << b;
    }
} BitwiseLeftShiftOp;
} // namespace op::bitwise_left_shift::cpu

#endif // __BITWISE_LEFT_SHIFT_CPU_H__
