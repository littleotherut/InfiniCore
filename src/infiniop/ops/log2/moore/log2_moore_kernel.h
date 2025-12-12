#ifndef __LOG2_MOORE_H__
#define __LOG2_MOORE_H__

#include <cmath>
#include <type_traits>

namespace op::log2::cuda {
typedef struct Log2Op {
    static constexpr size_t num_inputs = 1;

    __device__ __forceinline__ float log2_f32_func(float x) const {
        return log2f(x);
    }

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            return hlog2(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16(log2_f32_func(xf));
        } else if constexpr (std::is_same_v<T, half2>) {
            return h2log2(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float f0 = __bfloat162float(__low2bfloat16(x));
            float f1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(log2_f32_func(f0), log2_f32_func(f1));
        } else if constexpr (std::is_same_v<T, float>) {
            return log2_f32_func(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::log2(x);
        } else {
            return log2_f32_func(static_cast<float>(x));
        }
    }
} Log2Op;
} // namespace op::log2::moore

#endif // __LOG2_MOORE_H__
