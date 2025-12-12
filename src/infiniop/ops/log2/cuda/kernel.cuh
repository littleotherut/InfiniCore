#ifndef __LOG2_CUDA_H__
#define __LOG2_CUDA_H__

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
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return hlog2(x);
        } else if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, cuda_bfloat162>) {
            return h2log2(x);
        } else if constexpr (std::is_same_v<T, float>) {
            return log2_f32_func(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::log2(x);
        } else {
            return log2_f32_func(static_cast<float>(x));
        }
    }
} Log2Op;
} // namespace op::log2::cuda

#endif // __LOG2_CUDA_H__
