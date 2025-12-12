#ifndef __MISH_CUDA_H__
#define __MISH_CUDA_H__

#include <cmath>

namespace op::mish::cuda {
typedef struct MishOp {
    static constexpr size_t num_inputs = 1;

    __device__ __forceinline__ float mish_f32_func(float x) const {
        float sp = (x > 20.0f) ? x : log1pf(expf(x));
        return x * tanhf(sp);
    }

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            return __float2half(mish_f32_func(xf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16(mish_f32_func(xf));
        } else if constexpr (std::is_same_v<T, half2>) {
            float2 xf = __half22float2(x);
            return __floats2half2_rn(mish_f32_func(xf.x), mish_f32_func(xf.y));
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float f0 = __bfloat162float(__low2bfloat16(x));
            float f1 = __bfloat162float(__high2bfloat16(x));
            return __floats2bfloat162_rn(mish_f32_func(f0), mish_f32_func(f1));
        } else if constexpr (std::is_same_v<T, float>) {
            return mish_f32_func(x);
        } else if constexpr (std::is_same_v<T, double>) {
            double sp = (x > 20.0) ? x : log1p(exp(x));
            return x * std::tanh(sp);
        } else {
            return mish_f32_func(static_cast<float>(x));
        }
    }
} MishOp;
} // namespace op::mish::cuda

#endif // __MISH_CUDA_H__