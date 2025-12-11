#ifndef __BITWISE_LEFT_SHIFT_CUDA_H__
#define __BITWISE_LEFT_SHIFT_CUDA_H__

namespace op::bitwise_left_shift::cuda {
typedef struct BitwiseLeftShiftOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a << b;
    }
} BitwiseLeftShiftOp;
} // namespace op::bitwise_left_shift::cuda

#endif // __BITWISE_LEFT_SHIFT_CUDA_H__
