#ifndef __BITWISE_LEFT_SHIFT_CUDA_H__
#define __BITWISE_LEFT_SHIFT_CUDA_H__

namespace op::bitwise_left_shift::iluvatar {
typedef struct BitwiseLeftShiftOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
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
} // namespace op::bitwise_left_shift::cuda

#endif // __BITWISE_LEFT_SHIFT_CUDA_H__
