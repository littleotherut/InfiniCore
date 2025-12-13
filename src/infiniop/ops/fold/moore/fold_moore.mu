#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_handle.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "../cuda/kernel.cuh"
#include "fold_moore.cuh"
#include <algorithm>
#include <cstring>

namespace op::fold::moore {

template <typename accT, typename T>
static infiniStatus_t fold_moore_impl(
    const FoldInfo &info,
    T *y,
    const T *x,
    musaStream_t stream) {

    // 目前仅支持 2D (ndim == 2)
    if (info.ndim() != 2) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    const size_t batch = info.batch();
    const size_t channels = info.channels();
    const size_t height = info.output_dim(0);  // H
    const size_t width = info.output_dim(1);   // W
    const size_t height_col = info.col_dim(0); // 滑窗位置数 (H 方向)
    const size_t width_col = info.col_dim(1);  // 滑窗位置数 (W 方向)
    const size_t kernel_h = info.kernel_dim(0);
    const size_t kernel_w = info.kernel_dim(1);
    const size_t pad_h = info.pad_info(0);
    const size_t pad_w = info.pad_info(1);
    const ptrdiff_t stride_h = info.stride_info(0);
    const ptrdiff_t stride_w = info.stride_info(1);
    const size_t dilation_h = info.dilation_info(0);
    const size_t dilation_w = info.dilation_info(1);

    // 输入 x 的 batch stride: C * kH * kW * L
    const size_t channels_col = channels * kernel_h * kernel_w;
    const size_t L = height_col * width_col;
    const size_t x_batch_stride = channels_col * L;

    // 输出 y 的 batch stride: C * H * W
    const size_t y_batch_stride = channels * height * width;

    col2im_batched<accT, T>(
        stream, x, x_batch_stride,
        batch,
        channels,
        height,
        width,
        height_col,
        width_col,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        y,
        y_batch_stride);

    return INFINI_STATUS_SUCCESS;
}

inline size_t calculateOutputSize(const FoldInfo &info) {
    return info.batch() * info.channels() * info.spatial_sizes();
}

inline bool needsPadding(const FoldInfo &info) {
    return info.padded_shape_size() > 0;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    void *output_sizes,
    void *kernel_sizes,
    void *dilations,
    void *pads,
    void *strides,
    size_t n) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = FoldInfo::create(handle_, y_desc, x_desc,
                                   output_sizes, kernel_sizes,
                                   dilations, pads, strides, n);
    CHECK_RESULT(result);

    const FoldInfo &info = result.take();

    *desc_ptr = new Descriptor(
        dtype, std::move(info), 0,
        nullptr,
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void * /*workspace*/,
    size_t /*workspace_size*/,
    void *y,
    const void *x,
    void *stream_) const {

    musaStream_t stream = reinterpret_cast<musaStream_t>(stream_);

    switch (_dtype) {
    case INFINI_DTYPE_F32:
        return fold_moore_impl<float, float>(
            _info,
            reinterpret_cast<float *>(y),
            reinterpret_cast<const float *>(x),
            stream);

    case INFINI_DTYPE_F16:
        return fold_moore_impl<float, half>(
            _info,
            reinterpret_cast<half *>(y),
            reinterpret_cast<const half *>(x),
            stream);

    case INFINI_DTYPE_BF16:
        return fold_moore_impl<float, cuda_bfloat16>(
            _info,
            reinterpret_cast<cuda_bfloat16 *>(y),
            reinterpret_cast<const cuda_bfloat16 *>(x),
            stream);

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fold::moore