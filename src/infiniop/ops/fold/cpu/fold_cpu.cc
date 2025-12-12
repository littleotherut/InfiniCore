#include "fold_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cstring>

namespace op::fold::cpu {

// ========== 工具函数 ==========

/**
 * 计算输出张量的元素总数: batch * channels * ∏(output_dims)
 */
inline size_t calculateOutputSize(const FoldInfo &info) {
    return info.batch() * info.channels() * info.spatial_sizes();
}

/**
 * 判断是否需要 padding
 */
inline bool needsPadding(const FoldInfo &info) {
    return info.padded_shape_size() > 0;
}

// ========== Descriptor 生命周期 ==========

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

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    // 支持的数据类型
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // 创建 FoldInfo，校验参数
    auto result = FoldInfo::create(handle_, y_desc, x_desc,
                                   output_sizes, kernel_sizes,
                                   dilations, pads, strides, n);
    CHECK_RESULT(result);

    const FoldInfo &info = result.take();

    // CPU fold 不需要额外 workspace
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        dtype, std::move(info), workspace_size,
        nullptr,
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// ========== col2im 核心算法（2D 专用，channels-first 布局）==========

/**
 * col2im: 将 col 格式的 patches 累加回 image
 *
 * @param data_col   输入: (C * kH * kW, height_col * width_col)，实际是展平的 (channels_col, L)
 * @param channels   输出通道数 C
 * @param height     输出高度 H
 * @param width      输出宽度 W
 * @param height_col 滑窗位置数（高度方向）
 * @param width_col  滑窗位置数（宽度方向）
 * @param kernel_h   卷积核高度
 * @param kernel_w   卷积核宽度
 * @param pad_h      padding（高度方向）
 * @param pad_w      padding（宽度方向）
 * @param stride_h   步长（高度方向）
 * @param stride_w   步长（宽度方向）
 * @param dilation_h 膨胀系数（高度方向）
 * @param dilation_w 膨胀系数（宽度方向）
 * @param data_im    输出: (C, H, W)
 */
template <typename T>
static void col2im_2d(
    const T *data_col,
    size_t channels,
    size_t height,
    size_t width,
    size_t height_col,
    size_t width_col,
    size_t kernel_h,
    size_t kernel_w,
    size_t pad_h,
    size_t pad_w,
    ptrdiff_t stride_h,
    ptrdiff_t stride_w,
    size_t dilation_h,
    size_t dilation_w,
    T *data_im) {

    // 先清零输出
    // std::fill_n(data_im, height * width * channels, T(0));

    // channels_col = C * kH * kW
    const size_t channels_col = channels * kernel_h * kernel_w;

    // 遍历 col 的每一行（对应一个 (c, kh, kw) 组合）
    for (size_t c_col = 0; c_col < channels_col; ++c_col) {
        // 从 c_col 反推 (c_im, h_offset, w_offset)
        size_t w_offset = c_col % kernel_w;
        size_t h_offset = (c_col / kernel_w) % kernel_h;
        size_t c_im = c_col / (kernel_h * kernel_w);

        // 遍历所有滑窗位置
        for (size_t h_col = 0; h_col < height_col; ++h_col) {
            // 计算该 patch 对应的 image 坐标（有符号，可能为负）
            ptrdiff_t h_im = static_cast<ptrdiff_t>(h_col) * stride_h
                           - static_cast<ptrdiff_t>(pad_h)
                           + static_cast<ptrdiff_t>(h_offset * dilation_h);

            for (size_t w_col = 0; w_col < width_col; ++w_col) {
                ptrdiff_t w_im = static_cast<ptrdiff_t>(w_col) * stride_w
                               - static_cast<ptrdiff_t>(pad_w)
                               + static_cast<ptrdiff_t>(w_offset * dilation_w);

                // 边界检查：只有在有效范围内才累加
                if (h_im >= 0 && h_im < static_cast<ptrdiff_t>(height) &&
                    w_im >= 0 && w_im < static_cast<ptrdiff_t>(width)) {

                    // data_im 索引: (c_im, h_im, w_im) -> c_im * H * W + h_im * W + w_im
                    size_t im_idx = (c_im * height + static_cast<size_t>(h_im)) * width
                                  + static_cast<size_t>(w_im);

                    // data_col 索引: (c_col, h_col, w_col) -> c_col * L + h_col * width_col + w_col
                    size_t col_idx = (c_col * height_col + h_col) * width_col + w_col;

                    data_im[im_idx] = utils::cast<T,double>(
                        utils::cast<double,T>(data_im[im_idx]) + 
                        utils::cast<double,T>(data_col[col_idx]));
                }
            }
        }
    }
}

// ========== 带 batch 的 fold 入口 ==========

/**
 * fold_cpu_impl: 对整个 batch 执行 fold (col2im)
 *
 * 输入 x: (N, C * kH * kW, L) 其中 L = height_col * width_col
 * 输出 y: (N, C, H, W)
 */
template <typename T>
static infiniStatus_t fold_cpu_impl(
    const FoldInfo &info,
    T *y,
    const T *x) {

    // 目前仅支持 2D (ndim == 2)
    if (info.ndim() != 2) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    const size_t batch = info.batch();
    const size_t channels = info.channels();
    const size_t height = info.output_dim(0);    // H
    const size_t width = info.output_dim(1);     // W
    const size_t height_col = info.col_dim(0);   // 滑窗位置数 (H 方向)
    const size_t width_col = info.col_dim(1);    // 滑窗位置数 (W 方向)
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

    // 对每个 batch 执行 col2im
    for (size_t n = 0; n < batch; ++n) {
        const T *x_n = x + n * x_batch_stride;
        T *y_n = y + n * y_batch_stride;

        col2im_2d<T>(
            x_n,
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
            y_n);
    }

    return INFINI_STATUS_SUCCESS;
}

// ========== Descriptor::calculate 入口 ==========

infiniStatus_t Descriptor::calculate(
    void * /*workspace*/,
    size_t /*workspace_size*/,
    void *y,
    const void *x,
    void * /*stream*/) const {

    switch (_dtype) {
        case INFINI_DTYPE_F32:
            return fold_cpu_impl<float>(
                _info,
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(x));

        case INFINI_DTYPE_F16:
            return fold_cpu_impl<fp16_t>(
                _info,
                reinterpret_cast<fp16_t *>(y),
                reinterpret_cast<const fp16_t *>(x));

        case INFINI_DTYPE_BF16:
            return fold_cpu_impl<bf16_t>(
                _info,
                reinterpret_cast<bf16_t *>(y),
                reinterpret_cast<const bf16_t *>(x));

        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fold::cpu