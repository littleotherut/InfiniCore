#ifndef __FOLD_INFO_H__
#define __FOLD_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef ENABLE_CUDA_API
#include "../../devices/nvidia/nvidia_handle.cuh"
#endif

namespace op::fold {
class FoldInfo;
} // namespace op::fold

namespace op::fold {

/**
 * FoldInfo: 存储 Fold (col2im) 操作所需的元信息
 *
 * Fold 将滑动局部块 (patches) 组合回一个大张量，对重叠区域求和。
 *
 * 输入张量 x: (N, C * ∏(kernel_size), L)
 *   - N: batch size
 *   - C * ∏(kernel_size): 每个 patch 包含的元素数（C 通道 × 每个 patch 的空间点数）
 *   - L: patch 总数 = ∏(col_dims)，即滑窗位置数
 *
 * 输出张量 y: (N, C, *output_size)
 *   - N: batch size
 *   - C: 通道数
 *   - *output_size: 空间维度 (H, W) 或更高维
 *
 * L 的计算公式（每个空间维度 d）:
 *   col_dim[d] = floor((output_size[d] + 2*padding[d] - dilation[d]*(kernel_size[d]-1) - 1) / stride[d]) + 1
 *   L = ∏ col_dim[d]
 */
class FoldInfo {
private:
    // 元数据缓冲区，连续存储所有维度相关信息
    // 布局: [col_dims(n)] [out_dims(n)] [kernel_dims(n)] [pads(n)] [strides(n)] [dilations(n)] [padded_shape(n+2)?]
    std::vector<size_t> _meta;

    size_t _ndim;              // 空间维度数（通常为 2，即 H/W）
    size_t _batch;             // batch size (N)
    size_t _channels;          // 通道数 (C)
    size_t _spatial_sizes;     // 输出空间元素总数 = ∏(output_dims)
    size_t _padded_shape_size; // padded_shape 数组长度（0 表示无 padding）

    FoldInfo(std::vector<size_t> meta,
             size_t ndim,
             size_t batch,
             size_t channels,
             size_t spatial_sizes,
             size_t padded_shape_size)
        : _meta(std::move(meta)),
          _ndim(ndim),
          _batch(batch),
          _channels(channels),
          _spatial_sizes(spatial_sizes),
          _padded_shape_size(padded_shape_size) {}

public:
    // ========== 基本访问器 ==========
    inline size_t ndim() const { return _ndim; }
    inline size_t batch() const { return _batch; }
    inline size_t channels() const { return _channels; }
    inline size_t spatial_sizes() const { return _spatial_sizes; }
    inline size_t padded_shape_size() const { return _padded_shape_size; }

    // ========== meta 内存访问（用于 GPU memcpy）==========
    inline size_t getMetaMemSize() const {
        return _meta.size() * sizeof(size_t);
    }
    inline const int8_t *getMetaStart() const {
        return reinterpret_cast<const int8_t *>(_meta.data());
    }

    // ========== meta 各段指针访问器 ==========
    // col_dims: 每个空间维度的滑窗位置数，∏(col_dims) = L
    inline const size_t *getColDims() const {
        return _meta.data();
    }
    // output_dims: 输出空间维度大小
    inline const size_t *getOutputDims() const {
        return getColDims() + _ndim;
    }
    // kernel_dims: 卷积核/滑窗大小
    inline const size_t *getKernelDims() const {
        return getOutputDims() + _ndim;
    }
    // pads: 每个维度的 padding
    inline const size_t *getPadsInfo() const {
        return getKernelDims() + _ndim;
    }
    // strides: 滑窗步长
    inline const ptrdiff_t *getStridesInfo() const {
        return reinterpret_cast<const ptrdiff_t *>(getPadsInfo() + _ndim);
    }
    // dilations: 空洞/膨胀系数
    inline const size_t *getDilationsInfo() const {
        return reinterpret_cast<const size_t *>(getStridesInfo() + _ndim);
    }
    // padded_shape: 带 padding 的输出形状（仅当有 padding 时存在）
    inline const size_t *getPaddedShape() const {
        return getDilationsInfo() + _ndim;
    }

    // ========== 单元素便捷访问 ==========
    inline size_t col_dim(size_t i) const {
        return i < _ndim ? getColDims()[i] : 0;
    }
    inline size_t output_dim(size_t i) const {
        return i < _ndim ? getOutputDims()[i] : 0;
    }
    inline size_t kernel_dim(size_t i) const {
        return i < _ndim ? getKernelDims()[i] : 0;
    }
    inline size_t pad_info(size_t i) const {
        return i < _ndim ? getPadsInfo()[i] : 0;
    }
    inline ptrdiff_t stride_info(size_t i) const {
        return i < _ndim ? getStridesInfo()[i] : 1;
    }
    inline size_t dilation_info(size_t i) const {
        return i < _ndim ? getDilationsInfo()[i] : 1;
    }
    inline size_t padded_shape(size_t i) const {
        return i < _padded_shape_size ? getPaddedShape()[i] : 0;
    }

    /**
     * 创建 FoldInfo，校验输入/输出张量形状与参数一致性
     *
     * @param y_desc 输出张量描述符，形状 (N, C, *output_size)
     * @param x_desc 输入张量描述符，形状 (N, C*∏(kernel), L)
     * @param output_sizes 输出空间维度数组，长度 n
     * @param kernel_sizes 卷积核大小数组，长度 n
     * @param dilations 膨胀系数数组，长度 n（可为 nullptr，默认 1）
     * @param pads padding 数组，长度 n（可为 nullptr，默认 0）
     * @param strides 步长数组，长度 n（可为 nullptr，默认 1）
     * @param n 空间维度数
     */
    static utils::Result<FoldInfo> create(
        infiniopHandle_t handle,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        const void *output_sizes,
        const void *kernel_sizes,
        const void *dilations,
        const void *pads,
        const void *strides,
        size_t n);
};

/**
 * 计算单个维度的 col_size（滑窗位置数）
 *
 * 公式: col_size = floor((output_size + 2*padding - effective_kernel) / stride) + 1
 * 其中 effective_kernel = dilation * (kernel_size - 1) + 1
 */
inline utils::Result<size_t> calculateFoldColSize(
    size_t output_size,
    size_t kernel_size,
    size_t padding,
    ptrdiff_t stride,
    size_t dilation) {

    if (stride <= 0) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    if (dilation == 0 || kernel_size == 0) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    // effective_kernel: 考虑 dilation 后卷积核实际覆盖的空间跨度
    size_t effective_kernel = dilation * (kernel_size - 1) + 1;
    size_t padded_output = output_size + 2 * padding;

    if (padded_output < effective_kernel) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    size_t col_size = (padded_output - effective_kernel) / static_cast<size_t>(stride) + 1;
    return utils::Result<size_t>(col_size);
}

inline utils::Result<FoldInfo> FoldInfo::create(
    infiniopHandle_t /*handle*/,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    const void *output_sizes,
    const void *kernel_sizes,
    const void *dilations,
    const void *pads,
    const void *strides,
    size_t n) {

    // ========== dtype 校验 ==========
    auto dtype = y_desc->dtype();
    if (dtype != x_desc->dtype()) {
        return utils::Result<FoldInfo>(INFINI_STATUS_BAD_TENSOR_DTYPE);
    }

    size_t ndim = n;
    if (ndim == 0) {
        return utils::Result<FoldInfo>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    // ========== 张量维度校验 ==========
    // y: (N, C, *output_size) -> ndim + 2 维
    CHECK_OR_RETURN(y_desc->ndim() == ndim + 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    // x: (N, C*∏(kernel), L) -> 固定 3 维
    CHECK_OR_RETURN(x_desc->ndim() == 3, INFINI_STATUS_BAD_TENSOR_SHAPE);

    size_t batch = y_desc->shape()[0];
    size_t channels = y_desc->shape()[1];

    // batch 维度必须匹配
    CHECK_OR_RETURN(x_desc->shape()[0] == batch, INFINI_STATUS_BAD_TENSOR_SHAPE);

    // ========== 参数指针转换 ==========
    const size_t *output_ptr = reinterpret_cast<const size_t *>(output_sizes);
    const size_t *kernel_ptr = reinterpret_cast<const size_t *>(kernel_sizes);
    const size_t *pads_ptr = reinterpret_cast<const size_t *>(pads);
    const ptrdiff_t *strides_ptr = reinterpret_cast<const ptrdiff_t *>(strides);
    const size_t *dilations_ptr = reinterpret_cast<const size_t *>(dilations);

    // kernel_sizes 是必需参数
    if (kernel_ptr == nullptr) {
        return utils::Result<FoldInfo>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    // ========== 检查是否有 padding ==========
    bool has_padding = false;
    if (pads_ptr != nullptr) {
        for (size_t i = 0; i < ndim; ++i) {
            if (pads_ptr[i] > 0) {
                has_padding = true;
                break;
            }
        }
    }
    size_t padded_shape_size = has_padding ? (ndim + 2) : 0;

    // ========== 分配并填充 meta ==========
    // 布局: [col_dims(n)] [out_dims(n)] [kernel_dims(n)] [pads(n)] [strides(n)] [dilations(n)] [padded_shape?]
    size_t meta_size = ndim * 6 + padded_shape_size;
    std::vector<size_t> meta(meta_size);

    size_t *col_dims = meta.data();
    size_t *out_dims = col_dims + ndim;
    size_t *ker_dims = out_dims + ndim;
    size_t *pads_info = ker_dims + ndim;
    ptrdiff_t *strides_info = reinterpret_cast<ptrdiff_t *>(pads_info + ndim);
    size_t *dilations_info = reinterpret_cast<size_t *>(strides_info + ndim);
    size_t *padded_shape = dilations_info + ndim;

    size_t kernel_prod = 1; // ∏(kernel_dims)

    for (size_t i = 0; i < ndim; ++i) {
        // 从参数或 y_desc 获取 output_size
        size_t out_i = output_ptr ? output_ptr[i] : y_desc->shape()[i + 2];
        size_t ker_i = kernel_ptr[i];
        size_t pad_i = pads_ptr ? pads_ptr[i] : 0;
        ptrdiff_t stride_i = strides_ptr ? strides_ptr[i] : 1;
        size_t dil_i = dilations_ptr ? dilations_ptr[i] : 1;

        // output_size 必须与 y_desc 的空间维度匹配
        CHECK_OR_RETURN(y_desc->shape()[i + 2] == out_i, INFINI_STATUS_BAD_TENSOR_SHAPE);

        out_dims[i] = out_i;
        ker_dims[i] = ker_i;
        pads_info[i] = pad_i;
        strides_info[i] = stride_i;
        dilations_info[i] = dil_i;

        // 计算该维度的 col_size（滑窗位置数）
        auto col_res = calculateFoldColSize(out_i, ker_i, pad_i, stride_i, dil_i);
        CHECK_RESULT(col_res);
        col_dims[i] = col_res.take();

        if (ker_i == 0) {
            return utils::Result<FoldInfo>(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }
        kernel_prod *= ker_i;
    }

    if (kernel_prod == 0) {
        return utils::Result<FoldInfo>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    // ========== 校验 x 的第二维 = C * ∏(kernel) ==========
    CHECK_OR_RETURN(channels > 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(x_desc->shape()[1] % kernel_prod == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
    size_t channels_from_x = x_desc->shape()[1] / kernel_prod;
    CHECK_OR_RETURN(channels_from_x == channels, INFINI_STATUS_BAD_TENSOR_SHAPE);

    // ========== 校验 x 的第三维 L = ∏(col_dims) ==========
    size_t expected_L = 1;
    size_t spatial_sizes = 1;
    for (size_t i = 0; i < ndim; ++i) {
        expected_L *= col_dims[i];
        spatial_sizes *= out_dims[i];
    }
    CHECK_OR_RETURN(x_desc->shape()[2] == expected_L, INFINI_STATUS_BAD_TENSOR_SHAPE);

    // ========== 填充 padded_shape（如果有 padding）==========
    if (padded_shape_size > 0) {
        padded_shape[0] = batch;
        padded_shape[1] = channels;
        for (size_t i = 0; i < ndim; ++i) {
            padded_shape[i + 2] = out_dims[i] + 2 * pads_info[i];
        }
    }

    FoldInfo info(std::move(meta), ndim, batch, channels, spatial_sizes, padded_shape_size);
    return utils::Result<FoldInfo>(info);
}

} // namespace op::fold

#endif // __FOLD_INFO_H__
