#include "index_select_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include <algorithm>
#include <cmath>

namespace op::index_select::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t indices_desc,
    int dim) {
    auto result = IndexSelectInfo::create(y_desc, x_desc, indices_desc, dim);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tidx>
infiniStatus_t indexSelect(const IndexSelectInfo *info, Tdata *y, const Tdata *x, const Tidx *indices) {

    const int dim = info->dim;
    const size_t ndim = info->ndim();
    const size_t num_indices = info->num_indices;

    // 计算 outer_size: dim 之前所有维度的乘积
    size_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= info->shape[i];
    }

    // 计算 inner_size: dim 之后所有维度的乘积
    size_t inner_size = 1;
    for (size_t i = dim + 1; i < ndim; ++i) {
        inner_size *= info->shape[i];
    }

    // dim 维度上的 stride
    const ptrdiff_t x_dim_stride = info->x_strides[dim];
    const ptrdiff_t y_dim_stride = info->y_strides[dim];

    // 计算 outer 维度的 stride（从 dim-1 到 0 的累积）
    // x_outer_strides[i] 表示 outer 索引在第 i 维的 stride
    std::vector<ptrdiff_t> x_outer_strides(dim);
    std::vector<ptrdiff_t> y_outer_strides(dim);
    std::vector<size_t> outer_shape(dim);
    for (int i = 0; i < dim; ++i) {
        x_outer_strides[i] = info->x_strides[i];
        y_outer_strides[i] = info->y_strides[i];
        outer_shape[i] = info->shape[i];
    }

    // 计算 inner 维度的 stride
    std::vector<ptrdiff_t> x_inner_strides(ndim - dim - 1);
    std::vector<ptrdiff_t> y_inner_strides(ndim - dim - 1);
    std::vector<size_t> inner_shape(ndim - dim - 1);
    for (size_t i = dim + 1; i < ndim; ++i) {
        x_inner_strides[i - dim - 1] = info->x_strides[i];
        y_inner_strides[i - dim - 1] = info->y_strides[i];
        inner_shape[i - dim - 1] = info->shape[i];
    }

    const ptrdiff_t total_tasks = static_cast<ptrdiff_t>(outer_size * num_indices);

#pragma omp parallel for
    for (ptrdiff_t task_idx = 0; task_idx < total_tasks; ++task_idx) {
        const size_t outer_idx = task_idx / num_indices;
        const size_t idx_pos = task_idx % num_indices;

        // 计算 outer 部分的 offset（支持非连续 stride）
        ptrdiff_t x_base = 0;
        ptrdiff_t y_base = 0;
        size_t tmp = outer_idx;
        for (int i = dim - 1; i >= 0; --i) {
            size_t coord = tmp % outer_shape[i];
            tmp /= outer_shape[i];
            x_base += coord * x_outer_strides[i];
            y_base += coord * y_outer_strides[i];
        }

        // 获取索引值
        const Tidx idx = indices[idx_pos];

        // dim 维度上的偏移
        const ptrdiff_t x_dim_offset = x_base + idx * x_dim_stride;
        const ptrdiff_t y_dim_offset = y_base + idx_pos * y_dim_stride;

        // 拷贝 inner 部分的所有元素
        for (size_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
            // 计算 inner 部分的 offset（支持非连续 stride）
            ptrdiff_t x_inner_offset = 0;
            ptrdiff_t y_inner_offset = 0;
            size_t tmp_inner = inner_idx;
            for (int i = static_cast<int>(inner_shape.size()) - 1; i >= 0; --i) {
                size_t coord = tmp_inner % inner_shape[i];
                tmp_inner /= inner_shape[i];
                x_inner_offset += coord * x_inner_strides[i];
                y_inner_offset += coord * y_inner_strides[i];
            }

            y[y_dim_offset + y_inner_offset] = x[x_dim_offset + x_inner_offset];
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *indices,
    void *stream) const {

    // indices 类型固定为 int64_t (INFINI_DTYPE_I64)
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(indices);

    if (_info.atype == INFINI_DTYPE_F32) {
        return indexSelect<float, int64_t>(&_info, (float *)y, (const float *)x, idx_ptr);
    } else if (_info.atype == INFINI_DTYPE_F16) {
        return indexSelect<fp16_t, int64_t>(&_info, (fp16_t *)y, (const fp16_t *)x, idx_ptr);
    } else if (_info.atype == INFINI_DTYPE_BF16) {
        return indexSelect<bf16_t, int64_t>(&_info, (bf16_t *)y, (const bf16_t *)x, idx_ptr);
    } else if (_info.atype == INFINI_DTYPE_F64) {
        return indexSelect<double, int64_t>(&_info, (double *)y, (const double *)x, idx_ptr);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::index_select::cpu