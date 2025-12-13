#include "../../../devices/moore/moore_common.h"
#include "index_select_moore.h"

#include "../../../devices/moore/moore_kernel_common.h"

#include "../cuda/kernel.cuh"

namespace op::index_select::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t indices_desc,
    int dim) {
    auto result = IndexSelectInfo::create(y_desc, x_desc, indices_desc, dim);
    CHECK_RESULT(result);
    auto info = result.take();

    size_t ndim = info.ndim();
    size_t workspace_size = 0;
    workspace_size += ndim * sizeof(size_t);    // shape
    workspace_size += ndim * sizeof(ptrdiff_t); // x_strides
    workspace_size += ndim * sizeof(ptrdiff_t); // y_strides

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        std::move(info),
        workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tidx>
infiniStatus_t launchKernel(
    const IndexSelectInfo &info,
    void *workspace,
    Tdata *y,
    const Tdata *x,
    const Tidx *indices,
    musaStream_t stream) {

    size_t ndim = info.ndim();
    size_t *d_shape = reinterpret_cast<size_t *>(workspace);
    ptrdiff_t *d_x_strides = reinterpret_cast<ptrdiff_t *>(d_shape + ndim);
    ptrdiff_t *d_y_strides = reinterpret_cast<ptrdiff_t *>(d_x_strides + ndim);

    musaMemcpyAsync(d_shape, info.shape.data(), ndim * sizeof(size_t), musaMemcpyHostToDevice, stream);
    musaMemcpyAsync(d_x_strides, info.x_strides.data(), ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream);
    musaMemcpyAsync(d_y_strides, info.y_strides.data(), ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream);

    size_t outer_size = 1;
    for (int i = 0; i < info.dim; ++i) {
        outer_size *= info.shape[i];
    }
    size_t inner_size = 1;
    for (size_t i = info.dim + 1; i < ndim; ++i) {
        inner_size *= info.shape[i];
    }

    size_t total_elements = outer_size * info.num_indices * inner_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    infiniop::index_select::cuda::index_select_kernel<<<grid_size, block_size, 0, stream>>>(
        y, x, indices,
        info.dim, ndim, info.num_indices,
        outer_size, inner_size,
        d_shape, d_x_strides, d_y_strides);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *indices,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    musaStream_t musa_stream = reinterpret_cast<musaStream_t>(stream);
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(indices);

    if (_info.atype == INFINI_DTYPE_F32) {
        return launchKernel(_info, workspace, (float *)y, (const float *)x, idx_ptr, musa_stream);
    } else if (_info.atype == INFINI_DTYPE_F16) {
        return launchKernel(_info, workspace, (half *)y, (const half *)x, idx_ptr, musa_stream);
    } else if (_info.atype == INFINI_DTYPE_BF16) {
        return launchKernel(_info, workspace, (__mt_bfloat16 *)y, (const __mt_bfloat16 *)x, idx_ptr, musa_stream);
    } else if (_info.atype == INFINI_DTYPE_F64) {
        return launchKernel(_info, workspace, (double *)y, (const double *)x, idx_ptr, musa_stream);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::index_select::moore