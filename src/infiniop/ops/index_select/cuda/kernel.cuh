#ifndef __INDEX_SELECT_CUDA_KERNEL_H__
#define __INDEX_SELECT_CUDA_KERNEL_H__


namespace infiniop::index_select::cuda {

template <typename Tdata, typename Tidx>
__global__ void index_select_kernel(
    Tdata * __restrict__ y,
    const Tdata * __restrict__ x,
    const Tidx * __restrict__ indices,
    int dim,
    int ndim,
    size_t num_indices,
    size_t outer_size,
    size_t inner_size,
    const size_t * __restrict__ shape,
    const ptrdiff_t * __restrict__ x_strides,
    const ptrdiff_t * __restrict__ y_strides) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = outer_size * num_indices * inner_size;

    if (idx >= total_elements) {
        return;
    }
    
    size_t inner_idx = idx % inner_size;
    size_t tmp = idx / inner_size;
    size_t idx_pos = tmp % num_indices;
    size_t outer_idx = tmp / num_indices;

    ptrdiff_t x_offset = 0;
    ptrdiff_t y_offset = 0;

    size_t rem_inner = inner_idx;
    for (int i = ndim - 1; i > dim; --i) {
        size_t s = shape[i];
        size_t coord = rem_inner % s;
        rem_inner /= s;
        x_offset += coord * x_strides[i];
        y_offset += coord * y_strides[i];
    }

    Tidx index_val = indices[idx_pos];
    x_offset += index_val * x_strides[dim];
    y_offset += idx_pos * y_strides[dim];

    size_t rem_outer = outer_idx;
    for (int i = dim - 1; i >= 0; --i) {
        size_t s = shape[i];
        size_t coord = rem_outer % s;
        rem_outer /= s;
        x_offset += coord * x_strides[i];
        y_offset += coord * y_strides[i];
    }

    y[y_offset] = x[x_offset];
}

} // namespace infiniop::index_select::cuda

#endif