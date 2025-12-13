#ifndef __FOLD_CUDA_H__
#define __FOLD_CUDA_H__

template <typename accT, typename dt>
__forceinline__ __device__ void col2im_device(
    const size_t index,
    const dt *data_col,
    const size_t height,
    const size_t width,
    const size_t kernel_h,
    const size_t kernel_w,
    const size_t pad_height,
    const size_t pad_width,
    const size_t stride_height,
    const size_t stride_width,
    const size_t dilation_height,
    const size_t dilation_width,
    const size_t height_col,
    const size_t width_col,
    dt *data_im) {

    accT val = static_cast<accT>(0);
    const size_t w_im = index % width + pad_width;
    const size_t h_im = (index / width) % height + pad_height;
    const size_t c_im = index / (width * height);

    size_t kernel_extent_w = (kernel_w - 1) * dilation_width + 1;
    size_t kernel_extent_h = (kernel_h - 1) * dilation_height + 1;

    // compute the start and end of the output
    const size_t w_col_start = (w_im < kernel_extent_w)
                                 ? 0
                                 : (w_im - kernel_extent_w) / stride_width + 1;
    const size_t w_col_end = ::min(w_im / stride_width + 1, width_col);
    const size_t h_col_start = (h_im < kernel_extent_h)
                                 ? 0
                                 : (h_im - kernel_extent_h) / stride_height + 1;
    const size_t h_col_end = ::min(h_im / stride_height + 1, height_col);

    for (size_t h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        for (size_t w_col = w_col_start; w_col < w_col_end; w_col += 1) {
            size_t h_k = (h_im - h_col * stride_height);
            size_t w_k = (w_im - w_col * stride_width);
            if (h_k % dilation_height == 0 && w_k % dilation_width == 0) {
                h_k /= dilation_height;
                w_k /= dilation_width;
                size_t data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) * width_col + w_col;

                if constexpr (std::is_same_v<dt, half>) {
                    val += __half2float(data_col[data_col_index]);
                } else if constexpr (std::is_same_v<dt, cuda_bfloat16>) {
                    val += __bfloat162float(data_col[data_col_index]);
                } else {
                    val += data_col[data_col_index];
                }
            }
        }
    }

    if constexpr (std::is_same_v<dt, half>) {
        data_im[index] = __float2half(val);
    } else if constexpr (std::is_same_v<dt, cuda_bfloat16>) {
        data_im[index] = __float2bfloat16(val);
    } else {
        data_im[index] = val;
    }
}

template <typename accT, typename dt>
__launch_bounds__(512)
    __global__ void col2im_batched_kernel(
        const size_t n,
        const dt *data_col,
        const size_t col_batch_stride,
        const size_t nbatch,
        const size_t height,
        const size_t width,
        const size_t kernel_h,
        const size_t kernel_w,
        const size_t pad_height,
        const size_t pad_width,
        const size_t stride_height,
        const size_t stride_width,
        const size_t dilation_height,
        const size_t dilation_width,
        const size_t height_col,
        const size_t width_col,
        dt *data_im,
        const size_t im_batch_stride) {

    const size_t im_numel = n * nbatch;

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < im_numel; index += blockDim.x * gridDim.x) {
        const auto ibatch = index / n;
        const auto slice_index = index % n;

        col2im_device<accT>(
            slice_index,
            data_col + ibatch * col_batch_stride,
            height,
            width,
            kernel_h,
            kernel_w,
            pad_height,
            pad_width,
            stride_height,
            stride_width,
            dilation_height,
            dilation_width,
            height_col,
            width_col,
            data_im + ibatch * im_batch_stride);
    }
}

#endif // __FOLD_CUDA_H__