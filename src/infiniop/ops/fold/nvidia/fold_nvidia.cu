#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "fold_nvidia.cuh"
#include <algorithm>
#include <cstring>


namespace op::fold::nvidia {

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

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
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

// ============================================================================
// CUDA Kernels
// ============================================================================

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

template <typename accT, typename dt>
void col2im_batched(
    cudaStream_t stream,
    const dt *data_col,
    const size_t col_batch_stride,
    const size_t nbatch,
    const size_t channels,
    const size_t height,
    const size_t width,
    const size_t height_col,
    const size_t width_col,
    const size_t patch_height,
    const size_t patch_width,
    const size_t pad_height,
    const size_t pad_width,
    const size_t stride_height,
    const size_t stride_width,
    const size_t dilation_height,
    const size_t dilation_width,
    dt *data_im,
    const size_t im_batch_stride) {

    const size_t num_kernels = channels * height * width;
    const size_t output_numel = nbatch * num_kernels;
    if (output_numel == 0) {
        return;
    }

    size_t block = 512;
    size_t grid = (output_numel + block - 1) / block;

    col2im_batched_kernel<accT><<<grid, block, 0, stream>>>(
        num_kernels,
        data_col,
        col_batch_stride,
        nbatch,
        height,
        width,
        patch_height,
        patch_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        height_col,
        width_col,
        data_im,
        im_batch_stride);
}

template <typename accT, typename T>
static infiniStatus_t fold_cuda_impl(
    const FoldInfo &info,
    T *y,
    const T *x,
    cudaStream_t stream) {

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

    col2im_batched<accT,T>(
        stream,
        x,
        x_batch_stride,
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

infiniStatus_t Descriptor::calculate(
    void * /*workspace*/,
    size_t /*workspace_size*/,
    void *y,
    const void *x,
    void *stream_) const {

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

    switch (_dtype) {
    case INFINI_DTYPE_F32:
        return fold_cuda_impl<float, float>(
            _info,
            reinterpret_cast<float *>(y),
            reinterpret_cast<const float *>(x),
            stream);

    case INFINI_DTYPE_F16:
        return fold_cuda_impl<float, half>(
            _info,
            reinterpret_cast<half *>(y),
            reinterpret_cast<const half *>(x),
            stream);

    case INFINI_DTYPE_BF16:
        return fold_cuda_impl<float, cuda_bfloat16>(
            _info,
            reinterpret_cast<cuda_bfloat16 *>(y),
            reinterpret_cast<const cuda_bfloat16 *>(x),
            stream);

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
}
}

} // namespace op::fold::nvidia