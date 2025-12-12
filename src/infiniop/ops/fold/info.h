#ifndef __FOLD_INFO_H__
#define __FOLD_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

#ifdef ENABLE_CUDA_API
#include "../../devices/nvidia/nvidia_handle.cuh"
#endif

namespace op::fold {
class FoldInfo;
} // namespace op::fold

namespace op::fold {

class FoldInfo {
private:
    std::vector<size_t> _meta;
    size_t _ndim;
    size_t _batch;
    size_t _channels;
    size_t _input_height;
    size_t _input_width;
    size_t _output_height;
    size_t _output_width;
    size_t _kernel_height;
    size_t _kernel_width;
    size_t _dilation_height;
    size_t _dilation_width;
    size_t _padding_height;
    size_t _padding_width;
    size_t _stride_height;
    size_t _stride_width;

    FoldInfo(std::vector<size_t> meta,
             size_t ndim,
             size_t batch,
             size_t channels,
             size_t input_height,
             size_t input_width,
             size_t output_height,
             size_t output_width,
             size_t kernel_height,
             size_t kernel_width,
             size_t dilation_height,
             size_t dilation_width,
             size_t padding_height,
             size_t padding_width,
             size_t stride_height,
             size_t stride_width)
        : _meta(std::move(meta)),
          _ndim(ndim),
          _batch(batch),
          _channels(channels),
          _input_height(input_height),
          _input_width(input_width),
          _output_height(output_height),
          _output_width(output_width),
          _kernel_height(kernel_height),
          _kernel_width(kernel_width),
          _dilation_height(dilation_height),
          _dilation_width(dilation_width),
          _padding_height(padding_height),
          _padding_width(padding_width),
          _stride_height(stride_height),
          _stride_width(stride_width) {}

public:
    inline size_t ndim() const { return _ndim; }
    inline size_t batch() const { return _batch; }
    inline size_t channels() const { return _channels; }
    inline size_t inputHeight() const { return _input_height; }
    inline size_t inputWidth() const { return _input_width; }
    inline size_t outputHeight() const { return _output_height; }
    inline size_t outputWidth() const { return _output_width; }
    inline size_t kernelHeight() const { return _kernel_height; }
    inline size_t kernelWidth() const { return _kernel_width; }
    inline size_t dilationHeight() const { return _dilation_height; }
    inline size_t dilationWidth() const { return _dilation_width; }
    inline size_t paddingHeight() const { return _padding_height; }
    inline size_t paddingWidth() const { return _padding_width; }
    inline size_t strideHeight() const { return _stride_height; }
    inline size_t strideWidth() const { return _stride_width; }
    inline size_t getMetaMemSize() const {
        return _meta.size() * sizeof(size_t);
    }
    inline const int8_t *getMetaStart() const {
        return reinterpret_cast<const int8_t *>(_meta.data());
    }
};

}


#endif // __FOLD_INFO_H__