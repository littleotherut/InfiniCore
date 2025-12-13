#ifndef __INDEX_SELECT_H__
#define __INDEX_SELECT_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::index_select {

class IndexSelectInfo {
    IndexSelectInfo() = default;

public:
    infiniDtype_t atype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<ptrdiff_t> x_strides;
    int dim;
    size_t num_indices;
    size_t ndim() const { return shape.size(); }

    static utils::Result<IndexSelectInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t indices_desc,
        int dim) {

        auto atype = y_desc->dtype();
        if (x_desc->dtype() != atype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (atype != INFINI_DTYPE_F16 && atype != INFINI_DTYPE_BF16 && atype != INFINI_DTYPE_F32 && atype != INFINI_DTYPE_F64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        const size_t y_ndim = y_desc->ndim();
        const size_t x_ndim = x_desc->ndim();

        if (dim < 0) {
            dim += x_ndim;
        }

        if (y_ndim != x_ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        for (size_t i = 0; i < y_ndim; ++i) {
            if (i == static_cast<size_t>(dim)) {
                if (y_desc->dim(i) != indices_desc->dim(0)) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
                continue;
            }
            if (x_desc->dim(i) != y_desc->dim(i)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        return utils::Result<IndexSelectInfo>(IndexSelectInfo{
            atype,
            y_desc->shape(),
            y_desc->strides(),
            x_desc->strides(),
            dim,
            indices_desc->dim(0)});
    }
};
} // namespace op::index_select

#endif // __INDEX_SELECT_H__