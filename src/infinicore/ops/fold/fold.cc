#include "infinicore/ops/fold.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Fold::schema> &Fold::dispatcher() {
    static common::OpDispatcher<Fold::schema> dispatcher_;
    return dispatcher_;
};

void Fold::execute(Tensor y, Tensor x, Param2 output_size, Param2 kernel_size,
                   Param2 dilation, Param2 padding, Param2 stride) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    infinicore::context::setDevice(y->device());
    dispatcher().lookup(y->device().getType())(y, x, output_size, kernel_size,
                                               dilation, padding, stride);
}

Tensor fold(Tensor x, Param2 output_size, Param2 kernel_size,
            Param2 dilation, Param2 padding, Param2 stride) {
    // 输入 x: (N, C * kH * kW, L)
    // 输出 y: (N, C, output_size.h, output_size.w)
    // 其中 C = x.shape[1] / (kH * kW)
    auto x_shape = x->shape();
    assert(x_shape.size() == 3); // "fold input must be 3D (N, C*kH*kW, L)";

    int64_t batch = x_shape[0];
    int64_t kernel_prod = kernel_size.h * kernel_size.w;
    assert(x_shape[1] % kernel_prod == 0); // "input.shape[1] must be divisible by kernel_size product");
    int64_t channels = x_shape[1] / kernel_prod;

    // 构造输出形状: (N, C, H, W)
    std::vector<size_t> y_shape = {
        static_cast<size_t>(batch),
        static_cast<size_t>(channels),
        static_cast<size_t>(output_size.h),
        static_cast<size_t>(output_size.w)};

    auto y = Tensor::empty(y_shape, x->dtype(), x->device());
    fold_(y, x, output_size, kernel_size, dilation, padding, stride);
    return y;
}

void fold_(Tensor y, Tensor x, Param2 output_size, Param2 kernel_size,
           Param2 dilation, Param2 padding, Param2 stride) {
    Fold::execute(y, x, output_size, kernel_size, dilation, padding, stride);
}

} // namespace infinicore::op
