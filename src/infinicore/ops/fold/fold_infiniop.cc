#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/fold.hpp"
#include <infiniop.h>

namespace infinicore::op::fold_impl::infiniop {

thread_local common::OpCache<size_t, infiniopFoldDescriptor_t> caches(
    100, // capacity
    [](infiniopFoldDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyFoldDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x, Param2 output_size, Param2 kernel_size,
               Param2 dilation, Param2 padding, Param2 stride) {
    size_t seed = hash_combine(y, x, output_size.h, output_size.w,
                               kernel_size.h, kernel_size.w,
                               dilation.h, dilation.w,
                               padding.h, padding.w,
                               stride.h, stride.w);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopFoldDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // C API 期望 size_t* / ptrdiff_t*，需要转换
        // Param2 内部是 int64_t，在 64 位系统上与 size_t/ptrdiff_t 兼容
        size_t output_arr[2] = {
            static_cast<size_t>(output_size.h),
            static_cast<size_t>(output_size.w)};
        size_t kernel_arr[2] = {
            static_cast<size_t>(kernel_size.h),
            static_cast<size_t>(kernel_size.w)};
        size_t dilation_arr[2] = {
            static_cast<size_t>(dilation.h),
            static_cast<size_t>(dilation.w)};
        size_t padding_arr[2] = {
            static_cast<size_t>(padding.h),
            static_cast<size_t>(padding.w)};
        ptrdiff_t stride_arr[2] = {
            static_cast<ptrdiff_t>(stride.h),
            static_cast<ptrdiff_t>(stride.w)};

        INFINICORE_CHECK_ERROR(infiniopCreateFoldDescriptor(
            context::getInfiniopHandle(y->device()), &desc,
            y->desc(), x->desc(),
            output_arr, kernel_arr,
            dilation_arr, padding_arr,
            stride_arr, 2));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetFoldWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopFold(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), context::getStream()));
}

static bool registered = []() {
    Fold::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::fold_impl::infiniop
