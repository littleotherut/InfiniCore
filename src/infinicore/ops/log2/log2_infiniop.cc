#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/log2.hpp"
#include <infiniop.h>

namespace infinicore::op::log2_impl::infiniop {

thread_local common::OpCache<size_t, infiniopLog2Descriptor_t> caches(
    100, // capacity
    [](infiniopLog2Descriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyLog2Descriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x) {
    size_t seed = hash_combine(y, x);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopLog2Descriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateLog2Descriptor(
            context::getInfiniopHandle(y->device()), &desc,
            y->desc(), x->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetLog2WorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopLog2(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), context::getStream()));
}

static bool registered = []() {
    Log2::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::log2_impl::infiniop
