#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/index_select.hpp"
#include <infiniop.h>

namespace infinicore::op::index_select_impl::infiniop {

thread_local common::OpCache<size_t, infiniopIndexSelectDescriptor_t> caches(
    100, // capacity
    [](infiniopIndexSelectDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyIndexSelectDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x, int dim, Tensor indices) {
    size_t seed = hash_combine(y, x, dim, indices);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopIndexSelectDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateIndexSelectDescriptor(
            context::getInfiniopHandle(y->device()), &desc,
            y->desc(), x->desc(), dim, indices->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetIndexSelectWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopIndexSelect(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), indices->data(), context::getStream()));
}

static bool registered = []() {
    IndexSelect::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::index_select_impl::infiniop
