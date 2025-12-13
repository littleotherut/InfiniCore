#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/index_select.h"

#ifdef ENABLE_CPU_API
#include "cpu/index_select_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/index_select_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/index_select_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/index_select_moore.h"
#endif

__C infiniStatus_t infiniopCreateIndexSelectDescriptor(
    infiniopHandle_t handle,
    infiniopIndexSelectDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int dim,
    infiniopTensorDescriptor_t indices_desc) {    

#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        return op::index_select::NAMESPACE::Descriptor::create(                     \
            handle,                                                             \
            reinterpret_cast<op::index_select::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                             \
            x_desc,                                                             \
            indices_desc,                                                        \
            dim);

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif
    }
#undef CREATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetIndexSelectWorkspaceSize(
    infiniopIndexSelectDescriptor_t desc, 
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                \
    case CASE:                                                                                              \
        *size = reinterpret_cast<op::index_select::NAMESPACE::Descriptor *>(desc)->workspaceSize();  \
        return INFINI_STATUS_SUCCESS;
    
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopIndexSelect(
    infiniopIndexSelectDescriptor_t desc, 
    void *workspace, 
    size_t workspace_size,
    void *y, 
    const void *x, 
    const void *indices,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                \
        return reinterpret_cast<op::index_select::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, y, x, indices, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif
    }
#undef CALCULATE

        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyIndexSelectDescriptor(
    infiniopIndexSelectDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                              \
    case CASE:                                                                \
        delete reinterpret_cast<op::index_select::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, moore);
#endif
    }
#undef DESTROY

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}