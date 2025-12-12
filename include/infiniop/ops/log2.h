#ifndef __INFINIOP_LOG2_API_H__
#define __INFINIOP_LOG2_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLog2Descriptor_t;

__C __export infiniStatus_t infiniopCreateLog2Descriptor(infiniopHandle_t handle,
                                                         infiniopLog2Descriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t y,
                                                         infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetLog2WorkspaceSize(infiniopLog2Descriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLog2(infiniopLog2Descriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *y,
                                         const void *x,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroyLog2Descriptor(infiniopLog2Descriptor_t desc);

#endif
