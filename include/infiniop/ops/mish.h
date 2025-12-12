#ifndef __INFINIOP_MISH_API_H__
#define __INFINIOP_MISH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMishDescriptor_t;

__C __export infiniStatus_t infiniopCreateMishDescriptor(infiniopHandle_t handle,
                                                          infiniopMishDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y,
                                                          infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetMishWorkspaceSize(infiniopMishDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMish(infiniopMishDescriptor_t desc,
                                            void *workspace,
                                            size_t workspace_size,
                                            void *y,
                                            const void *x,
                                            void *stream);

__C __export infiniStatus_t infiniopDestroyMishDescriptor(infiniopMishDescriptor_t desc);

#endif 