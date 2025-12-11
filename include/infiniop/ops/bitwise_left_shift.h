#ifndef __INFINIOP_BITWISE_LEFT_SHIFT_API_H__
#define __INFINIOP_BITWISE_LEFT_SHIFT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBitwiseLeftShiftDescriptor_t;

__C __export infiniStatus_t infiniopCreateBitwiseLeftShiftDescriptor(infiniopHandle_t handle,
                                                        infiniopBitwiseLeftShiftDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c,
                                                        infiniopTensorDescriptor_t a,
                                                        infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetBitwiseLeftShiftWorkspaceSize(infiniopBitwiseLeftShiftDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopBitwiseLeftShift(infiniopBitwiseLeftShiftDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *c,
                                        const void *a,
                                        const void *b,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyBitwiseLeftShiftDescriptor(infiniopBitwiseLeftShiftDescriptor_t desc);

#endif
