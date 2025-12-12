#ifndef __INFINIOP_FOLD_H__
#define __INFINIOP_FOLD_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFoldDescriptor_t;

__C __export infiniStatus_t infiniopCreateFoldDescriptor(
    infiniopHandle_t handle,
    infiniopFoldDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    void *output,
    void *kernel,
    void *dilation,
    void *padding,
    void *stride,
    size_t n);


__C __export infiniStatus_t infiniopGetFoldWorkspaceSize(
    infiniopFoldDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopFold(
    infiniopFoldDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

__C __export infiniStatus_t infiniopDestroyFoldDescriptor(infiniopFoldDescriptor_t desc);

#endif