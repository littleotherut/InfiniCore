#ifndef __INFINIOP_INDEX_SELECT_H__
#define __INFINIOP_INDEX_SELECT_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopIndexSelectDescriptor_t;

__C __export infiniStatus_t infiniopCreateIndexSelectDescriptor(
    infiniopHandle_t handle,
    infiniopIndexSelectDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t indices_desc,
    int dim);

__C __export infiniStatus_t infiniopGetIndexSelectWorkspaceSize(
    infiniopIndexSelectDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopIndexSelect(
    infiniopIndexSelectDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *indices,
    void *stream);

__C __export infiniStatus_t infiniopDestroyIndexSelectDescriptor(infiniopIndexSelectDescriptor_t desc);

#endif 