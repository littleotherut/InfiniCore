#ifndef INDEX_SELECT_H
#define INDEX_SELECT_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::index_select::NAMESPACE {               \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        IndexSelectInfo _info;                             \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            IndexSelectInfo info,                          \
            size_t workspace_size,                               \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t y_desc,                   \
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t indices_desc,             \
            int dim);                                            \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *y,                                             \
            const void *x,                                       \
            const void *indices,                                 \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // INDEX_SELECT_H