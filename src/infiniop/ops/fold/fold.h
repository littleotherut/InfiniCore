#ifndef __FOLD_H__
#define __FOLD_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::fold::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        FoldInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            FoldInfo info,                                       \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t y,                        \
            infiniopTensorDescriptor_t x,                        \
            void *output_sizes,                                  \
            void *kernel_sizes,                                  \
            void *dilations,                                     \
            void *pads,                                          \
            void *strides,                                       \
            size_t n);                                           \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *y,                                             \
            const void *x,                                       \
            void *stream) const;                                 \
    };                                                           \
    }
#endif // __FOLD_H__
