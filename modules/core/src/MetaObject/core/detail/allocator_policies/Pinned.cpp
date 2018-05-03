#include "Pinned.hpp"
#include "../AllocatorImpl.hpp"
#include "../MemoryBlock.hpp"
#include "MetaObject/core/metaobject_config.hpp"
#if MO_HAVE_CUDA
#include <cuda_runtime.h>
#endif
namespace mo
{
    template <>
    unsigned char* PinnedPolicy<CPU>::allocate(std::size_t num_bytes, std::size_t elem_size)
    {
#if MO_HAVE_CUDA
        void* ptr = nullptr;
        MO_CUDA_ERROR_CHECK(cudaMallocHost(&ptr, num_bytes), " tried to allocate " << num_bytes);
        return static_cast<unsigned char*>(ptr);
#else
        return malloc(num_bytes);
#endif
    }

    template <>
    void PinnedPolicy<CPU>::deallocate(unsigned char* ptr, std::size_t num_bytes)
    {
#if MO_HAVE_CUDA
        MO_CUDA_ERROR_CHECK(cudaFreeHost(ptr), " tried to free " << num_bytes);
#else
        free(ptr);
#endif
    }

    template <>
    void PinnedPolicy<CPU>::release()
    {
    }
}
