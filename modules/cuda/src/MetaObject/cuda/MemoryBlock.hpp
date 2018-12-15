#ifndef MO_CUDA_MEMORY_BLOCK_HPP
#define MO_CUDA_MEMORY_BLOCK_HPP
#include "MetaObject/core/detail/MemoryBlock.hpp"
#include "MetaObject/detail/Export.hpp"
#include <cstdint>
namespace mo
{
    namespace cuda
    {
        struct MO_EXPORTS CUDA
        {
            static uint8_t* allocate(const uint64_t size, const int32_t elem_size = 1);
            static void deallocate(void* data, const uint64_t size = 0);
        };

        struct MO_EXPORTS HOST
        {
            static uint8_t* allocate(const uint64_t size, const int32_t elem_size = 1);
            static void deallocate(void* data, const uint64_t size = 0);
        };

        using HOSTMemoryBlock = MemoryBlock<HOST>;
        using CUDAMemoryBlock = MemoryBlock<CUDA>;
    }

    extern template class MemoryBlock<cuda::CUDA>;
    extern template class MemoryBlock<cuda::HOST>;
}

#endif // MO_CUDA_MEMORY_BLOCK_HPP
