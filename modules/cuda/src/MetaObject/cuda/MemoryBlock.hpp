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
            using Allocator_t = mo::DeviceAllocator;
            static uint8_t* allocate(uint64_t size, int32_t elem_size = 1);
            static void deallocate(void* data, uint64_t size = 0);
        };

        struct MO_EXPORTS HOST
        {
            using Allocator_t = mo::Allocator;
            static uint8_t* allocate(uint64_t size, int32_t elem_size = 1);
            static void deallocate(void* data, uint64_t size = 0);
        };

        using HOSTMemoryBlock = MemoryBlock<HOST>;
        using CUDAMemoryBlock = MemoryBlock<CUDA>;
    } // namespace cuda

    extern template struct MemoryBlock<cuda::CUDA>;
    extern template struct MemoryBlock<cuda::HOST>;
} // namespace mo

#endif // MO_CUDA_MEMORY_BLOCK_HPP
