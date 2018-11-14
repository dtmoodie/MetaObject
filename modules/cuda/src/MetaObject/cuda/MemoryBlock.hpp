#ifndef MO_CUDA_MEMORY_BLOCK_HPP
#define MO_CUDA_MEMORY_BLOCK_HPP
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/core/detail/MemoryBlock.hpp"
#include <cstdint>
namespace mo
{

struct MO_EXPORTS CUDA
{
    static void* allocate(const uint64_t size);
    static void deallocate(void* data);
};

struct MO_EXPORTS HOST
{
    static void* allocate(const uint64_t size);
    static void deallocate(void* data);
};

using HOSTMemory = Memory<HOST>;
using HOSTMemoryBlock = MemoryBlock<HOST>;
using CUDAMemory = Memory<CUDA>;
using CUDAMemoryBlock = MemoryBlock<CUDA>;

extern template class MemoryBlock<CUDA>;
extern template class MemoryBlock<HOST>;
extern template class Memory<CUDA>;
extern template class Memory<HOST>;
}

#endif //MO_CUDA_MEMORY_BLOCK_HPP
