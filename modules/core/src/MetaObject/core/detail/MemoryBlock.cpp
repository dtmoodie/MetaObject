#include "MetaObject/core/detail/MemoryBlock.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include <MetaObject/logging/logging.hpp>
#include <algorithm>
#include <utility>
#include <vector>
#ifdef HAVE_CUDA
#include "AllocatorImpl.hpp"
#include <cuda_runtime.h>
#endif

namespace mo
{

    void* CPU::allocate(size_t size)
    {
        void* ptr = nullptr;
        ptr = malloc(size);
        MO_ASSERT_FMT(ptr, "Unable to allocate {} bytes", size);
        return ptr;
    }

    void CPU::deallocate(unsigned char* data)
    {
        free(data);
    }

    template class Memory<CPU>;
    template class MemoryBlock<CPU>;
}
