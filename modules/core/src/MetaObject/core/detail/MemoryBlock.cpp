#include "MetaObject/core/detail/MemoryBlock.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include <MetaObject/logging/logging.hpp>
#include <algorithm>
#include <utility>
#include <vector>

namespace mo
{

    uint8_t* CPU::allocate(const uint64_t size)
    {
        uint8_t* ptr = nullptr;
        ptr = static_cast<uint8_t*>(malloc(size));
        MO_ASSERT_FMT(ptr, "Unable to allocate {} bytes", size);
        return ptr;
    }

    void CPU::deallocate(unsigned char* data)
    {
        free(data);
    }

    template class MemoryBlock<CPU>;
}
