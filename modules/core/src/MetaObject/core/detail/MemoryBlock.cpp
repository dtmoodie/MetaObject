

#include "MetaObject/core/detail/MemoryBlock.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#include <MetaObject/logging/logging.hpp>

namespace mo
{

    uint8_t* CPU::allocate(const size_t size, const size_t)
    {
        uint8_t* ptr = nullptr;
        ptr = static_cast<uint8_t*>(malloc(size));
        MO_ASSERT_FMT(ptr, "Unable to allocate {} bytes", size);
        return ptr;
    }

    void CPU::deallocate(uint8_t* data, const size_t)
    {
        free(data);
    }

    template struct MemoryBlock<CPU>;
} // namespace mo
