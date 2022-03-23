#ifndef MO_CORE_ALLOCATOR_DEFAULT_HPP
#define MO_CORE_ALLOCATOR_DEFAULT_HPP
#include <MetaObject/core/detail/Allocator.hpp>

namespace mo
{
    template <class XPU>
    struct DefaultAllocator : public XPU::Allocator_t
    {
        uint8_t* allocate(size_t num_bytes, size_t element_size = 1) override
        {
            return XPU::allocate(num_bytes, element_size);
        }

        void deallocate(uint8_t* ptr, size_t num_bytes) override
        {
            XPU::deallocate(ptr, num_bytes);
        }
    };
} // namespace mo

#endif // MO_CORE_ALLOCATOR_DEFAULT_HPP
