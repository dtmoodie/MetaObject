#pragma once
#include <cstdint>
namespace mo
{
    // This is just an example, do not use
    template <class XPU>
    struct AllocatorConcept
    {
        uint8_t* allocate(const size_t num_bytes, const size_t elem_size);
        void deallocate(uint8_t* ptr, const size_t num_bytes);
        void release();
    };
}
