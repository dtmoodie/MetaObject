#pragma once
#include <cstddef>
#include <cstdint>
namespace mo
{
    // This is just an example, do not use
    template <class XPU>
    struct AllocatorConcept
    {
        uint8_t* allocate(size_t num_bytes, size_t elem_size);
        void deallocate(uint8_t* ptr, size_t num_bytes);
        void release();
    };
}
