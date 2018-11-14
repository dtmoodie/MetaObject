#pragma once
#include <cstdint>
namespace mo
{
    // This is just an example, do not use
    template <class XPU>
    struct AllocatorConcept
    {
        uint8_t* allocate(const uint64_t num_bytes, const uint64_t elem_size);
        void deallocate(unsigned char* ptr, const uint64_t num_bytes);
        void release();
    };
}
