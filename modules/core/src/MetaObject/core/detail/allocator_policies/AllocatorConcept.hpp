#pragma once

namespace mo
{
    // This is just an example, do not use
    template<class XPU>
    struct AllocatorConcept
    {
        unsigned char* allocate(size_t num_bytes, size_t elem_size);
        void deallocate(unsigned char* ptr, size_t num_bytes);
        void release();
    };
}