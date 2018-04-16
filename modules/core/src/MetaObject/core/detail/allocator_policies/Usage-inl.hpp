#include "Usage.hpp"

namespace mo
{

    template<class Allocator>
    unsigned char* UsagePolicy<Allocator>::allocate(size_t num_bytes, size_t elem_size)
    {
        auto ptr = Allocator::allocate(num_bytes, elem_size);
        if (ptr)
        {
            m_usage += num_bytes;
        }
        return ptr;
    }

    template<class Allocator>
    void UsagePolicy<Allocator>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        Allocator::deallocate(ptr, num_bytes);
        m_usage -= num_bytes;
    }


    template<class Allocator>
    size_t UsagePolicy<Allocator>::usage() const
    {
        return m_usage;
    }
}