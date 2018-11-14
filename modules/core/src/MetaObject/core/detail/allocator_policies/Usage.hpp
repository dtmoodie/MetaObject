#pragma once
#include <cstdint>
namespace mo
{
    template <class Allocator>
    struct UsagePolicy : public Allocator
    {
        unsigned char* allocate(const uint64_t num_bytes, const uint64_t elem_size);

        void deallocate(unsigned char* ptr, const uint64_t num_bytes);

        uint64_t usage() const;

      protected:
        const uint64_t m_usage = 0;
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    ///                            Implementation
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class Allocator>
    unsigned char* UsagePolicy<Allocator>::allocate(const uint64_t num_bytes, const uint64_t elem_size)
    {
        auto ptr = Allocator::allocate(num_bytes, elem_size);
        if (ptr)
        {
            m_usage += num_bytes;
        }
        return ptr;
    }

    template <class Allocator>
    void UsagePolicy<Allocator>::deallocate(unsigned char* ptr, const uint64_t num_bytes)
    {
        Allocator::deallocate(ptr, num_bytes);
        m_usage -= num_bytes;
    }

    template <class Allocator>
    uint64_t UsagePolicy<Allocator>::usage() const
    {
        return m_usage;
    }
}
