#pragma once
#include "../Allocator.hpp"
#include <cstdint>
namespace mo
{
    template <class Allocator>
    struct UsagePolicy : public Allocator
    {
        uint8_t* allocate(const size_t num_bytes, const size_t elem_size) override;

        void deallocate(uint8_t* ptr, const size_t num_bytes) override;

        size_t usage() const;

      protected:
        const size_t m_usage = 0;
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    ///                            Implementation
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class Allocator>
    uint8_t* UsagePolicy<Allocator>::allocate(const size_t num_bytes, const size_t elem_size)
    {
        auto ptr = Allocator::allocate(num_bytes, elem_size);
        if (ptr)
        {
            m_usage += num_bytes;
        }
        return ptr;
    }

    template <class Allocator>
    void UsagePolicy<Allocator>::deallocate(uint8_t* ptr, const size_t num_bytes)
    {
        Allocator::deallocate(ptr, num_bytes);
        m_usage -= num_bytes;
    }

    template <class Allocator>
    size_t UsagePolicy<Allocator>::usage() const
    {
        return m_usage;
    }
}
