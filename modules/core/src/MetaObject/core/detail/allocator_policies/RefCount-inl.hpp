#pragma once
#include "RefCount.hpp"
#include <MetaObject/logging/logging.hpp>

namespace mo
{
    template<class Allocator>
    RefCountPolicy<Allocator>::~RefCountPolicy()
    {
        if (m_ref_count != 0)
        {
            MO_LOG(warning) << "Trying to delete allocator with " << m_ref_count << " mats still referencing it";
        }
    }

    template<class Allocator>
    unsigned char* RefCountPolicy<Allocator>::allocate(size_t num_bytes, size_t elem_size)
    {
        auto ptr = Allocator::allocate(num_bytes, elem_size);
        if (ptr)
        {
            ++m_ref_count;
        }
        return ptr;
    }

    template<class Allocator>
    void RefCountPolicy<Allocator>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        Allocator::deallocate(ptr, num_bytes);
        --m_ref_count;
    }

    template<class Allocator>
    size_t RefCountPolicy<Allocator>::refCount() const
    {
        return m_ref_count;
    }

}