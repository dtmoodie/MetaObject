#pragma once
#include "Lock.hpp"

namespace mo
{
    template<class Allocator>
    unsigned char* LockPolicy<Allocator>::allocate(size_t num_bytes, size_t elem_size)
    {
        boost::mutex::scoped_lock lock(m_mtx);
        return Allocator::allocate(num_bytes, elem_size);
    }

    template<class Allocator>
    void LockPolicy<Allocator>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        boost::mutex::scoped_lock lock(m_mtx);
        Allocator::deallocate(ptr, num_bytes);
    }
}