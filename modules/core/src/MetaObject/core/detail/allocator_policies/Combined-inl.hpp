#pragma once

#include "Combined.hpp"

namespace mo
{

    template<class SmallAllocator, class LargeAllocator>
    CombinedPolicy<SmallAllocator, LargeAllocator>::CombinedPolicy(size_t threshold) :
        m_threshold(threshold)
    {

    }

    template<class SmallAllocator, class LargeAllocator>
    unsigned char* CombinedPolicy<SmallAllocator, LargeAllocator>::allocate(size_t num_bytes, size_t elem_size)
    {
        if (num_bytes > m_threshold)
        {
            return SmallAllocator::allocate(num_bytes, elem_size);
        }
        else
        {
            return LargeAllocator::allocate(num_bytes, elem_size);
        }
    }

    template<class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        if (num_bytes > m_threshold)
        {
            return SmallAllocator::deallocate(ptr, num_bytes);
        }
        else
        {
            LargeAllocator::deallocate(ptr, num_bytes);
        }
    }
    template<class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::release()
    {
        SmallAllocator::release();
        LargeAllocator::release();
    }


    template<class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::setThreshold(size_t thresh)
    {
        m_threshold = thresh;
    }

    template<class SmallAllocator, class LargeAllocator>
    size_t CombinedPolicy<SmallAllocator, LargeAllocator>::getThreshold() const
    {
        return m_threshold;
    }
}