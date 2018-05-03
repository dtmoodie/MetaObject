#pragma once
#include "Pool.hpp"
#include <algorithm>
namespace mo
{
    template <class XPU>
    MemoryPool<XPU>::MemoryPool() : m_initial_block_size(20 * 1024 * 1024)
    {
    }

    template <class XPU>
    unsigned char* MemoryPool<XPU>::allocate(size_t num_bytes, size_t elem_size)
    {
        unsigned char* ptr = nullptr;
        for (auto& itr : m_blocks)
        {
            ptr = itr->allocate(num_bytes, elem_size);
            if (ptr)
            {
                return ptr;
            }
        }
        // If we get to this point, then no memory was found, need to allocate new memory
        m_blocks.push_back(
            std::unique_ptr<MemoryBlock<XPU>>(new MemoryBlock<XPU>(std::max(m_initial_block_size / 2, num_bytes))));
        ptr = (*m_blocks.rbegin())->allocate(num_bytes, elem_size);
        if (ptr)
        {
            return ptr;
        }
        // Should only ever reach this if we can't allocate more memory for some reason
        return nullptr;
    }

    template <class XPU>
    void MemoryPool<XPU>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        for (auto& itr : m_blocks)
        {
            if (itr->deAllocate(ptr, num_bytes))
            {
                return;
            }
        }
    }

    template <class XPU>
    void MemoryPool<XPU>::release()
    {
        m_blocks.clear();
    }

    template <class XPU>
    PoolPolicy<XPU>::PoolPolicy(const std::shared_ptr<MemoryPool<XPU>>& pool) : m_pool(pool)
    {
    }

    template <class XPU>
    unsigned char* PoolPolicy<XPU>::allocate(size_t num_bytes, size_t elem_size)
    {
        return m_pool->allocate(num_bytes, elem_size);
    }

    template <class XPU>
    void PoolPolicy<XPU>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        m_pool->deallocate(ptr, num_bytes);
    }

    template <class XPU>
    void PoolPolicy<XPU>::release()
    {
        m_pool->release();
    }
}
