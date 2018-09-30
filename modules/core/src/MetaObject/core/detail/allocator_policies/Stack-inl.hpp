#pragma once
#include "Stack.hpp"
#include <MetaObject/thread/cuda.hpp>
#include <MetaObject/logging/logging.hpp>

namespace mo
{
    template<class XPU>
    unsigned char* StackPolicy<XPU>::allocate(size_t num_bytes, size_t elem_size)
    {
        unsigned char* ptr = nullptr;
        for (auto itr = m_deallocate_list.begin(); itr != m_deallocate_list.end(); ++itr)
        {
            if (itr->size == num_bytes)
            {
                ptr = itr->ptr;
                m_deallocate_list.erase(itr);
                return ptr;
            }
        }
        Memory<XPU>::allocate(&ptr, num_bytes);
        return ptr;
    }

    template<class XPU>
    void StackPolicy<XPU>::deallocate(unsigned char* ptr, size_t num_bytes)
    {
        m_deallocate_list.push_back({ ptr, clock(), num_bytes });
        clear();
    }

    template<class XPU>
    void StackPolicy<XPU>::release()
    {
        clear();
    }

    template<class XPU>
    void StackPolicy<XPU>::clear()
    {
        if (isCudaThread())
            return;
        auto time = clock();
        for (auto itr = m_deallocate_list.begin(); itr != m_deallocate_list.end();)
        {
            if ((time - itr->free_time) > m_deallocate_delay)
            {
                MO_LOG(trace) << "[GPU] Deallocating block of size " << itr->size / (1024 * 1024)
                    << "MB. Which was stale for " << (time - itr->free_time) * 1000 / CLOCKS_PER_SEC
                    << " ms at " << static_cast<void*>(itr->ptr);
                Memory<XPU>::deallocate(itr->ptr);
                itr = m_deallocate_list.erase(itr);
            }
            else
            {
                ++itr;
            }
        }
    }
}