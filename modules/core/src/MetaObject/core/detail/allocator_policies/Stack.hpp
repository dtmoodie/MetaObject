#pragma once
#include "../Allocator.hpp"
#include "../MemoryBlock.hpp"
#include "MetaObject/logging/logging.hpp"
#include <ctime>
#include <list>
#include <unordered_map>
namespace mo
{
    template <class XPU>
    class StackPolicy : public Allocator
    {
      public:
        uint8_t* allocate(const uint64_t num_bytes, const uint64_t elem_size);

        void deallocate(uint8_t* ptr, const uint64_t num_bytes);
        void release();

      private:
        void clear();
        struct FreeMemory
        {
            uint8_t* ptr;
            clock_t free_time;
            uint64_t size;
        };
        std::list<FreeMemory> m_deallocate_list;
        size_t m_deallocate_delay; // ms
    };

    using CpuStackPolicy = StackPolicy<CPU>;

    ///////////////////////////////////////////////////////////////////////////
    ///                      Implementation
    ///////////////////////////////////////////////////////////////////////////

    template <class XPU>
    unsigned char* StackPolicy<XPU>::allocate(const uint64_t num_bytes, const uint64_t)
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
        ptr = XPU::allocate(num_bytes);
        return ptr;
    }

    template <class XPU>
    void StackPolicy<XPU>::deallocate(unsigned char* ptr, const uint64_t num_bytes)
    {
        m_deallocate_list.push_back({ptr, clock(), num_bytes});
        clear();
    }

    template <class XPU>
    void StackPolicy<XPU>::release()
    {
        clear();
    }

    template <class XPU>
    void StackPolicy<XPU>::clear()
    {

        auto time = clock();
        for (auto itr = m_deallocate_list.begin(); itr != m_deallocate_list.end();)
        {
            if ((time - itr->free_time) > m_deallocate_delay)
            {
                getDefaultLogger().trace(
                    "[GPU] Deallocating block of size {} which was stale for {} milliseconds at {}",
                    itr->size / (1024 * 1024),
                    (time - itr->free_time) * 1000 / CLOCKS_PER_SEC,
                    static_cast<void*>(itr->ptr));

                XPU::deallocate(itr->ptr);
                itr = m_deallocate_list.erase(itr);
            }
            else
            {
                ++itr;
            }
        }
    }
}
