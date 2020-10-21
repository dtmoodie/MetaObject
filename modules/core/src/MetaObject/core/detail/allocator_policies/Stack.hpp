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
    class StackPolicy : public XPU::Allocator_t
    {
      public:
        using Allocator_t = typename XPU::Allocator_t;
        uint8_t* allocate(size_t num_bytes, size_t elem_size) override;

        void deallocate(uint8_t* ptr, size_t num_bytes) override;
        void release() override;

      private:
        void clear();
        struct FreeMemory
        {
            uint8_t* ptr;
            clock_t free_time;
            size_t size;
        };
        std::list<FreeMemory> m_deallocate_list;
        size_t m_deallocate_delay; // ms
    };

    using CpuStackPolicy = StackPolicy<CPU>;

    ///////////////////////////////////////////////////////////////////////////
    ///                      Implementation
    ///////////////////////////////////////////////////////////////////////////

    template <class XPU>
    uint8_t* StackPolicy<XPU>::allocate(const size_t num_bytes, const size_t elem_size)
    {
        unsigned char* ptr = nullptr;
        for (auto itr = m_deallocate_list.begin(); itr != m_deallocate_list.end(); ++itr)
        {
            const size_t alignment_offset = alignmentOffset(itr->ptr, elem_size);
            if (itr->size == num_bytes)
            {
                ptr = itr->ptr;
                m_deallocate_list.erase(itr);
                return ptr;
            }
        }
        ptr = XPU::allocate(num_bytes, elem_size);
        return ptr;
    }

    template <class XPU>
    void StackPolicy<XPU>::deallocate(uint8_t* ptr, const size_t num_bytes)
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
                getDefaultLogger().trace("Deallocating block of size {} which was stale for {} milliseconds at {}",
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
} // namespace mo
