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
        StackPolicy(std::chrono::milliseconds deallocate_delay = std::chrono::milliseconds(100));
        ~StackPolicy();
        using Allocator_t = typename XPU::Allocator_t;
        uint8_t* allocate(size_t num_bytes, size_t elem_size) override;

        void deallocate(uint8_t* ptr, size_t num_bytes) override;
        void release() override;

      private:
        void clear();
        struct FreeMemory
        {
            uint8_t* ptr = nullptr;
            std::chrono::high_resolution_clock::time_point free_time;
            size_t size = 0;
        };
        std::list<FreeMemory> m_deallocate_list;
        std::chrono::milliseconds m_deallocate_delay; // ms
    };

    using CpuStackPolicy = StackPolicy<CPU>;

    ///////////////////////////////////////////////////////////////////////////
    ///                      Implementation
    ///////////////////////////////////////////////////////////////////////////

    template <class XPU>
    StackPolicy<XPU>::StackPolicy(std::chrono::milliseconds deallocate_delay):
        m_deallocate_delay(deallocate_delay)
    {

    }

    template <class XPU>
    StackPolicy<XPU>::~StackPolicy()
    {
        for(auto itr = m_deallocate_list.begin(); itr != m_deallocate_list.end(); )
        {
            XPU::deallocate(itr->ptr);
            itr = m_deallocate_list.erase(itr);
        }
    }

    template <class XPU>
    uint8_t* StackPolicy<XPU>::allocate(const size_t num_bytes, const size_t elem_size)
    {
        unsigned char* ptr = nullptr;
        for (auto itr = m_deallocate_list.begin(); itr != m_deallocate_list.end(); ++itr)
        {
            // const size_t alignment_offset = alignmentOffset(itr->ptr, elem_size);
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
        m_deallocate_list.push_back({ptr, std::chrono::high_resolution_clock::now(), num_bytes});
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

        std::chrono::high_resolution_clock::time_point time = std::chrono::high_resolution_clock::now();
        for (auto itr = m_deallocate_list.begin(); itr != m_deallocate_list.end();)
        {
            const auto delta = (time - itr->free_time);
            if (delta > m_deallocate_delay)
            {
                getDefaultLogger().trace("Deallocating block of size {} which was stale for {} milliseconds at {}",
                                         itr->size / (1024 * 1024),
                                         std::chrono::duration_cast<std::chrono::milliseconds>(delta).count(),
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
