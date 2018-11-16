#pragma once
#include "../Allocator.hpp"
#include <cstdint>
#include <memory>

namespace mo
{
    template <class SmallAllocator, class LargeAllocator>
    class CombinedPolicy : public Allocator
    {
      public:
        using Ptr = std::shared_ptr<CombinedPolicy>;
        static Ptr create();

        CombinedPolicy(const uint64_t threshold = 1 * 1024 * 512);

        uint8_t* allocate(const uint64_t num_bytes, const uint64_t elem_size);

        void deallocate(uint8_t* ptr, const uint64_t num_bytes);
        void release();
        void setThreshold(const uint64_t thresh);
        uint64_t getThreshold() const;

      private:
        const uint64_t m_threshold;
        SmallAllocator m_small_allocator;
        LargeAllocator m_large_allocator;
    };

    // implementation

    template <class SmallAllocator, class LargeAllocator>
    typename CombinedPolicy<SmallAllocator, LargeAllocator>::Ptr
    CombinedPolicy<SmallAllocator, LargeAllocator>::create()
    {
        return std::make_shared<CombinedPolicy<SmallAllocator, LargeAllocator>>();
    }

    template <class SmallAllocator, class LargeAllocator>
    CombinedPolicy<SmallAllocator, LargeAllocator>::CombinedPolicy(const uint64_t threshold)
        : m_threshold(threshold)
    {
    }

    template <class SmallAllocator, class LargeAllocator>
    uint8_t* CombinedPolicy<SmallAllocator, LargeAllocator>::allocate(const uint64_t num_bytes,
                                                                      const uint64_t elem_size)
    {
        if (num_bytes > m_threshold)
        {
            return m_small_allocator.allocate(num_bytes, elem_size);
        }
        else
        {
            return m_large_allocator.allocate(num_bytes, elem_size);
        }
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::deallocate(uint8_t* ptr, const uint64_t num_bytes)
    {
        if (num_bytes > m_threshold)
        {
            return m_small_allocator.deallocate(ptr, num_bytes);
        }
        else
        {
            m_large_allocator.deallocate(ptr, num_bytes);
        }
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::release()
    {
        m_small_allocator.release();
        m_large_allocator.release();
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::setThreshold(const uint64_t thresh)
    {
        m_threshold = thresh;
    }

    template <class SmallAllocator, class LargeAllocator>
    uint64_t CombinedPolicy<SmallAllocator, LargeAllocator>::getThreshold() const
    {
        return m_threshold;
    }
}
