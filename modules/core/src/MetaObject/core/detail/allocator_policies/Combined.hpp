#pragma once
#include "../Allocator.hpp"
#include <cstdint>
#include <memory>

namespace mo
{
    // Combines multiple allocators into one
    // Allocate from SmallAllocator unless the requested allocation is larger than threshold
    // in which case allocate out of LargeAllocator
    template <class SmallAllocator, class LargeAllocator>
    class CombinedPolicy : public SmallAllocator::Allocator_t
    {
      public:
        using Ptr_t = std::shared_ptr<CombinedPolicy>;
        static Ptr_t create();

        CombinedPolicy(size_t threshold = 1 * 1024 * 512);

        uint8_t* allocate(size_t num_bytes, size_t elem_size) override;

        void deallocate(uint8_t* ptr, size_t num_bytes) override;
        void release() override;

        void setThreshold(size_t thresh);
        size_t getThreshold() const;

      private:
        size_t m_threshold;
        SmallAllocator m_small_allocator;
        LargeAllocator m_large_allocator;
    };

    // implementation

    template <class SmallAllocator, class LargeAllocator>
    typename CombinedPolicy<SmallAllocator, LargeAllocator>::Ptr_t
    CombinedPolicy<SmallAllocator, LargeAllocator>::create()
    {
        return std::make_shared<CombinedPolicy<SmallAllocator, LargeAllocator>>();
    }

    template <class SmallAllocator, class LargeAllocator>
    CombinedPolicy<SmallAllocator, LargeAllocator>::CombinedPolicy(const size_t threshold)
        : m_threshold(threshold)
    {
    }

    template <class SmallAllocator, class LargeAllocator>
    uint8_t* CombinedPolicy<SmallAllocator, LargeAllocator>::allocate(const size_t num_bytes, const size_t elem_size)
    {
        if (num_bytes < m_threshold)
        {
            return m_small_allocator.allocate(num_bytes, elem_size);
        }

        return m_large_allocator.allocate(num_bytes, elem_size);
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::deallocate(uint8_t* ptr, const size_t num_bytes)
    {
        if (num_bytes < m_threshold)
        {
            return m_small_allocator.deallocate(ptr, num_bytes);
        }
        m_large_allocator.deallocate(ptr, num_bytes);
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::release()
    {
        m_small_allocator.release();
        m_large_allocator.release();
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::setThreshold(const size_t thresh)
    {
        m_threshold = thresh;
    }

    template <class SmallAllocator, class LargeAllocator>
    size_t CombinedPolicy<SmallAllocator, LargeAllocator>::getThreshold() const
    {
        return m_threshold;
    }
}
