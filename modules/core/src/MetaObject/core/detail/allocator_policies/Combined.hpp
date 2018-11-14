#pragma once
#include <cstdint>

namespace mo
{
    template <class SmallAllocator, class LargeAllocator>
    class CombinedPolicy : virtual public SmallAllocator, virtual public LargeAllocator
    {
      public:
        CombinedPolicy(const uint64_t threshold = 1 * 1024 * 512);

        unsigned char* allocate(const uint64_t num_bytes, const uint64_t elem_size);

        void deallocate(uint8_t* ptr, const uint64_t num_bytes);
        void release();
        void setThreshold(const uint64_t thresh);
        uint64_t getThreshold() const;

      private:
        const uint64_t m_threshold;
    };

    // implementation

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
            return SmallAllocator::allocate(num_bytes, elem_size);
        }
        else
        {
            return LargeAllocator::allocate(num_bytes, elem_size);
        }
    }

    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::deallocate(uint8_t* ptr, const uint64_t num_bytes)
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
    template <class SmallAllocator, class LargeAllocator>
    void CombinedPolicy<SmallAllocator, LargeAllocator>::release()
    {
        SmallAllocator::release();
        LargeAllocator::release();
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
