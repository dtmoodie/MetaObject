#pragma once

#include "MetaObject/logging/logging.hpp"

#include <cstdint>

namespace mo
{
    template <class BaseAllocator>
    class RefCountPolicy : public BaseAllocator
    {
      public:
        ~RefCountPolicy();
        uint8_t* allocate(const uint64_t num_bytes, const uint64_t elem_size) override;

        void deallocate(uint8_t* ptr, const uint64_t num_bytes) override;

        uint64_t refCount() const;

      private:
        uint64_t m_ref_count = 0;
    };

    // Implemtation

    template <class BaseAllocator>
    RefCountPolicy<BaseAllocator>::~RefCountPolicy()
    {
        if (m_ref_count != 0)
        {

            MO_LOG(warn, "Trying to delete allocator with {}  mats still referencing it", m_ref_count);
        }
    }

    template <class BaseAllocator>
    uint8_t* RefCountPolicy<BaseAllocator>::allocate(const uint64_t num_bytes, const uint64_t elem_size)
    {
        auto ptr = BaseAllocator::allocate(num_bytes, elem_size);
        if (ptr)
        {
            ++m_ref_count;
        }
        return ptr;
    }

    template <class BaseAllocator>
    void RefCountPolicy<BaseAllocator>::deallocate(uint8_t* ptr, const uint64_t num_bytes)
    {
        BaseAllocator::deallocate(ptr, num_bytes);
        --m_ref_count;
    }

    template <class BaseAllocator>
    uint64_t RefCountPolicy<BaseAllocator>::refCount() const
    {
        return m_ref_count;
    }
}
