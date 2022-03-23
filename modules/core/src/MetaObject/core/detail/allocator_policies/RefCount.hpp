#pragma once

#include "MetaObject/logging/logging.hpp"

#include <cstdint>

namespace mo
{

    // Maintain a reference count to the number of allocated blocks of data out of this allocator
    // Then warn on delete with non zero ref count
    template <class BaseAllocator>
    class RefCountPolicy : public BaseAllocator
    {
      public:
        RefCountPolicy() = default;
        RefCountPolicy(const RefCountPolicy&) = default;
        RefCountPolicy(RefCountPolicy&&) noexcept = default;
        RefCountPolicy& operator=(const RefCountPolicy&) = default;
        RefCountPolicy& operator=(RefCountPolicy&&) noexcept = default;
        ~RefCountPolicy();
        uint8_t* allocate(size_t num_bytes, size_t elem_size) override;

        void deallocate(uint8_t* ptr, size_t num_bytes) override;

        size_t refCount() const;

      private:
        size_t m_ref_count = 0;
    };

    // Implemtation

    template <class BaseAllocator>
    RefCountPolicy<BaseAllocator>::~RefCountPolicy()
    {
        if (m_ref_count != 0)
        {
            MO_LOG(warn, "Trying to delete allocator with {} data still referencing it", m_ref_count);
        }
    }

    template <class BaseAllocator>
    uint8_t* RefCountPolicy<BaseAllocator>::allocate(const size_t num_bytes, const size_t elem_size)
    {
        auto ptr = BaseAllocator::allocate(num_bytes, elem_size);
        if (ptr)
        {
            ++m_ref_count;
        }
        return ptr;
    }

    template <class BaseAllocator>
    void RefCountPolicy<BaseAllocator>::deallocate(uint8_t* ptr, const size_t num_bytes)
    {
        BaseAllocator::deallocate(ptr, num_bytes);
        --m_ref_count;
    }

    template <class BaseAllocator>
    size_t RefCountPolicy<BaseAllocator>::refCount() const
    {
        return m_ref_count;
    }
} // namespace mo
