#pragma once
#include "MetaObject/logging/logging.hpp"

#include <cstdint>

namespace mo
{
    template <class Allocator>
    class RefCountPolicy : virtual public Allocator
    {
      public:
        ~RefCountPolicy();
        uint8_t* allocate(const uint64_t num_bytes, const uint64_t elem_size);

        void deallocate(uint8_t* ptr, const uint64_t num_bytes);

        uint64_t refCount() const;

      private:
        uint64_t m_ref_count = 0;
    };

    // Implemtation

    template <class Allocator>
    RefCountPolicy<Allocator>::~RefCountPolicy()
    {
        if (m_ref_count != 0)
        {

            MO_LOG(warn, "Trying to delete allocator with {}  mats still referencing it", m_ref_count);
        }
    }

    template <class Allocator>
    uint8_t* RefCountPolicy<Allocator>::allocate(const uint64_t num_bytes, const uint64_t elem_size)
    {
        auto ptr = Allocator::allocate(num_bytes, elem_size);
        if (ptr)
        {
            ++m_ref_count;
        }
        return ptr;
    }

    template <class Allocator>
    void RefCountPolicy<Allocator>::deallocate(uint8_t* ptr, const uint64_t num_bytes)
    {
        Allocator::deallocate(ptr, num_bytes);
        --m_ref_count;
    }

    template <class Allocator>
    uint64_t RefCountPolicy<Allocator>::refCount() const
    {
        return m_ref_count;
    }
}
