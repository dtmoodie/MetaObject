#ifndef MO_CORE_ALLOCATOR_STL_ALLOCATOR_HPP
#define MO_CORE_ALLOCATOR_STL_ALLOCATOR_HPP
#include "Allocator.hpp"

namespace mo
{
    template <class T>
    class StlAllocator
    {
      public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        template <class U>
        struct rebind
        {
            using other = StlAllocator<U>;
        };

        StlAllocator(std::shared_ptr<Allocator> allocator)
            : m_allocator(std::move(allocator))
        {
        }

        StlAllocator(const StlAllocator& allocator) = default;
        StlAllocator(StlAllocator&& allocator) = default;

        template <class U>
        StlAllocator(const StlAllocator<U>& other)
        {
            m_allocator = other.getAllocator();
        }

        template <class U>
        StlAllocator(StlAllocator<U>&& other)
        {
            m_allocator = other.getAllocator();
        }

        pointer allocate(size_type n, std::allocator<void>::const_pointer)
        {
            return allocate(n);
        }

        pointer allocate(size_type n)
        {
            auto output = static_cast<pointer>(static_cast<void*>(m_allocator->allocate(sizeof(T) * n, sizeof(T))));
            return output;
        }

        void deallocate(pointer ptr, size_type n)
        {
            m_allocator->deallocate(static_cast<uint8_t*>(static_cast<void*>(ptr)), n * sizeof(T));
        }

        bool operator==(const StlAllocator<T>& rhs) const
        {
            return this->m_allocator == rhs.m_allocator;
        }

        bool operator!=(const StlAllocator<T>& rhs) const
        {
            return this->m_allocator != rhs.m_allocator;
        }

        std::shared_ptr<Allocator> getAllocator() const
        {
            return m_allocator;
        }

      protected:
        std::shared_ptr<Allocator> m_allocator;
    };

} // namespace mo
#endif // MO_CORE_ALLOCATOR_STL_ALLOCATOR_HPP
