#include "allocator_policies/Pool.hpp"

namespace mo
{
    template <class T, class XPU>
    class PinnedStlAllocator
    {
      public:
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        template <class U>
        struct rebind
        {
            typedef PinnedStlAllocator<U, XPU> other;
        };

        pointer allocate(size_type n, std::allocator<void>::const_pointer)
        {
            return allocate(n);
        }

        pointer allocate(size_type n)
        {
            pointer output = XPU::allocate(n * sizeof(pointer));
            return output;
        }

        void deallocate(pointer ptr, size_type)
        {
            XPU::deallocate(ptr);
        }
    };

    template <class T, class XPU>
    bool operator==(const PinnedStlAllocator<T, XPU>& lhs, const PinnedStlAllocator<T, XPU>& rhs)
    {
        return &lhs == &rhs;
    }
    template <class T, class XPU>
    bool operator!=(const PinnedStlAllocator<T, XPU>& lhs, const PinnedStlAllocator<T, XPU>& rhs)
    {
        return &lhs != &rhs;
    }

    template <class T, class XPU = CPU>
    class PinnedStlAllocatorPool
    {
      public:
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        PinnedStlAllocatorPool(const std::shared_ptr<CPUMemoryPool>& pool)
            : m_pool(pool)
        {
        }

        template <class U>
        struct rebind
        {
            typedef PinnedStlAllocatorPool<U> other;
        };

        pointer allocate(size_type n, std::allocator<void>::const_pointer)
        {
            return allocate(n);
        }

        pointer allocate(size_type n)
        {
            pointer ptr = nullptr;
            ptr = reinterpret_cast<pointer>(m_pool->allocate(n * sizeof(T), sizeof(T)));
            return ptr;
        }

        void deallocate(pointer ptr, size_type n)
        {
            m_pool->deallocate(reinterpret_cast<unsigned char*>(ptr), n * sizeof(T));
        }

      private:
        std::shared_ptr<CPUMemoryPool> m_pool;
    };

} // namespace mo
