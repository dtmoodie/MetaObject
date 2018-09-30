#include "AllocatorImpl.hpp"
#include "MemoryPool.hpp"

namespace mo
{
    template <class T>
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
            typedef PinnedStlAllocator<U> other;
        };

        pointer allocate(size_type n, std::allocator<void>::const_pointer hint) { return allocate(n); }

        pointer allocate(size_type n)
        {
            pointer output = nullptr;
            MO_CUDA_ERROR_CHECK(cudaMallocHost(&output, n * sizeof(pointer)), "");
            return output;
        }

        void deallocate(pointer ptr, size_type n) { MO_CUDA_ERROR_CHECK(cudaFreeHost(ptr), ""); }
    };

    template <class T>
    bool operator==(const PinnedStlAllocator<T>& lhs, const PinnedStlAllocator<T>& rhs)
    {
        return &lhs == &rhs;
    }
    template <class T>
    bool operator!=(const PinnedStlAllocator<T>& lhs, const PinnedStlAllocator<T>& rhs)
    {
        return &lhs != &rhs;
    }

    // Share pinned pool with CpuPoolPolicy
    template <class T>
    class PinnedStlAllocatorPoolThread
    {
      public:
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        PinnedStlAllocatorPoolThread(const std::shared_ptr<CPUMemoryPool>& pool = CPUMemoryPool::instance())
            : m_pool(pool)
        {
        }

        template <class U>
        struct rebind
        {
            typedef PinnedStlAllocatorPoolThread<U> other;
        };
        pointer allocate(size_type n, std::allocator<void>::const_pointer hint) { return allocate(n); }

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
