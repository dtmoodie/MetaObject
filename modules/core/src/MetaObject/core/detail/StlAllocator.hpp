#include "Allocator.hpp"

namespace mo
{
    template<class T> class PinnedStlAllocator {
    public:
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        template< class U > struct rebind {
            typedef PinnedStlAllocator<U> other;
        };
        pointer allocate(size_type n, std::allocator<void>::const_pointer hint) {
            return allocate(n);
        }

        pointer allocate(size_type n) {
            pointer output = nullptr;
            cudaSafeCall(cudaMallocHost(&output, n * sizeof(pointer)));
            return output;
        }

        void deallocate(pointer ptr, size_type n) {
            cudaSafeCall(cudaFreeHost(ptr));
        }
    };


    template<class T> bool operator==(const PinnedStlAllocator<T>& lhs, const PinnedStlAllocator<T>& rhs) {
        return &lhs == &rhs;
    }
    template<class T> bool operator!=(const PinnedStlAllocator<T>& lhs, const PinnedStlAllocator<T>& rhs) {
        return &lhs != &rhs;
    }

    // Share pinned pool with CpuPoolPolicy
    template<class T> class PinnedStlAllocatorPoolThread {
    public:
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        template< class U > struct rebind {
            typedef PinnedStlAllocatorPoolThread<U> other;
        };
        pointer allocate(size_type n, std::allocator<void>::const_pointer hint) {
            return allocate(n);
        }

        pointer allocate(size_type n) {
            pointer ptr = nullptr;
            CpuMemoryPool::threadInstance()->allocate((void**)&ptr, n * sizeof(T), sizeof(T));
            return ptr;
        }

        void deallocate(pointer ptr, size_type n) {
            CpuMemoryPool::threadInstance()->deallocate(ptr, n * sizeof(T));
        }
    };

    template<class T> class PinnedStlAllocatorPoolGlobal {
    public:
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        template< class U > struct rebind {
            typedef PinnedStlAllocatorPoolGlobal<U> other;
        };

        pointer allocate(size_type n, std::allocator<void>::const_pointer hint) {
            return allocate(n);
        }

        pointer allocate(size_type n) {
            pointer ptr = nullptr;
            CpuMemoryPool::globalInstance()->allocate((void**)&ptr, n * sizeof(T), sizeof(T));
            return ptr;
        }

        void deallocate(pointer ptr, size_type n) {
            CpuMemoryPool::globalInstance()->deallocate(ptr, n * sizeof(T));
        }
    };

    template<> class PinnedStlAllocatorPoolGlobal<void> {
    public:
        typedef void value_type;
        typedef void* pointer;
        typedef const void* const_pointer;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        template< class U > struct rebind {
            typedef PinnedStlAllocatorPoolGlobal<U> other;
        };

        pointer allocate(size_type n, std::allocator<void>::const_pointer hint) {
            return allocate(n);
        }

        pointer allocate(size_type n) {
            pointer ptr = nullptr;
            CpuMemoryPool::globalInstance()->allocate(&ptr, n, 1);
            return ptr;
        }

        void deallocate(pointer ptr, size_type n) {
            CpuMemoryPool::globalInstance()->deallocate(ptr, n);
        }
    };
    template<class T, class U> bool operator==(const PinnedStlAllocatorPoolGlobal<T>& lhs, const PinnedStlAllocatorPoolGlobal<U>& rhs) {
        return &lhs == &rhs;
    }
    template<class T, class U> bool operator!=(const PinnedStlAllocatorPoolGlobal<T>& lhs, const PinnedStlAllocatorPoolGlobal<U>& rhs) {
        return &lhs != &rhs;
    }
} // namespace mo