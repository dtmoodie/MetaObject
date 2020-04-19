#ifndef MO_PARAMS_TSTLALLOCATOR_HPP
#define MO_PARAMS_TSTLALLOCATOR_HPP

namespace mo
{
    template <class T>
    class TStlAllocator
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
            using other = TStlAllocator<U>;
        };

        TStlAllocator(typename ParamAllocator::Ptr_t allocator = ParamAllocator::create())
            : m_allocator(std::move(allocator))
        {
        }

        TStlAllocator(TStlAllocator<T>&& other) = default;
        TStlAllocator(const TStlAllocator<T>& other) = default;

        template <class U>
        TStlAllocator(TStlAllocator<U>&& other, ct::EnableIf<sizeof(U) == sizeof(T), const void*> = nullptr)
        {
            m_allocator = other.getAllocator();
        }

        template <class U>
        TStlAllocator(TStlAllocator<U>&& other, ct::EnableIf<sizeof(U) != sizeof(T), const void*> = nullptr)
        {
            auto alloc = other.getAllocator();
            if (alloc)
            {
                m_allocator = ParamAllocator::create(alloc->getAllocator());
            }
        }

        template <class U>
        TStlAllocator(const TStlAllocator<U>& other, ct::EnableIf<sizeof(U) == sizeof(T), const void*> = nullptr)
            : m_allocator(other.getAllocator())
        {
        }

        template <class U>
        TStlAllocator(const TStlAllocator<U>& other, ct::EnableIf<sizeof(U) != sizeof(T), const void*> = nullptr)
        {
            auto alloc = other.getAllocator();
            if (alloc)
            {
                m_allocator = ParamAllocator::create(alloc->getAllocator());
            }
        }

        pointer allocate(size_type n, std::allocator<void>::const_pointer)
        {
            return allocate(n);
        }

        pointer allocate(size_type n)
        {
            auto output = m_allocator->allocate<T>(n);
            return output;
        }

        void deallocate(pointer ptr, size_type)
        {
            if (m_allocator)
            {
                m_allocator->deallocate(ptr);
            }
        }

        typename ParamAllocator::Ptr_t getAllocator() const
        {
            return m_allocator;
        }

        bool operator==(const TStlAllocator<T>& rhs) const
        {
            return this->m_allocator == rhs.m_allocator;
        }

        bool operator!=(const TStlAllocator<T>& rhs) const
        {
            return this->m_allocator != rhs.m_allocator;
        }

      private:
        typename ParamAllocator::Ptr_t m_allocator;
    };
} // namespace mo
#endif // MO_PARAMS_TSTLALLOCATOR_HPP