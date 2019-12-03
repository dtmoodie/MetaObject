/*#ifndef MO_PARAMS_PARAM_ALLOCATOR_HPP
#define MO_PARAMS_PARAM_ALLOCATOR_HPP
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/logging/logging.hpp>

#include <vector>

namespace mo
{
    template <class T, class U>
    T* ptrCast(U* ptr)
    {
        return static_cast<T*>(static_cast<void*>(ptr));
    }

    template <class T, class U>
    const T* ptrCast(const U* ptr)
    {
        return static_cast<const T*>(static_cast<const void*>(ptr));
    }

    struct IParamDataAllocator
    {
        struct AllocationData
        {
            uint8_t* allocated_data = nullptr;
            uint8_t* allocated_begin = nullptr;
            uint8_t* allocated_end = nullptr;
        };

        IParamDataAllocator(Allocator::Ptr_t alloc);

        void setPaddingSize(std::size_t header, std::size_t footer = 0);

        AllocationData allocate(size_t num_bytes, size_t element_size = 1);
        void deallocate(uint8_t* ptr, size_t num_bytes);

        void release();

        std::shared_ptr<mo::Allocator> getAllocator() const;
        void setAllocator(std::shared_ptr<mo::Allocator> allocator);

      protected:
        std::size_t m_header_size = 0;
        std::size_t m_footer_size = 0;
        std::shared_ptr<mo::Allocator> m_allocator;
    };

    template <class T>
    struct TParamDataAllocator : public IParamDataAllocator
    {
        using type = T;

        TParamDataAllocator(Allocator::Ptr_t allocator = Allocator::getDefault())
            : IParamDataAllocator(std::move(allocator))
        {
        }

        template <class... ARGS>
        type create(ARGS&&... args)
        {
            return type(std::forward<ARGS>(args)...);
        }
    };

    template <class T>
    struct TDataContainerAllocator
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
            using other = TDataContainerAllocator<U>;
        };

        TDataContainerAllocator(IParamDataAllocator& allocator)
            : m_allocator(allocator)
        {
        }

        pointer allocate(size_type n, std::allocator<void>::const_pointer)
        {
            return allocate(n);
        }

        pointer allocate(size_type n)
        {
            if (m_current_allocation.allocated_begin)
            {
                m_previous_allocation = m_current_allocation;
                m_current_allocation = {nullptr, nullptr, nullptr};
            }
            MO_ASSERT(m_current_allocation.allocated_data == nullptr);
            m_current_allocation = m_allocator.allocate(sizeof(T) * n, sizeof(T));
            m_current_allocator = m_allocator.getAllocator();
            return static_cast<pointer>(static_cast<void*>(m_current_allocation.allocated_data));
        }

        void deallocate(pointer ptr, size_type)
        {
            MO_ASSERT(m_current_allocation.allocated_begin);
            MO_ASSERT(ptrCast<uint8_t>(ptr) >= m_current_allocation.allocated_begin &&
                      ptrCast<uint8_t>(ptr) < m_current_allocation.allocated_end);
            m_current_allocator->deallocate(m_current_allocation.allocated_begin,
                                            m_current_allocation.allocated_end - m_current_allocation.allocated_begin);
        }

        bool operator==(const TDataContainerAllocator<T>& rhs) const
        {
            return &m_allocator == &rhs.m_allocator;
        }

        bool operator!=(const TDataContainerAllocator<T>& rhs) const
        {
            return &m_allocator != &rhs.m_allocator;
        }

        void* allocateSerializationBuffer(std::size_t header, std::size_t data, std::size_t footer = 0)
        {
            if (m_current_allocation.allocated_begin)
            {
                if ((m_current_allocation.allocated_data - m_current_allocation.allocated_begin >= header) &&
                    (m_current_allocation.allocated_data + data + footer <= m_current_allocation.allocated_end))
                {
                    return m_current_allocation.allocated_data - header;
                }

                // current allocation size not suitable :/
                m_allocator.setPaddingSize(header, footer);

                // auto allocation = m_state->m_allocator.allocate()
            }
            else
            {

                m_allocator.setPaddingSize(header, footer);
            }
        }

      protected:
        IParamDataAllocator& m_allocator;
        Allocator::Ptr_t m_current_allocator;
        IParamDataAllocator::AllocationData m_current_allocation;
        IParamDataAllocator::AllocationData m_previous_allocation;
    };

    template <class T>
    struct TParamDataAllocator<std::vector<T>> : public IParamDataAllocator
    {
        using type = std::vector<T, TDataContainerAllocator<T>>;

        TParamDataAllocator(Allocator::Ptr_t allocator = Allocator::getDefault())
            : IParamDataAllocator(std::move(allocator))
        {
        }

        type create(std::size_t sz = 0)
        {
            type output(TDataContainerAllocator<T>(*this));
            output.reserve(sz);
            return output;
        }
    };
}

#endif // MO_PARAMS_PARAM_ALLOCATOR_HPP
*/
