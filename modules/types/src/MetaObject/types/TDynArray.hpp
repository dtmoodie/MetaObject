#ifndef MO_TYPES_TARRAY_HPP
#define MO_TYPES_TARRAY_HPP
#include <MetaObject/logging/logging.hpp>
#include <ct/types/TArrayView.hpp>

#include <memory>
#include <vector>
namespace mo
{
    template <class T, class ALLOC>
    struct TDynArray
    {
        TDynArray(std::shared_ptr<ALLOC> alloc = ALLOC::getDefault(), size_t num_elems = 0);
        TDynArray(const TDynArray&);
        TDynArray(TDynArray&&) noexcept;
        TDynArray& operator=(const TDynArray&);
        TDynArray& operator=(TDynArray&&) noexcept;
        TDynArray& operator=(const std::vector<T>&);
        ~TDynArray();

        operator ct::TArrayView<T>();

        operator ct::TArrayView<const T>() const;

        ct::TArrayView<T> view();

        ct::TArrayView<const T> view() const;

        void resize(size_t size);

        bool empty() const
        {
            return m_data.empty();
        }

        size_t size() const
        {
            return m_data.size();
        }

      private:
        ct::TArrayView<T> m_data;
        std::weak_ptr<ALLOC> m_allocator;
    };

    ///////////////////////////////////////////////////////////////////
    /// IMPLEMENTATION
    ///////////////////////////////////////////////////////////////////

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>::TDynArray(std::shared_ptr<ALLOC> alloc, size_t num_elems)
        : m_allocator(alloc)
        , m_data(alloc->template allocate<T>(num_elems), num_elems)
    {
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>::TDynArray(TDynArray&& other) noexcept : m_data(std::move(other.m_data)),
                                                                 m_allocator(std::move(other.m_allocator))
    {
        other.m_data = nullptr;
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>::TDynArray(const TDynArray& other)
    {
        auto allocator = other.m_allocator.lock();
        m_data = ct::TArrayView<T>(allocator->template allocate<T>(other.m_data.size()), other.m_data.size());
        m_allocator = allocator;
        memcpy(m_data.data(), other.m_data.data(), sizeof(T) * other.m_data.size());
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>& TDynArray<T, ALLOC>::operator=(TDynArray&& other) noexcept
    {
        m_data = std::move(other.m_data);
        other.m_data = nullptr;
        return *this;
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>& TDynArray<T, ALLOC>::operator=(const std::vector<T>& data)
    {
        resize(data.size());
        memcpy(m_data.data(), data.data(), sizeof(T) * data.size());
        return *this;
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>& TDynArray<T, ALLOC>::operator=(const TDynArray& other)
    {
        m_allocator = other.m_allocator;
        auto allocator = other.m_allocator.lock();
        m_data = ct::TArrayView<T>(allocator->template allocate<T>(other.size()), other.size());
        memcpy(m_data.data(), other.m_data.data(), sizeof(T) * other.size());
        return *this;
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>::~TDynArray()
    {
        if (this->m_data.data())
        {
            auto locked = m_allocator.lock();
            if (locked)
            {
                locked->deallocate(m_data.data(), m_data.size());
            }
            else
            {
                MO_LOG(error, "The allocator for this data has been dsetroyed before the data has been released");
            }
        }
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>::operator ct::TArrayView<T>()
    {
        return m_data;
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>::operator ct::TArrayView<const T>() const
    {
        return m_data;
    }

    template <class T, class ALLOC>
    ct::TArrayView<T> TDynArray<T, ALLOC>::view()
    {
        return m_data;
    }

    template <class T, class ALLOC>
    ct::TArrayView<const T> TDynArray<T, ALLOC>::view() const
    {
        return m_data;
    }

    template <class T, class ALLOC>
    void TDynArray<T, ALLOC>::resize(size_t size)
    {
        auto ptr = m_data.data();
        auto allocator = m_allocator.lock();
        MO_ASSERT(allocator != nullptr);
        if (ptr)
        {
            allocator->deallocate(ptr, m_data.size());
        }
        m_data = ct::TArrayView<T>(allocator->template allocate<T>(size), size);
    }
}
#endif // MO_TYPES_TARRAY_HPP
