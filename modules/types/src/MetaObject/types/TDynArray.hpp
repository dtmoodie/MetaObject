#ifndef MO_TYPES_TARRAY_HPP
#define MO_TYPES_TARRAY_HPP
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/logging/logging.hpp>

#include <ct/reflect.hpp>
#include <ct/types/TArrayView.hpp>

#include <ct/reflect_macros.hpp>

#include <memory>
#include <vector>
namespace mo
{
    template <class T, class ALLOC>
    struct TDynArray
    {
        using type = T;

        TDynArray(std::shared_ptr<ALLOC> alloc = ALLOC::getDefault(), size_t num_elems = 0);
        TDynArray(const ct::TArrayView<const T>&, std::shared_ptr<ALLOC> alloc = ALLOC::getDefault());
        TDynArray(ct::TArrayView<T>&&, std::weak_ptr<ALLOC> alloc);
        template <class U>
        TDynArray(const TDynArray<U, ALLOC>& other);

        TDynArray(const TDynArray&);
        TDynArray(TDynArray&&) noexcept;

        TDynArray& operator=(const TDynArray&);
        TDynArray& operator=(TDynArray&&) noexcept;
        template <class U>
        TDynArray& operator=(const std::vector<U>&);
        ~TDynArray();

        operator ct::TArrayView<T>();

        operator ct::TArrayView<const T>() const;

        ct::TArrayView<T> mutableView();

        ct::TArrayView<const T> view() const;

        void resize(size_t size);

        bool empty() const;

        size_t size() const;

        template <class U>
        operator TDynArray<U, ALLOC>() &&;

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
    {
        if (num_elems)
        {
            resize(num_elems);
        }
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>::TDynArray(const ct::TArrayView<const T>& view, std::shared_ptr<ALLOC> alloc)
        : m_allocator(alloc)
    {
        resize(view.size());
        if (view.size())
        {
            std::memcpy(m_data.begin(), view.begin(), SizeOf<T>::value * view.size());
        }
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>::TDynArray(ct::TArrayView<T>&& view, std::weak_ptr<ALLOC> alloc)
        : m_data(std::move(view))
        , m_allocator(std::move(alloc))
    {
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>::TDynArray(TDynArray&& other) noexcept
        : m_data(std::move(other.m_data))
        , m_allocator(std::move(other.m_allocator))
    {
        other.m_data = nullptr;
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>::TDynArray(const TDynArray& other)
    {
        m_allocator = other.m_allocator;
        auto view = other.view();
        resize(view.size());
        if (view.size())
        {
            memcpy(m_data.data(), view.data(), SizeOf<T>::value * view.size());
        }
    }

    template <class T, class ALLOC>
    template <class U>
    TDynArray<T, ALLOC>::TDynArray(const TDynArray<U, ALLOC>& other)
    {
        static_assert(SizeOf<T>::value == SizeOf<U>::value, "");
        m_allocator = other.m_allocator;

        auto view = other.view();
        resize(view.size());
        if (view.size())
        {
            memcpy(m_data.data(), view.data(), SizeOf<U>::value * view.size());
        }
    }

    template <class T, class ALLOC>
    template <class U>
    TDynArray<T, ALLOC>::operator TDynArray<U, ALLOC>() &&
    {
        static_assert(SizeOf<T>::value == SizeOf<U>::value, "");
        return TDynArray<U, ALLOC>(std::move(m_data), std::move(m_allocator));
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>& TDynArray<T, ALLOC>::operator=(TDynArray&& other) noexcept
    {
        m_data = std::move(other.m_data);
        other.m_data = nullptr;
        return *this;
    }

    template <class T, class ALLOC>
    template <class U>
    TDynArray<T, ALLOC>& TDynArray<T, ALLOC>::operator=(const std::vector<U>& data)
    {
        resize(data.size());
        if (data.size())
        {
            memcpy(m_data.data(), data.data(), SizeOf<U>::value * data.size());
        }

        return *this;
    }

    template <class T, class ALLOC>
    TDynArray<T, ALLOC>& TDynArray<T, ALLOC>::operator=(const TDynArray& other)
    {
        m_allocator = other.m_allocator;
        auto view = other.view();
        resize(view.size());
        if (view.size())
        {
            memcpy(m_data.data(), other.m_data.data(), SizeOf<T>::value * other.size());
        }

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
        ct::TArrayView<const T> ret(m_data);
        return ret;
    }

    template <class T, class ALLOC>
    ct::TArrayView<T> TDynArray<T, ALLOC>::mutableView()
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
        if (size > 0)
        {
            m_data = ct::TArrayView<T>(allocator->template allocate<T>(size), size);
        }
        else
        {
            m_data = ct::TArrayView<T>();
        }
    }

    template <class T, class ALLOC>
    bool TDynArray<T, ALLOC>::empty() const
    {
        return m_data.empty();
    }

    template <class T, class ALLOC>
    size_t TDynArray<T, ALLOC>::size() const
    {
        return m_data.size();
    }
} // namespace mo

namespace ct
{
    REFLECT_TEMPLATED_BEGIN(mo::TDynArray)
        MEMBER_FUNCTION(view, &DataType::mutableView)
        MEMBER_FUNCTION(constView, &DataType::view)
        MEMBER_FUNCTION(resize)
        MEMBER_FUNCTION(empty)
        MEMBER_FUNCTION(size)
    REFLECT_END;
} // namespace ct
#endif // MO_TYPES_TARRAY_HPP
