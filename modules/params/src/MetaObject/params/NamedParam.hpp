#pragma once
#include <boost/optional.hpp>

#include <ct/VariadicTypedef.hpp>
#include <ct/type_traits.hpp>

namespace mo
{
    // https://www.fluentcpp.com/2016/12/08/strong-types-for-strong-interfaces/
    struct NamedParamBase
    {
    };

    template <class Tag, class T, class U = const T&>
    struct TNamedParam : public NamedParamBase
    {
        using type = T;
        using tag = Tag;
        using storage_type = U;
        using pointer_type = const T*;

        explicit TNamedParam(storage_type value)
            : m_value(value)
        {
        }

        TNamedParam(const TNamedParam<Tag, T, T>& other)
            : m_value(other.get())
        {
        }

        operator storage_type() const
        {
            return m_value;
        }

        const storage_type& get() const
        {
            return m_value;
        }

        const T* ptr() const
        {
            return &m_value;
        }

      private:
        storage_type m_value;
    };

    template <class Tag, class T>
    struct TNamedParam<Tag, T, T> : public NamedParamBase
    {
        using type = T;
        using tag = Tag;
        using storage_type = T;
        using pointer_type = const T*;

        explicit TNamedParam(storage_type value)
            : m_value(std::move(value))
        {
        }

        operator storage_type() const
        {
            return m_value;
        }

        const storage_type& get() const
        {
            return m_value;
        }

        const T* ptr() const
        {
            return &m_value;
        }

      private:
        storage_type m_value;
    };

    template <class Tag, class T>
    struct TNamedParam<Tag, const T*, const T*> : public NamedParamBase
    {
        using type = T;
        using tag = Tag;
        using storage_type = const T*;
        using pointer_type = const T*;

        explicit TNamedParam(storage_type value)
            : m_value(value)
        {
        }

        explicit TNamedParam(const boost::optional<const T>& value)
        {
            if (value)
            {
                m_value = &(*value);
            }
        }

        explicit TNamedParam(const boost::optional<T>& value)
        {
            if (value)
            {
                m_value = &(*value);
            }
        }

        operator storage_type() const
        {
            return m_value;
        }

        storage_type get() const
        {
            return m_value;
        }

        storage_type ptr() const
        {
            return m_value;
        }

      private:
        storage_type m_value = nullptr;
    };

    // Example
    // struct TimestampParameter;
    // using Timestamp = TNamedParam<TimestampParameter, mo::Time>

    template <class T>
    struct IsNamedParam
    {
        static constexpr bool value = ct::IsBase<ct::Base<NamedParamBase>, ct::Derived<T>>::value;
    };

    constexpr bool hasNamedParam(ct::VariadicTypedef<> = {})
    {
        return false;
    }

    template <class T, class... Ts>
    constexpr bool hasNamedParam(ct::VariadicTypedef<T, Ts...> = {})
    {
        return IsNamedParam<ct::decay_t<T>>::value || hasNamedParam(ct::VariadicTypedef<Ts...>{});
    }

    template <class Tag, class Type, class Storage, class Enable = void>
    struct TKeywordBase
    {
        using type = typename Tag::type;
        using tag = typename Tag::tag;

        Tag operator=(Storage data) const
        {
            return Tag(data);
        }
    };

    template <class Tag, class Type, class Storage>
    struct TKeywordBase<Tag, Type, Storage, ct::EnableIf<!std::is_pointer<typename std::decay<Type>::type>::value>>
    {
        using type = typename Tag::type;
        using tag = typename Tag::tag;

        // This returns a TNamedParam with the same tag but it stores a copy of the data instead of a const ref.
        // This is needed since the operator = could be passed a temporary which we can't take a const ref of
        TNamedParam<tag, type, type> operator=(type&& data) const
        {
            return TNamedParam<tag, type, type>(std::forward<type>(data));
        }

        Tag operator=(Storage data) const
        {
            return Tag(data);
        }
    };

    // Example
    // TKeyword<Timestamp>
    // static const constexpr TKeyword<Timestamp> timestamp;
    template <class Tag>
    struct TKeyword : TKeywordBase<Tag, typename Tag::type, typename Tag::storage_type>
    {
        using TKeywordBase<Tag, typename Tag::type, typename Tag::storage_type>::operator=;
        constexpr TKeyword()
        {
        }
    };

    // This simplifies the number of overloads needed below for some compilers
    struct NullArgType
    {
    };

    template <class Tag, class T>
    struct SameTag
    {
        static constexpr const bool value = false;
    };
    template <class Tag, class U, class V>
    struct SameTag<Tag, TNamedParam<typename Tag::tag, U, V>>
    {
        static constexpr const bool value = true;
    };

    template <class Tag, class T>
    constexpr typename Tag::pointer_type getKeywordInputOptionalImpl(T&&)
    {
        return nullptr;
    }

    template <class Tag, class T, class... Ts>
    constexpr auto getKeywordInputOptionalImpl(T&&, Ts&&... args)
        -> ct::DisableIf<SameTag<Tag, ct::decay_t<T>>::value, typename Tag::pointer_type>
    {
        return getKeywordInputOptionalImpl<Tag>(std::forward<Ts>(args)...);
    }

    template <class Tag, class U, class V, class... Ts>
    constexpr typename Tag::pointer_type getKeywordInputOptionalImpl(const TNamedParam<typename Tag::tag, U, V>& arg,
                                                                     Ts&&...)
    {
        return arg.ptr();
    }

    template <class Tag, class... Ts>
    constexpr typename Tag::pointer_type getKeywordInputOptional(Ts&&... args)
    {
        return getKeywordInputOptionalImpl<Tag>(std::forward<Ts>(args)..., NullArgType{});
    }

    //////////////////////////////////////////////////////////////////
    // getKeywordInputDefault
    template <class Tag>
    constexpr typename Tag::storage_type getKeywordInputDefault(typename Tag::storage_type dv)
    {
        return dv;
    }

    template <class Tag, class T, class... Ts>
    constexpr typename Tag::storage_type getKeywordInputDefault(typename Tag::storage_type dv, T&&, Ts&&... args)
    {
        return getKeywordInputDefault<Tag>(dv, std::forward<Ts>(args)...);
    }

    template <class Tag, class... Ts>
    constexpr typename Tag::storage_type getKeywordInputDefault(typename Tag::storage_type, Tag arg, Ts&&...)
    {
        return arg.get();
    }
} // namespace mo
