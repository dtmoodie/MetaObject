#pragma once
#include <boost/optional.hpp>

#include <ct/VariadicTypedef.hpp>
#include <ct/type_traits.hpp>

namespace mo
{
    template <class T>
    const T* getPointer(const T& ref)
    {
        return &ref;
    }

    template <class T>
    const T* getPointer(const T* ptr)
    {
        return ptr;
    }

    template <class T>
    T* getPointer(T* ptr)
    {
        return ptr;
    }

    template <class T>
    T* getPointer(T& ref)
    {
        return &ref;
    }

    template <class Storage, class Pointer>
    struct TaggedBase
    {
        template <class T>
        TaggedBase(T&& val)
            : m_value(std::forward<T>(val))
        {
        }

        operator Storage() const
        {
            return m_value;
        }

        operator Pointer()
        {
            return getPointer(m_value);
        }

      private:
        Storage m_value;
    };

    template <class Storage>
    struct TaggedBase<Storage, Storage>
    {
        template <class T>
        TaggedBase(T&& val)
            : m_value(std::forward<T>(val))
        {
        }

        template <class T>
        TaggedBase(boost::optional<T>& val)
            : m_value(val.get_ptr())
        {
        }

        operator Storage() const
        {
            return m_value;
        }

      private:
        Storage m_value;
    };

    template <class Tag>
    struct TaggedValue : TaggedBase<typename Tag::storage_type, typename Tag::pointer_type>
    {
        using type = typename Tag::value_type;
        using tag = Tag;
        using storage_type = typename Tag::storage_type;

        using Super_t = TaggedBase<typename Tag::storage_type, typename Tag::pointer_type>;

        template <class T>
        TaggedValue(T&& value)
            : Super_t(std::forward<T>(value))
        {
        }

        TaggedValue(const TaggedValue& other) = default;
        TaggedValue(TaggedValue&& other) = default;

        TaggedValue& operator=(const TaggedValue& other) = default;
        TaggedValue& operator=(TaggedValue&& other) = default;
    };

    // Example
    // struct TimestampParameter;
    // using Timestamp = TaggedValue<TimestampParameter, mo::Time>

    template <class T>
    struct IsTaggedValue : std::false_type
    {
    };

    template <class T>
    struct IsTaggedValue<TaggedValue<T>> : std::true_type
    {
    };

    constexpr bool hasTaggedValue(ct::VariadicTypedef<> = {})
    {
        return false;
    }

    template <class T, class... Ts>
    constexpr bool hasTaggedValue(ct::VariadicTypedef<T, Ts...> = {})
    {
        return IsTaggedValue<ct::decay_t<T>>::value || hasTaggedValue(ct::VariadicTypedef<Ts...>{});
    }

    template <class Tag, class Type, class Storage>
    struct TKeywordBase
    {
        template <class T>
        constexpr TaggedValue<Tag> operator=(T&& data) const
        {
            return TaggedValue<Tag>(std::forward<T>(data));
        }
    };

    // Example
    // TKeyword<Timestamp>
    // static const constexpr TKeyword<Timestamp> timestamp;
    template <class Tag>
    struct TKeyword : TKeywordBase<Tag, typename Tag::value_type, typename Tag::storage_type>
    {
        using TKeywordBase<Tag, typename Tag::value_type, typename Tag::storage_type>::operator=;
        constexpr TKeyword()
        {
        }
    };

    struct NullArgType
    {
        constexpr NullArgType()
        {
        }
    };

    template <class Tag, class T>
    typename Tag::pointer_type getKeywordInputOptionalImpl(T&&)
    {
        return nullptr;
    }

    template <class Tag, class... Ts>
    typename Tag::pointer_type getKeywordInputOptionalImpl(TaggedValue<Tag>&& arg, Ts&&...)
    {
        return arg;
    }

    template <class Tag, class T, class... Ts>
    auto getKeywordInputOptionalImpl(T&&, Ts&&... args)
        -> ct::EnableIf<!std::is_same<ct::decay_t<T>, TaggedValue<Tag>>::value, typename Tag::pointer_type>
    {
        // Stip off incorrect arg and forward to next iteration
        return getKeywordInputOptionalImpl<Tag>(std::forward<Ts>(args)...);
    }

    template <class Tag, class Types, bool STRICT>
    struct GetKeywordInputOptional
    {
    };

    template <class Tag, bool STRICT>
    struct GetKeywordInputOptional<Tag, ct::VariadicTypedef<>, STRICT>
    {
        using Pointer_t = typename Tag::pointer_type;

        static Pointer_t get()
        {
            return nullptr;
        }
    };

    // This is the case that just strips off an argument and forwards to the next one
    template <class Tag, class T, class... Ts, bool STRICT>
    struct GetKeywordInputOptional<Tag, ct::VariadicTypedef<T, Ts...>, STRICT>
    {
        using Pointer_t = typename Tag::pointer_type;
        template <class U, class... Us>
        static Pointer_t get(U&& arg, Us&&... args)
        {
            return GetKeywordInputOptional<Tag, ct::VariadicTypedef<Ts...>, STRICT>::get(std::forward<Us>(args)...);
        }
    };

    // if we're not being strict, we can do an implicit conversion
    template <class Tag, class... Ts>
    struct GetKeywordInputOptional<Tag, ct::VariadicTypedef<typename Tag::storage_type, Ts...>, false>
    {
        using Pointer_t = typename Tag::pointer_type;
        template <class U, class... Us>
        static Pointer_t get(U&& arg, Us&&...)
        {
            return getPointer(arg);
        }
    };

    template <class Tag, class... Ts, bool STRICT>
    struct GetKeywordInputOptional<Tag, ct::VariadicTypedef<TaggedValue<Tag>, Ts...>, STRICT>
    {
        using Pointer_t = typename Tag::pointer_type;
        template <class U, class... Us>
        static Pointer_t get(U&& arg, Us&&...)
        {
            return arg;
        }
    };

    template <class Tag, bool STRICT = false, class... Ts>
    typename Tag::pointer_type getKeywordInputOptional(Ts&&... args)
    {
        // have to use a NullArgType for msvc due to some weirdness in template parsing
        return GetKeywordInputOptional<Tag, ct::VariadicTypedef<ct::decay_t<Ts>...>, STRICT>::get(
            std::forward<Ts>(args)...);
    }

    //////////////////////////////////////////////////////////////////
    // getKeywordInputDefault

    template <class Tag, class T, bool STRICT>
    struct GetKeywordInputDefault
    {
    };

    template <class Tag, class... Ts>
    struct GetKeywordInputDefault<Tag, ct::VariadicTypedef<typename Tag::storage_type, Ts...>, false>
    {
        template <class U, class... Us>
        static typename Tag::storage_type get(typename Tag::storage_type&&, U&& arg, Us&&...)
        {
            return std::move(arg);
        }
    };

    // explicitly tagged value
    template <class Tag, class... Ts, bool STRICT>
    struct GetKeywordInputDefault<Tag, ct::VariadicTypedef<TaggedValue<Tag>, Ts...>, STRICT>
    {
        template <class U, class... Us>
        static typename Tag::storage_type get(typename Tag::storage_type&&, U&& arg, Us&&...)
        {
            return std::move(arg);
        }
    };

    template <class Tag, class T, class... Ts, bool STRICT>
    struct GetKeywordInputDefault<Tag, ct::VariadicTypedef<T, Ts...>, STRICT>
    {
        template <class U, class... Us>
        static typename Tag::storage_type get(typename Tag::storage_type&& dv, U&& arg, Us&&... args)
        {
            return GetKeywordInputDefault<Tag, ct::VariadicTypedef<Ts...>, STRICT>::get(std::move(dv),
                                                                                        std::forward<Us>(args)...);
        }
    };

    template <class Tag, bool STRICT>
    struct GetKeywordInputDefault<Tag, ct::VariadicTypedef<>, STRICT>
    {
        static typename Tag::storage_type get(typename Tag::storage_type&& dv)
        {
            return std::move(dv);
        }
    };

    template <class Tag, bool STRICT = false, class... Ts>
    constexpr typename Tag::storage_type getKeywordInputDefault(typename Tag::storage_type dv, Ts&&... args)
    {
        return GetKeywordInputDefault<Tag, ct::VariadicTypedef<ct::decay_t<Ts>...>, STRICT>::get(
            std::move(dv), std::forward<Ts>(args)...);
    }

    /*template <class Tag, class... Ts>
    constexpr typename Tag::storage_type getKeywordInputDefault(typename Tag::storage_type dv,
                                                                const ct::decay_t<typename Tag::storage_type>& val,
                                                                Ts&&... args)
    {
        return val;
    }*/

    /*template <class Tag, class... Ts>
    constexpr typename Tag::storage_type getKeywordInputDefault(typename Tag::storage_type, Tag arg, Ts&&...)
    {
        return arg.get();
    }*/
} // namespace mo
