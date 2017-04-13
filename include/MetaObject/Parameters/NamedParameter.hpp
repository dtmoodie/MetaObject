#pragma once
#include <boost/optional.hpp>
#define MO_KEYWORD_INPUT(name, type) \
namespace tag \
{ \
    struct name \
    { \
        typedef type Type; \
        typedef const Type& ConstRef; \
        typedef Type& Ref; \
        typedef ConstRef StorageType; \
        typedef const void* VoidType; \
    }; \
    static mo::kwargs::TypedKeyword<name>& _##name = \
            mo::kwargs::TypedKeyword<name>::instance; \
}

#define MO_KEYWORD_OUTPUT(name, type) \
namespace tag \
{ \
    struct name \
    { \
        typedef type Type; \
        typedef const Type& ConstRef; \
        typedef Type& Ref; \
        typedef Ref StorageType; \
        typedef void* VoidType; \
    }; \
    static mo::kwargs::TypedKeyword<name>& _##name = \
            mo::kwargs::TypedKeyword<name>::instance; \
}
template <int V1, int V2> struct AssertEquality
{
    static const char not_equal_warning = V1 + V2 + 256;
};

template <int V> struct AssertEquality<V, V>
{
    static const bool not_equal_warning = 0;
};

#define ASSERT_EQUALITY(V1, V2) static_assert( \
    AssertEquality<static_cast<int>(V1), \
                   static_cast<int>(V2)>::not_equal_warning == 0, \
    #V1 " != " #V2 );

namespace mo
{
    namespace kwargs
    {
        struct TaggedBase{};
        template<class Tag>
        struct TaggedArgument: public TaggedBase
        {
            typedef Tag TagType;
            explicit TaggedArgument(typename Tag::StorageType val):
                arg(&val)
            {

            }
            explicit TaggedArgument(const boost::optional<typename Tag::Type>& val)
            {
                if(val)
                {
                    arg = &(*val);
                }else
                {
                    arg = nullptr;
                }
            }

            typename Tag::VoidType get() const
            {
                return arg;
            }
        protected:
            typename Tag::VoidType arg;
        };

        template<class Tag>
        struct TypedKeyword
        {
            static TypedKeyword instance;
            TaggedArgument<Tag> operator=(typename Tag::StorageType data)
            {
                return TaggedArgument<Tag>(data);
            }
            TaggedArgument<Tag> operator=(const boost::optional<typename Tag::Type>& data)
            {
                return TaggedArgument<Tag>(data);
            }
        };
        template<class T> TypedKeyword<T> TypedKeyword<T>::instance;
    }

    template <int N, typename... T>
    struct ArgType;

    template <typename T0, typename... T>
    struct ArgType<0, T0, T...> {
        typedef T0 type;
    };
    template <int N, typename T0, typename... T>
    struct ArgType<N, T0, T...> {
        typedef typename ArgType<N-1, T...>::type type;
    };

    template<class Tag>
    typename Tag::VoidType GetKeyImpl()
    {
        return 0;
    }

    template<class T, class U>
    constexpr int CountTypeImpl(const U& value)
    {
        return std::is_same<T, U>::value ? 1 : 0;
    }

    template<class T, class U, class...Args>
    constexpr int CountTypeImpl(const U& value, const Args&... args)
    {
        return CountTypeImpl<T, Args...>(args...) + (std::is_same<T, U>::value ? 1 : 0);
    }

    template<class T, class ...Args>
    constexpr int CountType(const Args&... args)
    {
        return CountTypeImpl<T, Args...>(args...);
    }

    template <size_t N, typename... Args>
    auto GetPositionalInput(Args&&... as) noexcept ->decltype(std::get<N>(std::forward_as_tuple(std::forward<Args>(as)...)))
    {
        return std::get<N>(std::forward_as_tuple(std::forward<Args>(as)...));
    }

    template<class Tag, class T, class... Args>
    typename std::enable_if<std::is_base_of<kwargs::TaggedBase, T>::value, typename Tag::VoidType>::type
    GetKeyImpl(const T& arg, const Args&... args)
    {
        return std::is_same<typename T::TagType, Tag>::value?
                    const_cast<void*>(arg.get()) :
                    const_cast<void*>(GetKeyImpl<Tag, Args...>(args...));
    }
    template<class Tag> typename Tag::VoidType GetPtr(const boost::optional<typename Tag::Type>& arg)
    {
        if(arg)
            return arg.get_ptr();
        return nullptr;
    }
    template<class Tag> typename Tag::VoidType GetPtr(const typename Tag::Type& arg)
    {
        return &arg;
    }
    template<class Tag, class T> typename Tag::VoidType GetPtr(const T& arg)
    {
        return nullptr;
    }

    template<class Tag, class T, class... Args>
    typename std::enable_if<!std::is_base_of<kwargs::TaggedBase, T>::value, typename Tag::VoidType>::type
    GetKeyImpl(const T& arg, const Args&... args)
    {
#ifdef __GNUC__
        //static_assert(CountType<typename Tag::Type>(arg, args...) <= 1, "Cannot infer type when there are multiple variadic parameters with desired type");
#endif

        return std::is_same<boost::optional<typename Tag::Type>, T>::value ||
                        std::is_same<typename Tag::Type, T>::value ? // This infers the type
                            GetPtr<Tag>(arg) :
                                const_cast<void*>(GetKeyImpl<Tag, Args...>(args...));
    }

    template<class Tag, class... Args>
    typename Tag::ConstRef GetKeywordInput(const Args&... args)
    {
        const void* ptr = GetKeyImpl<Tag>(args...);
        assert(ptr);
        return *(static_cast<const typename Tag::Type*>(ptr));
    }

    template<class Tag, class... Args>
    typename Tag::ConstRef GetKeywordInputDefault(typename Tag::ConstRef def, const Args&... args)
    {
        const void* ptr = GetKeyImpl<Tag>(args...);
        if(ptr)
            return *(const typename Tag::Type*)ptr;
        return def;
    }

    template<class Tag, class... Args>
    const typename Tag::Type* GetKeywordInputOptional(const Args&... args)
    {
        const void* ptr = GetKeyImpl<Tag>(args...);
        if(ptr)
            return (const typename Tag::Type*)ptr;
        return nullptr;
    }

    template<class Tag, class... Args>
    typename Tag::Ref GetKeywordOutput(const Args&... args)
    {
        static_assert(!std::is_const<typename Tag::VoidType>::value, "Tag type is not an output tag");
        void* ptr = GetKeyImpl<Tag>(args...);
        assert(ptr);
        return *(static_cast<typename Tag::Type*>(ptr));
    }

    template<class Tag, class... Args>
    typename Tag::Type* GetKeywordOutputOptional(const Args&... args)
    {
        static_assert(!std::is_const<typename Tag::VoidType>::value, "Tag type is not an output tag");
        void* ptr = GetKeyImpl<Tag>(args...);
        if(ptr)
            return (static_cast<typename Tag::Type*>(ptr));
        return nullptr;
    }
}
