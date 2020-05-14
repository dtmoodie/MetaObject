#pragma once
#include <MetaObject/detail/TypeInfo.hpp>
#include <ct/Indexer.hpp>
#include <tuple>

namespace mo
{
    template <class F, class... T>
    struct TypeSelector
    {
        TypeSelector(F& functor)
            : m_func(functor)
        {
        }

        template <class... Args>
        bool apply(const mo::TypeInfo& type, Args&&... args)
        {
            return applyImpl(ct::Indexer<static_cast<int>(sizeof...(T)) - 1>{}, type, std::forward<Args>(args)...);
        }

      private:
        template <class... Args>
        bool applyImpl(const ct::Indexer<0>, const mo::TypeInfo& type, Args&&... args)
        {
            using Type = typename std::tuple_element<0, std::tuple<T...>>::type;
            if (mo::TypeInfo(typeid(Type)) == type)
            {
                m_func.template apply<Type>(std::forward<Args>(args)...);
                return true;
            }
            return false;
        }

        template <int I, class... Args>
        bool applyImpl(const ct::Indexer<I> cnt,
                       const typename std::enable_if<I != 0, mo::TypeInfo>::type& type,
                       Args&&... args)
        {
            using Type = typename std::tuple_element<I, std::tuple<T...>>::type;
            if (mo::TypeInfo(typeid(Type)) == type)
            {
                m_func.template apply<Type>(std::forward<Args>(args)...);
                return true;
            }
            return applyImpl(--cnt, type, std::forward<Args>(args)...);
        }

        F& m_func;
    };

    template <class F, class... T>
    struct TypeSelector<F, std::tuple<T...>>
    {
        TypeSelector(F& functor)
            : m_func(functor)
        {
        }

        template <class... Args>
        bool apply(const mo::TypeInfo& type, Args&&... args)
        {
            const ct::Indexer<static_cast<ct::index_t>(sizeof...(T)) - 1> start_index;
            return applyImpl(start_index, type, std::forward<Args>(args)...);
        }

      private:
        template <class... Args>
        bool applyImpl(ct::Indexer<0>, const mo::TypeInfo& type, Args&&... args)
        {
            using Type = typename std::tuple_element<0, std::tuple<T...>>::type;
            if (type.template isType<Type>())
            {
                m_func.template apply<Type>(std::forward<Args>(args)...);
                return true;
            }
            return false;
        }

        template <ct::index_t I, class... Args>
        bool applyImpl(ct::Indexer<I> index, const ct::EnableIf<I != 0, mo::TypeInfo>& type, Args&&... args)
        {
            using Type = typename std::tuple_element<I, std::tuple<T...>>::type;
            if (type.template isType<Type>())
            {
                m_func.template apply<Type>(std::forward<Args>(args)...);
                return true;
            }
            const auto next_index = --index;
            return applyImpl(next_index, type, std::forward<Args>(args)...);
        }

        F& m_func;
    };

    template <class F, class... T>
    struct TypeLoop
    {
        TypeLoop(F& func)
            : m_func(func)
        {
        }
        template <class... Args>
        void apply(Args&&... args)
        {
            applyImpl(std::integral_constant<int, sizeof...(T) - 1>(), std::forward<Args>(args)...);
        }

      private:
        template <class... Args>
        void applyImpl(std::integral_constant<int, 0>, Args&&... args)
        {
            using Type = typename std::tuple_element<0, std::tuple<T...>>::type;
            m_func.template apply<Type>(std::forward<Args>(args)...);
        }
        template <int I, class... Args>
        void applyImpl(std::integral_constant<int, I>, Args&&... args)
        {
            using Type = typename std::tuple_element<I, std::tuple<T...>>::type;
            m_func.template apply<Type>(std::forward<Args>(args)...);
            applyImpl(std::integral_constant<int, I - 1>(), std::forward<Args>(args)...);
        }
        F& m_func;
    };

    template <class... Types, class F, class... Args>
    bool selectType(F& func, const mo::TypeInfo& type, Args&&... args)
    {
        static_assert(sizeof...(Types) > 0, "Must have multiple type inputs");
        TypeSelector<F, Types...> selector(func);
        return selector.apply(type, std::forward<Args>(args)...);
    }

    template <class... Types, class F, class... Args>
    void typeLoop(F& func, Args&&... args)
    {
        TypeLoop<F, Types...> loop(func);
        loop.apply(std::forward<Args>(args)...);
    }
} // namespace mo
