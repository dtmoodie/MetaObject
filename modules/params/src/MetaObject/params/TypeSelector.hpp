#pragma once
#include <MetaObject/core/detail/Counter.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <tuple>

namespace mo
{
    /*template <class F, class T>
    struct TypeSelector;*/

    template <class F, class... T>
    struct TypeSelector
    {
        TypeSelector(F& functor) : m_func(functor) {}

        template <class... Args>
        bool apply(const mo::TypeInfo& type, Args&&... args)
        {
            return applyImpl(_counter_<static_cast<int>(sizeof...(T)) - 1>(), type, std::forward<Args>(args)...);
        }

      private:
        template <class... Args>
        bool applyImpl(_counter_<0>, const mo::TypeInfo& type, Args&&... args)
        {
            typedef typename std::tuple_element<0, std::tuple<T...>>::type Type;
            if (mo::TypeInfo(typeid(Type)) == type)
            {
                m_func.template apply<Type>(std::forward<Args>(args)...);
                return true;
            }
            return false;
        }

        template <int I, class... Args>
        bool
        applyImpl(_counter_<I> cnt, const typename std::enable_if<I != 0, mo::TypeInfo>::type& type, Args&&... args)
        {
            typedef typename std::tuple_element<I, std::tuple<T...>>::type Type;
            if (mo::TypeInfo(typeid(Type)) == type)
            {
                m_func.template apply<Type>(std::forward<Args>(args)...);
                return true;
            }
            else
            {
                return applyImpl(--cnt, type, std::forward<Args>(args)...);
            }
        }

        F& m_func;
    };

    template <class F, class... T>
    struct TypeSelector<F, std::tuple<T...>>
    {
        TypeSelector(F& functor) : m_func(functor) {}

        template <class... Args>
        bool apply(const mo::TypeInfo& type, Args&&... args)
        {
            return applyImpl(_counter_<static_cast<int>(sizeof...(T)) - 1>(), type, std::forward<Args>(args)...);
        }

      private:
        template <class... Args>
        bool applyImpl(_counter_<0>, const mo::TypeInfo& type, Args&&... args)
        {
            typedef typename std::tuple_element<0, std::tuple<T...>>::type Type;
            if (mo::TypeInfo(typeid(Type)) == type)
            {
                m_func.template apply<Type>(std::forward<Args>(args)...);
                return true;
            }
            return false;
        }

        template <int I, class... Args>
        bool
        applyImpl(_counter_<I> cnt, const typename std::enable_if<I != 0, mo::TypeInfo>::type& type, Args&&... args)
        {
            typedef typename std::tuple_element<I, std::tuple<T...>>::type Type;
            if (mo::TypeInfo(typeid(Type)) == type)
            {
                m_func.template apply<Type>(std::forward<Args>(args)...);
                return true;
            }
            else
            {
                return applyImpl(--cnt, type, std::forward<Args>(args)...);
            }
        }

        F& m_func;
    };

    template <class F, class... T>
    struct TypeLoop
    {
        TypeLoop(F& func) : m_func(func) {}
        template <class... Args>
        void apply(Args&&... args)
        {
            applyImpl(std::integral_constant<int, sizeof...(T)-1>(), std::forward<Args>(args)...);
        }

      private:
        template <class... Args>
        void applyImpl(std::integral_constant<int, 0>, Args&&... args)
        {
            typedef typename std::tuple_element<0, std::tuple<T...>>::type Type;
            m_func.template apply<Type>(std::forward<Args>(args)...);
        }
        template <int I, class... Args>
        void applyImpl(std::integral_constant<int, I>, Args&&... args)
        {
            typedef typename std::tuple_element<I, std::tuple<T...>>::type Type;
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
}
