#pragma once
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/params/detail/reflect_data.hpp"
#include <stdint.h>
#include <ostream>

namespace mo
{
    namespace reflect
    {
        template<class T>
        struct ReflectData
        {
            static constexpr bool IS_SPECIALIZED = false;
        };

        template<int I, class T>
        static constexpr inline auto& get(T& data)
        {
            return ReflectData<T>::get(data, mo::_counter_<I>());
        }

        template<int I, class T>
        static constexpr inline const auto& get(const T& data)
        {
            return ReflectData<T>::get(data, mo::_counter_<I>());
        }

        template<int I, class T>
        static constexpr inline const char* getName()
        {
            return ReflectData<T>::getName(mo::_counter_<I>());
        }

        template<class T>
        constexpr int len()
        {
            return ReflectData<T>::N;
        }

        template<class T, class T2 = void>
        using enable_if_reflected = typename std::enable_if<ReflectData<T>::IS_SPECIALIZED, T2>::type;

        template<class T, class T2 = void>
        using enable_if_not_reflected = typename std::enable_if<!ReflectData<T>::IS_SPECIALIZED, T2>::type;

        template<class T>
        static constexpr inline void printStruct(std::ostream& os, const T& data)
        {
            detail::printStructHelper(os, data, mo::_counter_<ReflectData<T>::N - 1>());
        }

        template<class AR, class T>
        enable_if_reflected<T> serialize(AR& ar, T& data)
        {
            detail::serializeHelper(ar, data, mo::_counter_<ReflectData<T>::N - 1>());
        }
    }

}

#define REFLECT_DATA_START(TYPE) \
template<> \
struct ReflectData<TYPE> \
{ \
    static constexpr int START = __COUNTER__; \
    static constexpr bool IS_SPECIALIZED = true; \
    typedef TYPE DType

#define REFLECT_DATA_MEMBER(NAME) REFLECT_DATA_MEMBER_(NAME, __COUNTER__)

#define REFLECT_DATA_MEMBER_(NAME, N) \
static constexpr auto&       get(      DType& data, mo::_counter_<N - START - 1>) { return data.NAME; } \
static constexpr const auto& get(const DType& data, mo::_counter_<N - START - 1>) { return data.NAME; } \
static constexpr const char* getName(mo::_counter_<N- START - 1> /*dummy*/){ return #NAME; }

#define REFLECT_DATA_END() static constexpr int N = __COUNTER__ - START - 1; };

#define REFLECT_TEMPLATED_DATA_START(TYPE) \
template<class... T> \
struct ReflectData<TYPE<T...>> \
{ \
    static constexpr int START = __COUNTER__; \
    static constexpr bool IS_SPECIALIZED = true; \
    typedef TYPE<T...> DType

