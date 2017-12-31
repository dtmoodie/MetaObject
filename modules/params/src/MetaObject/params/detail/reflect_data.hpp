#pragma once
#include "MetaObject/params/reflect_data.hpp"
#include <cereal/cereal.hpp>
#include <type_traits>
namespace mo
{
    namespace reflect
    {

        template <class T, class Enable = void>
        struct ReflectData;

        template <int I, class T>
        static constexpr inline auto get(T& data)
            -> decltype(ReflectData<typename std::remove_const<T>::type>::get(data, mo::_counter_<I>()));

        template <int I, class T>
        static constexpr inline auto get(const T& data)
            -> decltype(ReflectData<typename std::remove_const<T>::type>::get(data, mo::_counter_<I>()));

        template <int I, class T>
        static constexpr inline auto getValue(const T& data)
        {
            return get<I, T>(data);
        }

        template <int I, class T>
        static constexpr inline void setValue(T& data, const decltype(get<I, T>(data))& value)
        {
            get<I, T>(data) = value;
        }

        template <int I, class T>
        static constexpr inline const char* getName();

        // Specialization for internally reflected data
        template <class T>
        struct ReflectData<T, decltype(T::get(std::declval<T>(), mo::_counter_<0>()), void())>
        {
            static constexpr bool IS_SPECIALIZED = true;
            static constexpr int N = T::N;

            static constexpr auto get(const T& data, mo::_counter_<0>) -> decltype(T::get(data, mo::_counter_<0>()))
            {
                return T::get(data, mo::_counter_<0>());
            }

            static constexpr auto get(T& data, mo::_counter_<0>) -> decltype(T::get(data, mo::_counter_<0>()))
            {
                return T::get(data, mo::_counter_<0>());
            }

            template <int I>
            static constexpr auto get(const T& data, mo::_counter_<I>) -> decltype(T::get(data, mo::_counter_<I>()))
            {
                return T::get(data, mo::_counter_<I>());
            }

            template <int I>
            static constexpr auto get(T& data, mo::_counter_<I>) -> decltype(T::get(data, mo::_counter_<I>()))
            {
                return T::get(data, mo::_counter_<I>());
            }

            static constexpr const char* getName(mo::_counter_<0> cnt) { return T::getName(cnt); }

            template <int I>
            static constexpr const char* getName(mo::_counter_<I> cnt)
            {
                return T::getName(cnt);
            }
        };

        namespace detail
        {
            template <class T>
            static constexpr inline void printStructHelper(std::ostream& os, const T& data, mo::_counter_<0>)
            {
                os << getName<0, T>() << ':' << get<0>(data);
            }

            template <int I, class T>
            static constexpr inline void printStructHelper(std::ostream& os, const T& data, mo::_counter_<I>)
            {
                printStructHelper(os, data, mo::_counter_<I - 1>());
                os << ' ' << getName<I, T>() << ':' << get<I>(data);
            }

            template <class AR, class T>
            static constexpr inline void serializeHelper(AR& ar, T& data, mo::_counter_<0>)
            {
                ar(cereal::make_nvp(getName<0, T>(), get<0>(data)));
            }

            template <int I, class AR, class T>
            static constexpr inline void serializeHelper(AR& ar, T& data, mo::_counter_<I>)
            {
                serializeHelper(ar, data, mo::_counter_<I - 1>());
                ar(cereal::make_nvp(getName<I, T>(), get<I>(data)));
            }
        }
    }
}
