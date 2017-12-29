#pragma once
#include "MetaObject/params/reflect_data.hpp"
#include <cereal/cereal.hpp>

namespace mo
{
    namespace reflect
    {

        template<class T, class Enable = void>
        struct ReflectData;

        template<int I, class T>
        static constexpr inline auto get(T& data) -> decltype(ReflectData<std::remove_const_t<T>>::get(data, mo::_counter_<I>()));

        template<int I, class T>
        static constexpr inline auto get(const T& data) -> decltype(ReflectData<std::remove_const_t<T>>::get(data, mo::_counter_<I>()));

        template<int I, class T>
        static constexpr inline auto getValue(const T& data){return get<I, T>(data);}

        template<int I, class T>
        static constexpr inline void setValue(T& data, const decltype(get<I, T>(data))& value) { get<I, T>(data) = value; }

        template<int I, class T>
        static constexpr inline const char* getName();

        namespace detail
        {
            template<class T>
            static constexpr inline void printStructHelper(std::ostream& os, const T& data, mo::_counter_<0>)
            {
                os << getName<0, T>() << ':' << get<0>(data);
            }

            template<int I, class T>
            static constexpr inline void printStructHelper(std::ostream& os, const T& data, mo::_counter_<I>)
            {
                printStructHelper(os, data, mo::_counter_<I-1>());
                os << ' ' << getName<I, T>() << ':' << get<I>(data);
            }

            template<class AR, class T>
            static constexpr inline void serializeHelper(AR& ar, T& data, mo::_counter_<0>)
            {
                ar(cereal::make_nvp(getName<0, T>(), get<0>(data)));
            }

            template<int I, class AR, class T>
            static constexpr inline void serializeHelper(AR& ar, T& data, mo::_counter_<I>)
            {
                ar(cereal::make_nvp(getName<I, T>(), get<I>(data)));
                serializeHelper(ar, data, mo::_counter_<I-1>());
            }
        }
    }
}
