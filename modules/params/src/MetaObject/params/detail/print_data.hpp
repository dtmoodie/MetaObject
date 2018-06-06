#pragma once

#include "ct/reflect/printer.hpp"
#include "ct/reflect/reflect_data.hpp"
#include <ct/detail/TypeTraits.hpp>
#include <MetaObject/core/Demangle.hpp>

namespace mo
{
    namespace detail
    {
        template <class T>
        struct stream_serializable {
            template <class U>
            static constexpr auto check(std::ostream* os, U* val) -> decltype(*os << *val, size_t(0))
            {
                return 0;
            }

            template <class U>
            static constexpr int check(...)
            {
                return 0;
            }
            static const bool value = sizeof(check<T>(static_cast<std::ostream*>(nullptr), static_cast<T*>(nullptr))) == sizeof(size_t);
        };
    } // detail::

    template<class T>
    auto printImpl(std::ostream& os, const T& data) -> typename std::enable_if<detail::stream_serializable<T>::value>::type
    {
        os << data;
    }

    template<class T>
    auto printImpl(std::ostream& os, const T& data) -> typename std::enable_if<!detail::stream_serializable<T>::value>::type
    {
        os << "No stringifier available for " << mo::Demangle::typeToName(mo::TypeInfo(typeid(T))) << std::endl;
    }

    template<class T>
    void print(std::ostream& os, const T& data)
    {
        printImpl(os, data);
    }
}
