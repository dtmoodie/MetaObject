#pragma once
#include <MetaObject/core/Demangle.hpp>
#include <MetaObject/core/metaobject_config.hpp>

#include "ct/reflect/printer.hpp"
#include "ct/reflect/reflect_data.hpp"
#include <ct/detail/TypeTraits.hpp>

namespace mo
{
    namespace detail
    {
        template <class T>
        struct stream_serializable
        {

            template <typename SS, typename TT>
            static auto test(int) -> decltype(std::declval<SS&>() << std::declval<TT>(), std::true_type());

            template <typename, typename>
            static auto test(...) -> std::false_type;

          public:
            static const bool value = decltype(test<std::ostream, T>(0))::value;
        };
    } // detail::

    template <class T>
    auto printImpl(std::ostream& os, const T& data) ->
        typename std::enable_if<detail::stream_serializable<T>::value>::type
    {
        os << data;
    }

    template <class T>
    auto printImpl(std::ostream& os, const T & /*data*/) ->
        typename std::enable_if<!detail::stream_serializable<T>::value>::type
    {
        os << "No stringifier available for " << mo::Demangle::typeToName(mo::TypeInfo(typeid(T))) << std::endl;
    }

    template <class T>
    void print(std::ostream& os, const T& data)
    {
//        printImpl(os, data);
    }
}
