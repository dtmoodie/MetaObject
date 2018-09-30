#pragma once
#include <MetaObject/core/Demangle.hpp>
#include <MetaObject/core/metaobject_config.hpp>

#include "ct/reflect/print.hpp"
#include "ct/reflect.hpp"
#include <ct/TypeTraits.hpp>

namespace mo
{
    template <class T>
    auto printImpl(std::ostream& os, const T& data) ->
        typename std::enable_if<ct::StreamWritable<T>::value>::type
    {
        os << data;
    }

    template <class T>
    auto printImpl(std::ostream& os, const T & /*data*/) ->
        typename std::enable_if<!ct::StreamWritable<T>::value>::type
    {
        os << "No stringifier available for " << mo::Demangle::typeToName(mo::TypeInfo(typeid(T))) << std::endl;
    }

    template <class T>
    void print(std::ostream& os, const T& data)
    {
        printImpl(os, data);
    }
}
