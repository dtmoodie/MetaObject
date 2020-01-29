#pragma once
#include <MetaObject/core/TypeTable.hpp>
#include <MetaObject/core/metaobject_config.hpp>

#include "ct/reflect.hpp"
#include "ct/reflect/print.hpp"
#include <ct/type_traits.hpp>

namespace mo
{
    template <class T>
    auto printImpl(std::ostream& os, const T& data) -> typename std::enable_if<ct::StreamWritable<T>::value>::type
    {
        os << data;
    }

    template <class T>
    auto printImpl(std::ostream& os, const T & /*data*/) -> typename std::enable_if<!ct::StreamWritable<T>::value>::type
    {
        os << "No stringifier available for " << mo::TypeTable::instance()->typeToName(mo::TypeInfo(typeid(T)))
           << std::endl;
    }

    template <class T>
    void print(std::ostream& os, const T& data)
    {
        printImpl(os, data);
    }
}
