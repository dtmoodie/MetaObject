#pragma once
#define MO_HAVE_PYTHON2 1
#include "DataConverter.hpp"
#include "PythonSetup.hpp"
#include "converters.hpp"

#include <MetaObject/params/MetaParam.hpp>
#include <MetaObject/python/DataConverter.hpp>
#include <MetaObject/python/converters.hpp>

#include <ct/interop/boost_python/PythonConverter.hpp>
#include <utility>

namespace mo
{

    template <class T>
    void registerPythonConverters(SystemTable* table, const char* name)
    {
        python::ParamConverter<T> param_converter(table);
        ct::registerToPython<T>();
    }

#define MO_METAPARAM_PYTHON_(N)                                                                                        \
    template <class T>                                                                                                 \
    struct MetaParam<T, N> : public MetaParam<T, N - 1, void>                                                          \
    {                                                                                                                  \
        MetaParam<T, N>(SystemTable * table, const char* name)                                                         \
            : MetaParam<T, N - 1>(table, name)                                                                         \
        {                                                                                                              \
            registerPythonConverters<T>(table, name);                                                                  \
        }                                                                                                              \
    };

    MO_METAPARAM_PYTHON_(__COUNTER__);
} // namespace mo
