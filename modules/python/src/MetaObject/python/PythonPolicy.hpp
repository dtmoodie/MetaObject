#pragma once
#define MO_HAVE_PYTHON 1
#include "DataConverter.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "PythonSetup.hpp"
#include "converters.hpp"
#include <utility>

namespace mo
{

template<class T>
void registerPythonConverters(SystemTable* table, const char* name)
{
    python::ParamConverter<T> param_converter(table);
    python::ToPythonDataConverter<T, void> data_converter(table, name);
}

#define MO_METAPARAM_PYTHON_(N)                                                                                        \
    template <class T>                                                                                                 \
    struct MetaParam<T, N> : public MetaParam<T, N - 1, void>                                                          \
    {                                                                                                                  \
        MetaParam<T, N>(SystemTable * table, const char* name) : MetaParam<T, N - 1>(table, name)                      \
        {                                                                                                              \
            registerPythonConverters<T>(table, name);                                                                  \
        }                                                                                                              \
    };

    MO_METAPARAM_PYTHON_(__COUNTER__);
}
