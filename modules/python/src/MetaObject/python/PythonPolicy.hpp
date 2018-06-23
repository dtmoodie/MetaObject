#pragma once
#include "DataConverter.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "PythonSetup.hpp"
#include "converters.hpp"
#include <utility>

namespace mo
{

#define MO_METAPARAM_PYTHON_(N)                                                                                        \
    template <class T>                                                                                                 \
    struct MetaParam<T, N> : public MetaParam<T, N - 1, void>                                                          \
    {                                                                                                                  \
        MetaParam<T, N>(SystemTable * table, const char* name) : MetaParam<T, N - 1>(table, name)                      \
        {                                                                                                              \
            python::ParamConverter<T>();                                                                               \
            python::ToPythonDataConverter<T, void>(table, name);                                                       \
        }                                                                                                              \
    };

    MO_METAPARAM_PYTHON_(__COUNTER__);
}
