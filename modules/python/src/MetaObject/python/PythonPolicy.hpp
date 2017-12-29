#pragma once
#include "converters.hpp"
#include "DataConverter.hpp"
#include "PythonSetup.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include <utility>

namespace mo
{

#define MO_METAPARAM_PYTHON_(N)                                           \
    template <class T>                                                    \
    struct MetaParam<T, N> : public MetaParam<T, N - 1, void> {           \
        static python::ParamConverter< T > _python_converter;             \
        static python::ToPythonDataConverter< T, void > _python_data_converter; \
        MetaParam<T, N>(const char* name)                                 \
            : MetaParam<T, N - 1>(name) {                                 \
            (void)&_python_converter;                                     \
            (void)&_python_data_converter;                                \
        }                                                                 \
    };                                                                    \
    template <class T>                                                    \
    python::ParamConverter< T > MetaParam<T, N>::_python_converter;       \
    template <class T>                                                    \
    python::ToPythonDataConverter< T, void > MetaParam<T, N>::_python_data_converter

MO_METAPARAM_PYTHON_(__COUNTER__);

}
