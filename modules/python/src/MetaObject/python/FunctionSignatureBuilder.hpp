#pragma once

#include <ct/VariadicTypedef.hpp>

namespace mo
{
    template <class T, int N>
    struct FunctionSignatureBuilder
    {
        typedef typename ct::Append<T, typename FunctionSignatureBuilder<T, N - 1>::VariadicTypedef_t>::type
            VariadicTypedef_t;
    };

    template <class T>
    struct FunctionSignatureBuilder<T, 0>
    {
        typedef ct::VariadicTypedef<T> VariadicTypedef_t;
    };
}
