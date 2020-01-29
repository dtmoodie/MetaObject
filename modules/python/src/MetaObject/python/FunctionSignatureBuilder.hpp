#pragma once

#include <ct/VariadicTypedef.hpp>

namespace mo
{
    template <class T, int N>
    struct FunctionSignatureBuilder
    {
        using VariadicTypedef_t = typename ct::Append<T, typename FunctionSignatureBuilder<T, N - 1>::VariadicTypedef_t>::type;
    };

    template <class T>
    struct FunctionSignatureBuilder<T, 0>
    {
        using VariadicTypedef_t = ct::VariadicTypedef<T>;
    };

    template <class T>
    struct FunctionSignatureBuilder<T, -1>
    {
        using VariadicTypedef_t = ct::VariadicTypedef<void> ;
    };
}
