#pragma once
#include <boost/python.hpp>
#include <ce/VariadicTypedef.hpp>

template<class T, int N>
struct FunctionSignatureBuilder
{
    typedef typename ce::append_to_tupple<T, typename FunctionSignatureBuilder<T, N - 1>::VariadicTypedef_t>::type VariadicTypedef_t;
};

template<class T>
struct FunctionSignatureBuilder<T, 0>
{
    typedef ce::variadic_typedef<T> VariadicTypedef_t;
};

struct IObjectConstructor;

namespace mo
{
    boost::python::object createConstructor(IObjectConstructor* ctr);
}