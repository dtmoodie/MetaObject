#include <MetaObject/python/FunctionSignatureBuilder.hpp>
#include <MetaObject/python/MetaObject.hpp>
#include <iostream>

template <class... T>
struct FunctionBuilder
{
};

template <class... Args>
struct FunctionBuilder<ct::variadic_typedef<Args...>>
{
    static void function(Args&&... args) {}
};

template <class... Args1, class... Args2>
struct FunctionBuilder<ct::variadic_typedef<Args1...>, ct::variadic_typedef<Args2...>>
{
    static void function(Args1&&... args1, Args2&&... args2) {}
};

int main()
{
    std::cout << typeid(mo::FunctionSignatureBuilder<float, 4>::VariadicTypedef_t).name() << std::endl;
    FunctionBuilder<ct::variadic_typedef<int, double, float>,
                    mo::FunctionSignatureBuilder<float, 4>::VariadicTypedef_t>::function(0, 0, 0, 0, 0, 0, 0, 0);
}
