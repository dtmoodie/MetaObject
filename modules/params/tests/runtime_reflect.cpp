#include <MetaObject/runtime_reflection.hpp>
#include <MetaObject/params.hpp>

#include <gtest/gtest.h>

#include "../../runtime_reflection/tests/common.hpp"

using namespace mo;

template<class T>
void reflectImpl(StaticVisitor& visitor)
{
    TParam<T> param;
    param.visit(visitor);
}

template<class T>
void reflect()
{
    PrintVisitor printer;
    reflectImpl<T>(printer);
}

namespace
{
    struct ReflectTester
    {
        template<class T>
        void test(const T&)
        {
            std::cout << "-------------------------------" << std::endl;
            reflect<T>();
        }
    };

    struct SerializeTester
    {
        template<class T>
        void test(const T& data)
        {
            TParam<T> param;
            param.updateData(data);

        }
    };
}


void acceptsPtr(const uint32_t* ptr)
{

}

void acceptsPtr(const int64_t* ptr)
{

}

TEST(runtime_reflection, reflect)
{
    ReflectTester tester;
    testTypes(tester);
}

TEST(runtime_reflection, serialize)
{

}
