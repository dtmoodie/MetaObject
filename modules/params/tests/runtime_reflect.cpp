#include <MetaObject/params.hpp>
#include <MetaObject/runtime_reflection.hpp>

#include <gtest/gtest.h>

#include "../../runtime_reflection/tests/common.hpp"

using namespace mo;

template <class T>
void reflectImpl(StaticVisitor& visitor)
{
    TParam<T> param;
    param.visit(visitor);
}

template <class T>
void reflect()
{
    std::stringstream ss;
    PrintVisitor printer(ss);
    reflectImpl<T>(printer);
    std::string str = ss.str();
    // TODO expected string output validation
}

namespace
{
    struct ReflectTester
    {
        template <class T>
        void test(const T&)
        {
            reflect<T>();
        }
    };

    struct SerializeTester
    {
        template <class T>
        void test(const T& data)
        {
            TParam<T> param;
            param.updateData(data);
        }
    };
} // namespace

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
