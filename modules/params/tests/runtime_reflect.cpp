#include <MetaObject/runtime_reflection.hpp>
#include <MetaObject/params.hpp>

#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


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

BOOST_AUTO_TEST_CASE(Reflect)
{
    ReflectTester tester;
    testTypes(tester);
}

BOOST_AUTO_TEST_CASE(Serialize)
{

}
