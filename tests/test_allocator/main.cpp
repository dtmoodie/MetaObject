#define BOOST_TEST_MAIN
#include "MetaObject/Detail/Allocator.hpp"

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "MetaObjectInheritance"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;

BOOST_AUTO_TEST_CASE(initialize_thread)
{
    BOOST_REQUIRE(mo::Allocator::GetThreadSpecificAllocator());
}

BOOST_AUTO_TEST_CASE(initialize_global)
{
    BOOST_REQUIRE(mo::Allocator::GetThreadSafeAllocator());
}
