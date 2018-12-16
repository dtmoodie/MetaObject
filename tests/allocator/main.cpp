#define BOOST_TEST_MAIN
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/core/detail/StlAllocator.hpp>
#include <MetaObject/core/detail/Time.hpp>

#include <MetaObject/core/detail/allocator_policies/Combined.hpp>
#include <MetaObject/core/detail/allocator_policies/Continuous.hpp>
#include <MetaObject/core/detail/allocator_policies/Lock.hpp>
#include <MetaObject/core/detail/allocator_policies/Pool.hpp>
#include <MetaObject/core/detail/allocator_policies/Stack.hpp>
#include <MetaObject/core/detail/allocator_policies/opencv.hpp>

#include <MetaObject/cuda/CvAllocator.hpp>
#include <MetaObject/cuda/MemoryBlock.hpp>

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/profiling.hpp>

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
    // TODO fix unit test
    // BOOST_REQUIRE(mo::Allocator::getDefaultAllocator());
    mo::initLogging();
    mo::initProfiling();
}

BOOST_AUTO_TEST_CASE(initialize_global)
{
    // TODO fix unit test
    // BOOST_REQUIRE(mo::Allocator::getDefaultAllocator());
}

BOOST_AUTO_TEST_CASE(test_cpu_set_allocator)
{
}

BOOST_AUTO_TEST_CASE(test_gpu_set_allocator)
{
}
