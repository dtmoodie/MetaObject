#define BOOST_TEST_MAIN

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "core"
#include <boost/test/included/unit_test.hpp>
#endif
