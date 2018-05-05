#define BOOST_TEST_MAIN
#include <MetaObject/core/SystemTable.hpp>
#include <iostream>

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif

BOOST_AUTO_TEST_CASE(system_table_singleton)
{
    auto sys_table = SystemTable::instance();
    BOOST_REQUIRE(sys_table);
    BOOST_REQUIRE_EQUAL(sys_table, SystemTable::instance());
}

BOOST_AUTO_TEST_CASE(inherited_table)
{
}
