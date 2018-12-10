#define BOOST_TEST_MAIN

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "cuda"
#include <boost/test/included/unit_test.hpp>
#endif

#include <MetaObject/core.hpp>

struct GlobalFixture
{
    GlobalFixture()
        : m_system_table(SystemTable::instance())
    {
    }

    ~GlobalFixture()
    {
        m_system_table.reset();
    }

    SystemTable::Ptr_t m_system_table;
};

BOOST_GLOBAL_FIXTURE(GlobalFixture);
