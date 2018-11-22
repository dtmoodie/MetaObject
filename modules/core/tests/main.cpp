#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/thread/FiberScheduler.hpp>
#define BOOST_TEST_MAIN

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "core"
#include <boost/test/included/unit_test.hpp>
#endif

struct GlobalFixture
{
    GlobalFixture()
        : m_system_table(SystemTable::instance())
    {
        boost::fibers::use_scheduling_algorithm<mo::PriorityScheduler>();
    }

    SystemTable::Ptr_t m_system_table;
};

BOOST_GLOBAL_FIXTURE(GlobalFixture);
