#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/thread/FiberScheduler.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/detail/throw_exception.hpp>
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
        auto module = PerModuleInterface::GetInstance();
        module->SetSystemTable(m_system_table.get());
        std::shared_ptr<mo::ThreadPool> pool = mo::sharedSingleton<mo::ThreadPool>(m_system_table.get());
        boost::fibers::use_scheduling_algorithm<mo::PriorityScheduler>(pool);
    }

    ~GlobalFixture()
    {
    }

    SystemTable::Ptr_t m_system_table;
};

BOOST_GLOBAL_FIXTURE(GlobalFixture);
