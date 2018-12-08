#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params.hpp>
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
        , m_factory(m_system_table.get())
    {
        mo::params::init(m_system_table.get());
        m_factory.registerTranslationUnit();
        auto module = PerModuleInterface::GetInstance();
        module->SetSystemTable(m_system_table.get());
        std::shared_ptr<mo::ThreadPool> pool = mo::sharedSingleton<mo::ThreadPool>(m_system_table.get());
        boost::fibers::use_scheduling_algorithm<mo::PriorityScheduler>(pool);
    }

    ~GlobalFixture()
    {
        m_system_table.reset();
    }

    SystemTable::Ptr_t m_system_table;
    mo::MetaObjectFactory m_factory;
};

BOOST_GLOBAL_FIXTURE(GlobalFixture);
