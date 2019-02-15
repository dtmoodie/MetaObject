#include <MetaObject/core/SystemTable.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/detail/throw_exception.hpp>
#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "params"
#include <boost/test/included/unit_test.hpp>
#endif

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

BOOST_AUTO_TEST_CASE(TestInitialization)
{
    auto table = SystemTable::instance();
    BOOST_REQUIRE(table != nullptr);


}
