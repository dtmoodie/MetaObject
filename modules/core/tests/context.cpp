#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/core/Context.hpp>

using namespace mo;

namespace
{
    struct Fixture
    {
        std::shared_ptr<Context> ctx;

        Fixture()
        {
            ctx = Context::create();
        }

        ~Fixture()
        {
        }

        void testInit()
        {
        }
    };
}

BOOST_FIXTURE_TEST_CASE(init, Fixture)
{
    testInit();
}
