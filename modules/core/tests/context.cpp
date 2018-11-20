#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/core/AsyncStream.hpp>

using namespace mo;

namespace
{

    struct Fixture
    {
        std::shared_ptr<IAsyncStream> stream;

        Fixture()
        {
            stream = AsyncStreamFactory::instance()->create();
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
    BOOST_REQUIRE(stream);
}
