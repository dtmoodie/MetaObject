#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/core/detail/Time.hpp>
using namespace mo;

BOOST_AUTO_TEST_CASE(TimeComparableWhole)
{
    Time t0(0 * ms);
    Time t1(1 * ms);
    Time t1_(1 * ms);

    BOOST_REQUIRE_EQUAL(t1, t1_);
    BOOST_REQUIRE_GT(t1, t0);
    BOOST_REQUIRE_LT(t0, t1);
}

BOOST_AUTO_TEST_CASE(TimeComparableFraction)
{
    Time t0(0.9 * ms);
    Time t1(1.1 * ms);
    BOOST_REQUIRE_GT(t1, t0);
    BOOST_REQUIRE_LT(t0, t1);
}

BOOST_AUTO_TEST_CASE(TimeDelta)
{
    Time t0(0.9 * ms);
    Time t1(1.1 * ms);
    auto delta = t1 - t0;
    BOOST_REQUIRE_EQUAL(delta, 200 * us);

    Time t2(0.8 * ms);
    auto delta2 = t1 - t2;
    BOOST_REQUIRE_GE(delta2, delta);
}
