#include "gtest/gtest.h"

#include <MetaObject/core/detail/Time.hpp>
using namespace mo;

TEST(time, comparison_whole)
{
    Time t0(0 * ms);
    Time t1(1 * ms);
    Time t1_(1 * ms);

    ASSERT_EQ(t1, t1_);
    ASSERT_GT(t1, t0);
    ASSERT_LT(t0, t1);
}

TEST(time, comparison_fration)
{
    Time t0(0.9 * ms);
    Time t1(1.1 * ms);
    ASSERT_GT(t1, t0);
    ASSERT_LT(t0, t1);
}

TEST(time, prefixes)
{
    ASSERT_EQ((100 * ns).count(), 100);

    ASSERT_EQ((100 * us).count(), 100000);

    ASSERT_EQ((100 * ms).count(), 100000000);

    ASSERT_EQ(1 * second, 1000 * ms);

    ASSERT_EQ(1 * minutes, 60 * second);
}

TEST(time, fractional_prefixes)
{
    ASSERT_EQ((100 * ns).count(), 100);

    ASSERT_EQ((1.5 * us).count(), 1500);

    ASSERT_EQ((1.5 * ms).count(), 1500000);

    ASSERT_EQ(1.5 * second, 1500 * ms);

    ASSERT_EQ(1.5 * minutes, 90 * second);
}

TEST(time, delta)
{
    Time t0(0.9 * ms);
    Time t1(1.1 * ms);
    auto delta = t1 - t0;
    ASSERT_EQ(delta, 200 * us);

    Time t2(0.8 * ms);
    auto delta2 = t1 - t2;
    ASSERT_GE(delta2, delta);

    delta = t0 - t1;
    ASSERT_EQ(delta, -200 * us);
}
