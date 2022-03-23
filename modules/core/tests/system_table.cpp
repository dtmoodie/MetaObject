#include <MetaObject/core.hpp>
#include <gtest/gtest.h>

TEST(system_table, singleton)
{
    auto sys_table = SystemTable::instance();
    ASSERT_NE(sys_table, nullptr);
    ASSERT_EQ(sys_table, SystemTable::instance());
}
