#include <MetaObject/core/SystemTable.hpp>

#include <MetaObject/thread/Thread.hpp>

#include "gtest/gtest.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto system_table = SystemTable::instance();
    auto result = RUN_ALL_TESTS();
    return result;
}
