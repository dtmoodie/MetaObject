#include <MetaObject/core/SystemTable.hpp>

#include "gtest/gtest.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto system_table = SystemTable::instance();
    return RUN_ALL_TESTS();
}
