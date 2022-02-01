#include <MetaObject/core.hpp>
#include <MetaObject/cuda.hpp>

#include <gtest/gtest.h>


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto table = SystemTable::instance();
    mo::cuda::init(table.get());
    auto result = RUN_ALL_TESTS();
    table.reset();
    return result;
}
