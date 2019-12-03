#include <MetaObject/core.hpp>
#include <MetaObject/core/detail/MemoryBlock.hpp>
#include <MetaObject/core/detail/allocator_policies/Default.hpp>
#include <MetaObject/params.hpp>

#include <gtest/gtest.h>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto table = SystemTable::instance();
    PerModuleInterface::GetInstance()->SetSystemTable(table.get());
    mo::params::init(table.get());
    auto allocator = std::make_shared<mo::DefaultAllocator<mo::CPU>>();
    table->setDefaultAllocator(allocator);
    auto result = RUN_ALL_TESTS();
    return result;
}

TEST(params, initialization)
{
    auto table = SystemTable::instance();
    ASSERT_NE(table, nullptr);
}
