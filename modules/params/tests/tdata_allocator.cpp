#include <MetaObject/params/TDataContainer.hpp>
#include <gtest/gtest.h>

TEST(param_allocator, create)
{
    std::vector<float, mo::TStlAllocator<float>> vector;
}