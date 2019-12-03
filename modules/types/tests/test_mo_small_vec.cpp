#include <MetaObject/types/small_vec.hpp>
#include <gtest/gtest.h>

TEST(types, small_vec_small)
{
    mo::SmallVec<float, 5> test_vec;
    std::cout << test_vec << std::endl;
    {
        float test[4];
        test[0] = 0;
        test[1] = 1;
        test[2] = 2;
        test[3] = 3;

        test_vec.assign(test, test + 4);
        EXPECT_EQ(test_vec[0], 0);
        EXPECT_EQ(test_vec[1], 1);
        EXPECT_EQ(test_vec[2], 2);
        EXPECT_EQ(test_vec[3], 3);
    }
}

TEST(types, small_vec_large)
{
    mo::SmallVec<float, 5> test_vec;
    std::vector<float> test(20, 5);
    test_vec.assign(&test[0], &test[0] + 20);
    for (size_t i = 0; i < 20; ++i)
    {
        EXPECT_EQ(test_vec[i], 5);
    }
}

TEST(types, small_vec_wrap_vector)
{
    mo::SmallVec<float, 5> test_vec;
    std::vector<float> test(4, 4);
    test_vec = test;
    EXPECT_EQ(test_vec[0], 4);
    EXPECT_EQ(test_vec[1], 4);
    EXPECT_EQ(test_vec[2], 4);
    EXPECT_EQ(test_vec[3], 4);
}