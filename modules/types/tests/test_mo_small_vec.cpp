#include <MetaObject/types/small_vec.hpp>

int main()
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
        std::cout << test_vec << std::endl;
    }
    {
        std::vector<float> test(20, 5);
        test_vec.assign(&test[0], &test[0] + 20);
        std::cout << test_vec << std::endl;
    }

    {
        std::vector<float> test(4, 4);
        test_vec = test;
        std::cout << test_vec << std::endl;
    }

}