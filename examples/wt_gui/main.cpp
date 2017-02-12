#include "MetaObject/Parameters/IParameter.hpp"

#include <iostream>
using namespace mo;



template<class T>
void testSet(T&& ts)
{
    mo::time_t time(ts);
    std::cout << time << std::endl;
}

int main()
{
    mo::time_t time(100 * mo::milli * mo::second);
    std::cout << time << std::endl;
    time = mo::time_t(15 * mo::milli * mo::second);
    std::cout << time << std::endl;
    testSet(100 * mo::milli * mo::second);
    return 0;
}
