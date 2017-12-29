#include "MetaObject/params/reflect_data.hpp"
#include <iostream>
#include <cereal/archives/json.hpp>

using namespace mo;

struct ReflectedStruct
{
    float x;
    float y;
    float z;
    int id;
};

namespace mo
{
    namespace reflect
    {
        REFLECT_DATA_START(ReflectedStruct);
            REFLECT_DATA_MEMBER(x)
            REFLECT_DATA_MEMBER(y)
            REFLECT_DATA_MEMBER(z)
            REFLECT_DATA_MEMBER(id)
        REFLECT_DATA_END();
    }
}


REFLECT_INTERNAL_START(InternallyReflected)
    REFLECT_INTERNAL_MEMBER(float, x)
    REFLECT_INTERNAL_MEMBER(float, y)
    REFLECT_INTERNAL_MEMBER(float, z)
REFLECT_INTERNAL_END();


int main(int argc, char** argv)
{
    InternallyReflected data2;
    ReflectedStruct data;
    data.x = 0;
    data.y = 1;
    data.z = 2;
    data.id = 3;
    mo::reflect::printStruct(std::cout, data);
    static_assert(std::is_same<mo::reflect::enable_if_reflected<ReflectedStruct>, void>::value, "test1");
    cereal::JSONOutputArchive ar(std::cout);
    mo::reflect::serialize(ar, data);
    static_assert(std::is_same<mo::reflect::enable_if_reflected<InternallyReflected>, void>::value, "test2");
    std::cout << std::endl;
    data2.x = 5;
    data2.y = 6;
    data2.z = 10;
    mo::reflect::printStruct(std::cout, data2);
    std::cout << std::endl;
    return 0;
}
