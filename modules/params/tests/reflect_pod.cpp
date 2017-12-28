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
    /*template<>
    struct ReflectData<ReflectedStruct>
    {
        static constexpr int N = 4;

        static constexpr auto& get(ReflectedStruct& data, mo::_counter_<0> = mo::_counter_<0>())
        {
            return data.x;
        }

        static constexpr const auto& get(const ReflectedStruct& data, mo::_counter_<0> = mo::_counter_<0>())
        {
            return data.x;
        }

        static constexpr auto& get(ReflectedStruct& data, mo::_counter_<1> = mo::_counter_<1>())
        {
            return data.y;
        }

        static constexpr const auto& get(const ReflectedStruct& data, mo::_counter_<1> = mo::_counter_<1>())
        {
            return data.y;
        }

        static constexpr auto& get(ReflectedStruct& data, mo::_counter_<2> = mo::_counter_<2>())
        {
            return data.z;
        }

        static constexpr const auto& get(const ReflectedStruct& data, mo::_counter_<2> = mo::_counter_<2>())
        {
            return data.z;
        }

        static constexpr auto& get(ReflectedStruct& data, mo::_counter_<3> = mo::_counter_<3>())
        {
            return data.id;
        }

        static constexpr const auto& get(const ReflectedStruct& data, mo::_counter_<3> = mo::_counter_<3>())
        {
            return data.id;
        }

        static constexpr const char* getName(mo::_counter_<0> dummy)
        {
            return "x";
        }

        static constexpr const char* getName(mo::_counter_<1> dummy)
        {
            return "y";
        }

        static constexpr const char* getName(mo::_counter_<2> dummy)
        {
            return "z";
        }

        static constexpr const char* getName(mo::_counter_<3> dummy)
        {
            return "id";
        }

    };*/
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


int main(int argc, char** argv)
{
    ReflectedStruct data;
    data.x = 0;
    data.y = 1;
    data.z = 2;
    data.id = 3;
    mo::reflect::printStruct(std::cout, data);
    static_assert(std::is_same<mo::reflect::enable_if_reflected<ReflectedStruct>, void>::value, "test1");
    //std::stringstream ss;
    cereal::JSONOutputArchive ar(std::cout);
    mo::reflect::serialize(ar, data);
    return 0;
    //std::cout << getName<0, ReflectedStruct>() << ":" << get<0>(data) << std::endl;
    //std::cout << getName<1, ReflectedStruct>() << ":" << get<1>(data) << std::endl;
    //std::cout << getName<2, ReflectedStruct>() << ":" << get<2>(data) << std::endl;
    //std::cout << getName<3, ReflectedStruct>() << ":" << get<3>(data) << std::endl;
}
