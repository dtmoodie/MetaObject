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

#define REFLECT_INTERNAL_MEMBER(TYPE, NAME) \
    TYPE NAME; \
    REFLECT_DATA_MEMBER(NAME)

#define REFLECT_INTERNAL_START(TYPE) \
struct TYPE \
{ \
    static constexpr int START = __COUNTER__; \
    typedef TYPE DType; \
    typedef void INTERNALLY_REFLECTED;

#define REFLECT_INTERNAL_END() static constexpr int N = __COUNTER__ - START - 1; }

REFLECT_INTERNAL_START(InternallyReflected)
    REFLECT_INTERNAL_MEMBER(float, x)
    REFLECT_INTERNAL_MEMBER(float, y)
    REFLECT_INTERNAL_MEMBER(float, z)
REFLECT_INTERNAL_END();

namespace mo
{
    namespace reflect
    {
        template<class T>
        struct ReflectData<T, decltype(T::get(std::declval<T>(), mo::_counter_<0>()), void())>
        {
            static constexpr bool IS_SPECIALIZED = true;
            static constexpr int N = T::N;
            static constexpr auto get(const T& data, mo::_counter_<0>){return T::get(data, mo::_counter_<0>());}
            static constexpr auto get(T& data, mo::_counter_<0>){return T::get(data, mo::_counter_<0>());}
            template<int I>
            static constexpr auto get(const T& data, mo::_counter_<I>){return T::get(data, mo::_counter_<I>());}
            template<int I>
            static constexpr auto get(T& data, mo::_counter_<I>){return T::get(data, mo::_counter_<I>());}
            static constexpr const char* getName(mo::_counter_<0> cnt){return T::getName(cnt);}
            template<int I>
            static constexpr const char* getName(mo::_counter_<I> cnt){return T::getName(cnt);}
        };
    }
}

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
    //std::stringstream ss;
    cereal::JSONOutputArchive ar(std::cout);
    mo::reflect::serialize(ar, data);
    //decltype(InternallyReflected::get(std::declval<InternallyReflected>(), mo::_counter_<0>())) type;
    static_assert(std::is_same<mo::reflect::enable_if_reflected<InternallyReflected>, void>::value, "test2");
    std::cout << std::endl;
    data2.x = 5;
    data2.y = 6;
    data2.z = 10;
    mo::reflect::printStruct(std::cout, data2);
    std::cout << std::endl;
    return 0;
    //std::cout << getName<0, ReflectedStruct>() << ":" << get<0>(data) << std::endl;
    //std::cout << getName<1, ReflectedStruct>() << ":" << get<1>(data) << std::endl;
    //std::cout << getName<2, ReflectedStruct>() << ":" << get<2>(data) << std::endl;
    //std::cout << getName<3, ReflectedStruct>() << ":" << get<3>(data) << std::endl;
}
