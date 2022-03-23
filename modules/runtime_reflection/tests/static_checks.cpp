#include "common.hpp"
#include <MetaObject/runtime_reflection.hpp>
#include <MetaObject/runtime_reflection/type_traits.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/map.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/pair.hpp>
namespace mo
{
    void staticCheckFunction()
    {
        TTraits<mo::KVP<unsigned int, int>> t0;
        TTraits<std::map<std::string, int>> t1;
        TTraits<std::pair<std::string, std::string>> t2;
        static_assert(std::is_base_of<IContainerTraits, TTraits<std::map<std::string, int>>>::value, "");
        static_assert(IsPrimitive<uint8_t>::value, "");
        static_assert(IsPrimitive<int8_t>::value, "");
        static_assert(IsPrimitive<uint16_t>::value, "");
        static_assert(IsPrimitive<int16_t>::value, "");
        static_assert(IsPrimitive<int32_t>::value, "");
        static_assert(IsPrimitive<uint32_t>::value, "");
        static_assert(IsPrimitive<uint64_t>::value, "");
        static_assert(IsPrimitive<int64_t>::value, "");
        static_assert(IsPrimitive<long long>::value, "");
        static_assert(IsPrimitive<unsigned long long>::value, "");
        static_assert(IsPrimitive<char>::value, "");
        static_assert(IsPrimitive<bool>::value, "");
        static_assert(IsPrimitive<float>::value, "");
        static_assert(IsPrimitive<double>::value, "");
        static_assert(IsPrimitive<void*>::value, "");

        static_assert(ct::IsContainer<std::vector<int>>::value, "ct::IsContainer<std::vector<int>>::value");
        static_assert(ct::IsContainer<std::vector<Vec>>::value, "ct::IsContainer<std::vector<Vec>>::value");
        static_assert(is_complete<TTraits<std::vector<int>>>::value, "ct::IsContainer<std::vector<int>>::value");
        static_assert(is_complete<TTraits<std::vector<Vec>>>::value, "ct::IsContainer<std::vector<Vec>>::value");
        static_assert(is_complete<TTraits<Vec>>::value, "mo::is_complete<mo::TTraits<Vec>>::value");
        static_assert(is_complete<TTraits<const Vec>>::value, "mo::is_complete<mo::TTraits<Vec>>::value");
        // static_assert(is_complete<TTraits<std::string>>::value, "mo::is_complete<mo::TTraits<std::string>>::value");
    }
} // namespace mo
