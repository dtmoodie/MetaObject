
#include "common.hpp"

#include <map>
#include <ostream>
#include <string>

namespace std
{
    template <class K, class V>
    ostream& operator<<(ostream& os, const map<K, V>& map_)
    {
        for (const auto& kvp : map_)
        {
            os << kvp.first << ':' << kvp.second << ' ';
        }
        return os;
    }
} // namespace std

#include <MetaObject/params/MetaParam.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/map.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/string.hpp>

#include <stdint.h>
#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#define MO_EXPORTS __attribute__((visibility("default")))
#else
#define MO_EXPORTS
#endif

#include "MetaObject/params/detail/MetaParamImpl.hpp"

namespace mo
{
    namespace metaparams
    {
        void instantiatePOD(SystemTable*)
        {
            registerTrait<bool>();
            registerTrait<uint8_t>();
            registerTrait<int8_t>();
            registerTrait<uint16_t>();
            registerTrait<int16_t>();
            registerTrait<uint32_t>();
            registerTrait<int32_t>();
            registerTrait<uint64_t>();
            registerTrait<int64_t>();
            registerTrait<float>();
            registerTrait<double>();
#ifndef __arm__
            registerTrait<ssize_t>();
            registerTrait<size_t>();
#endif
            registerTrait<std::string>();
            registerTrait<std::map<std::string, std::string>>();
        }
    } // namespace metaparams
} // namespace mo
