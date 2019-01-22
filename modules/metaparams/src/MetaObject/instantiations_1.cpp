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
}

#include <MetaObject/metaparams/MetaParamsInclude.hpp>
#include <MetaObject/params/MetaParam.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/string.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/map.hpp>

#include <stdint.h>

#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include <MetaObject/types/cereal_map.hpp>


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
    namespace MetaParams
    {
        void instantiatePOD(SystemTable* table)
        {
            INSTANTIATE_META_PARAM(bool, table);
            INSTANTIATE_META_PARAM(int, table);
            INSTANTIATE_META_PARAM(unsigned short, table);
            INSTANTIATE_META_PARAM(unsigned int, table);
            INSTANTIATE_META_PARAM(char, table);
            INSTANTIATE_META_PARAM(unsigned char, table);
            INSTANTIATE_META_PARAM(long, table);
            INSTANTIATE_META_PARAM(long long, table);
#ifndef __arm__
            INSTANTIATE_META_PARAM(size_t, table);
#endif
            INSTANTIATE_META_PARAM(float, table);
            INSTANTIATE_META_PARAM(double, table);
            INSTANTIATE_META_PARAM(std::string, table);
            typedef std::map<std::string, std::string> StringMap;
            INSTANTIATE_META_PARAM(StringMap, table);
        }
    }
}
