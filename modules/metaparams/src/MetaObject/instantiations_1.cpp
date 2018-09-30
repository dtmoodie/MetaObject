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

#include "MetaObject/metaparams/MetaParamsInclude.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include <stdint.h>

#ifdef emit
#undef emit
#endif
#ifdef HAVE_WT
#define WT_NO_SLOT_MACROS
#include "MetaObject/params/ui/Wt/IParamProxy.hpp"
#include "MetaObject/params/ui/Wt/POD.hpp"
#include "MetaObject/params/ui/Wt/String.hpp"
#endif

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
            INSTANTIATE_META_PARAM(bool);
            INSTANTIATE_META_PARAM(int);
            INSTANTIATE_META_PARAM(unsigned short);
            INSTANTIATE_META_PARAM(unsigned int);
            INSTANTIATE_META_PARAM(char);
            INSTANTIATE_META_PARAM(unsigned char);
            INSTANTIATE_META_PARAM(long);
            INSTANTIATE_META_PARAM(long long);
#ifndef __arm__
            INSTANTIATE_META_PARAM(size_t);
#endif
            INSTANTIATE_META_PARAM(float);
            INSTANTIATE_META_PARAM(double);
            INSTANTIATE_META_PARAM(std::string);
            typedef std::map<std::string, std::string> StringMap;
            INSTANTIATE_META_PARAM(StringMap);
        }
    }
}
EXTERN_TYPE(bool);
EXTERN_TYPE(int);
EXTERN_TYPE(unsigned short);
EXTERN_TYPE(unsigned int);
EXTERN_TYPE(char);
EXTERN_TYPE(unsigned char);
EXTERN_TYPE(long);
EXTERN_TYPE(long long);
#ifndef __arm__
EXTERN_TYPE(size_t);
#endif
EXTERN_TYPE(float);
EXTERN_TYPE(double);
EXTERN_TYPE(std::string);
typedef std::map<std::string, std::string> StringMap;
EXTERN_TYPE(StringMap);
