#include "MetaObject/params/MetaParam.hpp"
#include "ct/reflect/cerealize.hpp"

#include "MetaObject/metaparams/MetaParamsInclude.hpp"
#include <MetaObject/visitation/visitor_traits/vector.hpp>

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
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace mo
{
    namespace MetaParams
    {
        void instantiateVectors(SystemTable* table)
        {
            INSTANTIATE_META_PARAM(std::vector<int>, table);
            INSTANTIATE_META_PARAM(std::vector<unsigned short>, table);
            INSTANTIATE_META_PARAM(std::vector<unsigned int>, table);
            INSTANTIATE_META_PARAM(std::vector<char>, table);
            INSTANTIATE_META_PARAM(std::vector<unsigned char>, table);
            INSTANTIATE_META_PARAM(std::vector<float>, table);
            INSTANTIATE_META_PARAM(std::vector<double>, table);
            INSTANTIATE_META_PARAM(std::vector<std::string>, table);
        }
    }
}

EXTERN_TYPE(std::vector<int>);
EXTERN_TYPE(std::vector<unsigned short>);
EXTERN_TYPE(std::vector<unsigned int>);
EXTERN_TYPE(std::vector<char>);
EXTERN_TYPE(std::vector<unsigned char>);
EXTERN_TYPE(std::vector<float>);
EXTERN_TYPE(std::vector<double>);
EXTERN_TYPE(std::vector<std::string>);
