#include "MetaObject/params/MetaParam.hpp"
#include "ct/reflect/cerealize.hpp"

#include "MetaObject/metaparams/MetaParamsInclude.hpp"
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/string.hpp>

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
