#include "MetaObject/params/MetaParam.hpp"

#include "MetaObject/metaparams/MetaParamsInclude.hpp"

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
INSTANTIATE_META_PARAM(std::vector<int>);
INSTANTIATE_META_PARAM(std::vector<unsigned short>);
INSTANTIATE_META_PARAM(std::vector<unsigned int>);
INSTANTIATE_META_PARAM(std::vector<char>);
INSTANTIATE_META_PARAM(std::vector<unsigned char>);
INSTANTIATE_META_PARAM(std::vector<float>);
INSTANTIATE_META_PARAM(std::vector<double>);
INSTANTIATE_META_PARAM(std::vector<std::string>);
