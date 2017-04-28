#include "MetaObject/Params/MetaParam.hpp"
#include "MetaObject/Params/UI/Qt/OpenCV.hpp"
#include "MetaObject/Params/UI/Qt/Containers.hpp"
#include "MetaObject/Params/UI/Qt/TParamProxy.hpp"
#include "MetaObject/Params/Buffers/CircularBuffer.hpp"
#include "MetaObject/Params/Buffers/StreamBuffer.hpp"
#include "MetaObject/Params/Buffers/Map.hpp"
#include "MetaObject/Params/IO/CerealPolicy.hpp"
#include "MetaObject/Params/IO/TextPolicy.hpp"

#ifdef emit
#undef emit
#endif
#ifdef HAVE_WT
#define WT_NO_SLOT_MACROS
#include "MetaObject/Params/UI/Wt/POD.hpp"
#include "MetaObject/Params/UI/Wt/String.hpp"
#include "MetaObject/Params/UI/Wt/IParamProxy.hpp"
#endif
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include "MetaParams.hpp"
#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && (defined MetaParams_EXPORTS)
#  define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_EXPORTS
#endif
#include "MetaObject/Params/detail/MetaParamsDetail.hpp"

INSTANTIATE_META_Param(bool);
INSTANTIATE_META_Param(int);
INSTANTIATE_META_Param(unsigned short);
INSTANTIATE_META_Param(unsigned int);
INSTANTIATE_META_Param(char);
INSTANTIATE_META_Param(unsigned char);
INSTANTIATE_META_Param(long long);
INSTANTIATE_META_Param(size_t);
INSTANTIATE_META_Param(float);
INSTANTIATE_META_Param(double);
INSTANTIATE_META_Param(std::string);
typedef std::map<std::string, std::string> StringMap;
INSTANTIATE_META_Param(StringMap);

void mo::MetaParams::initialize()
{

}
