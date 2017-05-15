#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/UI/Qt/OpenCV.hpp"
#include "MetaObject/params/UI/Qt/Containers.hpp"
#include "MetaObject/params/UI/Qt/TParamProxy.hpp"
#include "MetaObject/params/Buffers/CircularBuffer.hpp"
#include "MetaObject/params/Buffers/StreamBuffer.hpp"
#include "MetaObject/params/Buffers/Map.hpp"
#include "MetaObject/serialization/CerealPolicy.hpp"
#include "MetaObject/serialization/TextPolicy.hpp"

#ifdef emit
#undef emit
#endif
#ifdef HAVE_WT
#define WT_NO_SLOT_MACROS
#include "MetaObject/params/UI/Wt/POD.hpp"
#include "MetaObject/params/UI/Wt/String.hpp"
#include "MetaObject/params/UI/Wt/IParamProxy.hpp"
#endif
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include "MetaParameters.hpp"

#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#  define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_EXPORTS
#endif
#include "MetaObject/params/detail/MetaParamImpl.hpp"

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
