#include "MetaObject/params/MetaParam.hpp"
#ifdef HAVE_QT
#include "MetaObject/params/ui/Qt/OpenCV.hpp"
#include "MetaObject/params/ui/Qt/Containers.hpp"
#include "MetaObject/params/ui/Qt/TParamProxy.hpp"
#endif
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/serialization/CerealPolicy.hpp"
#include "MetaObject/serialization/TextPolicy.hpp"
#include "MetaObject/serialization/cereal_map.hpp"

#ifdef emit
#undef emit
#endif
#ifdef HAVE_WT
#define WT_NO_SLOT_MACROS
#include "MetaObject/params/ui/Wt/POD.hpp"
#include "MetaObject/params/ui/Wt/String.hpp"
#include "MetaObject/params/ui/Wt/IParamProxy.hpp"
#endif
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>


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

INSTANTIATE_META_PARAM(bool);
INSTANTIATE_META_PARAM(int);
INSTANTIATE_META_PARAM(unsigned short);
INSTANTIATE_META_PARAM(unsigned int);
INSTANTIATE_META_PARAM(char);
INSTANTIATE_META_PARAM(unsigned char);
INSTANTIATE_META_PARAM(long);
INSTANTIATE_META_PARAM(long long);
INSTANTIATE_META_PARAM(size_t);
INSTANTIATE_META_PARAM(float);
INSTANTIATE_META_PARAM(double);
INSTANTIATE_META_PARAM(std::string);
typedef std::map<std::string, std::string> StringMap;
INSTANTIATE_META_PARAM(StringMap);

