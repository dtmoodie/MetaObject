#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/Map.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include "MetaObject/Parameters/IO/TextPolicy.hpp"

#ifdef emit
#undef emit
#endif
#ifdef HAVE_WT
#define WT_NO_SLOT_MACROS
#include "MetaObject/Parameters/UI/Wt/POD.hpp"
#include "MetaObject/Parameters/UI/Wt/String.hpp"
#include "MetaObject/Parameters/UI/Wt/IParameterProxy.hpp"
#endif
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include "MetaParameters.hpp"
#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && (defined MetaParameters_EXPORTS)
#  define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_EXPORTS
#endif
#include "MetaObject/Parameters/detail/MetaParametersDetail.hpp"

INSTANTIATE_META_PARAMETER(bool);
INSTANTIATE_META_PARAMETER(int);
INSTANTIATE_META_PARAMETER(unsigned short);
INSTANTIATE_META_PARAMETER(unsigned int);
INSTANTIATE_META_PARAMETER(char);
INSTANTIATE_META_PARAMETER(unsigned char);
INSTANTIATE_META_PARAMETER(long long);
INSTANTIATE_META_PARAMETER(size_t);
INSTANTIATE_META_PARAMETER(float);
INSTANTIATE_META_PARAMETER(double);
INSTANTIATE_META_PARAMETER(std::string);
typedef std::map<std::string, std::string> StringMap;
INSTANTIATE_META_PARAMETER(StringMap);

void mo::MetaParameters::initialize()
{

}
