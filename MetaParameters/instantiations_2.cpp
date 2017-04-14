#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/Map.hpp"
#include "MetaObject/Parameters/IO/TextPolicy.hpp"
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
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
INSTANTIATE_META_PARAMETER(std::vector<int>);
INSTANTIATE_META_PARAMETER(std::vector<unsigned short>);
INSTANTIATE_META_PARAMETER(std::vector<unsigned int>);
INSTANTIATE_META_PARAMETER(std::vector<char>);
INSTANTIATE_META_PARAMETER(std::vector<unsigned char>);
INSTANTIATE_META_PARAMETER(std::vector<float>);
INSTANTIATE_META_PARAMETER(std::vector<double>);
INSTANTIATE_META_PARAMETER(std::vector<std::string>);
