#include "MetaObject/params/MetaParam.hpp"
#ifdef HAVE_QT
#include "MetaObject/params/ui/Qt/OpenCV.hpp"
#include "MetaObject/params/ui/Qt/Containers.hpp"
#include "MetaObject/params/ui/Qt/TParamProxy.hpp"
#endif
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/serialization/TextPolicy.hpp"
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include "MetaObject/serialization/CerealPolicy.hpp"
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
INSTANTIATE_META_PARAM(std::vector<int>);
INSTANTIATE_META_PARAM(std::vector<unsigned short>);
INSTANTIATE_META_PARAM(std::vector<unsigned int>);
INSTANTIATE_META_PARAM(std::vector<char>);
INSTANTIATE_META_PARAM(std::vector<unsigned char>);
INSTANTIATE_META_PARAM(std::vector<float>);
INSTANTIATE_META_PARAM(std::vector<double>);
INSTANTIATE_META_PARAM(std::vector<std::string>);
