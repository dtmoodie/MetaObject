#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/UI/Qt/OpenCV.hpp"
#include "MetaObject/params/UI/Qt/Containers.hpp"
#include "MetaObject/params/UI/Qt/TParamProxy.hpp"
#include "MetaObject/params/Buffers/CircularBuffer.hpp"
#include "MetaObject/params/Buffers/StreamBuffer.hpp"
#include "MetaObject/params/Buffers/Map.hpp"
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
INSTANTIATE_META_Param(std::vector<int>);
INSTANTIATE_META_Param(std::vector<unsigned short>);
INSTANTIATE_META_Param(std::vector<unsigned int>);
INSTANTIATE_META_Param(std::vector<char>);
INSTANTIATE_META_Param(std::vector<unsigned char>);
INSTANTIATE_META_Param(std::vector<float>);
INSTANTIATE_META_Param(std::vector<double>);
INSTANTIATE_META_Param(std::vector<std::string>);
