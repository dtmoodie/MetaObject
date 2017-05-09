#ifdef HAVE_OPENCV
#include "MetaObject/Params/MetaParam.hpp"
#include "MetaObject/Params/UI/Qt/OpenCV.hpp"
#include "MetaObject/Params/UI/Qt/Containers.hpp"
#include "MetaObject/Params/UI/Qt/TParamProxy.hpp"
#include "MetaObject/Params/Buffers/CircularBuffer.hpp"
#include "MetaObject/Params/Buffers/StreamBuffer.hpp"
#include "MetaObject/Params/Buffers/Map.hpp"
#include "MetaObject/Params/IO/CerealPolicy.hpp"
#include "MetaObject/Params/IO/TextPolicy.hpp"
#include <boost/lexical_cast.hpp>
#include "MetaObject/Params/IO/cvSpecializations.hpp"
#include "cereal/types/vector.hpp"
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
#include "MetaObject/Params/Detail/MetaParamImpl.hpp"
INSTANTIATE_META_Param(cv::Point2f);
INSTANTIATE_META_Param(cv::Point2d);
INSTANTIATE_META_Param(cv::Point3d);
INSTANTIATE_META_Param(cv::Point3f);
INSTANTIATE_META_Param(cv::Point);

#endif
