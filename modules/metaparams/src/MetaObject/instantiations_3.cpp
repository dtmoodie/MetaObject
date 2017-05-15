#ifdef HAVE_OPENCV
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/ui/Qt/OpenCV.hpp"
#include "MetaObject/params/ui/Qt/Containers.hpp"
#include "MetaObject/params/ui/Qt/TParamProxy.hpp"
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/serialization/CerealPolicy.hpp"
#include "MetaObject/serialization/TextPolicy.hpp"
#include <boost/lexical_cast.hpp>
#include "MetaObject/serialization/cvSpecializations.hpp"
#include "cereal/types/vector.hpp"
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
INSTANTIATE_META_Param(cv::Point2f);
INSTANTIATE_META_Param(cv::Point2d);
INSTANTIATE_META_Param(cv::Point3d);
INSTANTIATE_META_Param(cv::Point3f);
INSTANTIATE_META_Param(cv::Point);

#endif
