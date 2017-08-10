#ifdef HAVE_OPENCV
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
namespace cv{
    template<class T>
    std::ostream& operator<<(std::ostream& os, const cv::Point_<T>& pt){
        os << pt.x << ", " << pt.y;
        return os;
    }
    template<class T>
    std::ostream& operator<<(std::ostream& os, const cv::Point3_<T>& pt){
        os << pt.x << ", " << pt.y << ", " << pt.z;
        return os;
    }

}
#include "MetaObject/params/detail/MetaParamImpl.hpp"
INSTANTIATE_META_PARAM(cv::Point2f);
INSTANTIATE_META_PARAM(cv::Point2d);
INSTANTIATE_META_PARAM(cv::Point3d);
INSTANTIATE_META_PARAM(cv::Point3f);
INSTANTIATE_META_PARAM(cv::Point);

#endif
