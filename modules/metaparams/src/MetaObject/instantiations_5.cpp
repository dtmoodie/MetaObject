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
#include "MetaObject/serialization/cvSpecializations.hpp"
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include "cereal/types/vector.hpp"
#include <boost/lexical_cast.hpp>
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




namespace mo
{
namespace IO
{
namespace Text
{
    namespace imp
    {
        template<typename T>
        void Serialize_imp(std::ostream &os, const cv::Rect_<T>& obj, int)
        {
            os << obj.x << ", " << obj.y << ", " <<  obj.width << ", " << obj.height;
        }
        template<typename T>
        void DeSerialize_imp(std::istream &is, cv::Rect_<T>& obj, int)
        {
            char c;
            for(int i = 0; i < 4; ++i)
            {
                is >> obj[i];
                is >> c;
            }
        }
    }
}
}
}

INSTANTIATE_META_PARAM(cv::Rect);
INSTANTIATE_META_PARAM(cv::Rect2d);
INSTANTIATE_META_PARAM(cv::Rect2f);
INSTANTIATE_META_PARAM(std::vector<cv::Rect>);
INSTANTIATE_META_PARAM(std::vector<cv::Rect2f>);

#endif
