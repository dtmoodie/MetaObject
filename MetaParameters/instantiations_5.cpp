#ifdef HAVE_OPENCV

#include "MetaObject/Params/MetaParam.hpp"
#include "MetaObject/Params/UI/Qt/OpenCV.hpp"
#include "MetaObject/Params/UI/Qt/Containers.hpp"
#include "MetaObject/Params/UI/Qt/TParamProxy.hpp"
#include "MetaObject/Params/Buffers/CircularBuffer.hpp"
#include "MetaObject/Params/Buffers/StreamBuffer.hpp"
#include "MetaObject/Params/Buffers/Map.hpp"
#include "MetaObject/Params/IO/CerealPolicy.hpp"
#include "MetaObject/Params/IO/cvSpecializations.hpp"
#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && (defined MetaParams_EXPORTS)
#  define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_EXPORTS
#endif
#include "MetaObject/Params/detail/MetaParamsDetail.hpp"
#include "cereal/types/vector.hpp"
#include <boost/lexical_cast.hpp>



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

INSTANTIATE_META_Param(cv::Rect);
INSTANTIATE_META_Param(cv::Rect2d);
INSTANTIATE_META_Param(cv::Rect2f);
INSTANTIATE_META_Param(std::vector<cv::Rect>);
INSTANTIATE_META_Param(std::vector<cv::Rect2f>);

#endif
