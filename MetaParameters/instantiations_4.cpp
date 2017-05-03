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
#include "MetaObject/Params/Detail/MetaParamImpl.hpp"
#include <boost/lexical_cast.hpp>

namespace mo
{
namespace IO
{
namespace Text
{
    namespace imp
    {
        void Serialize_imp(std::ostream &os, const cv::Scalar& obj, int)
        {
            os << obj.val[0] << ", " << obj.val[1] << ", " <<  obj.val[2] << ", " << obj.val[3];
        }
        void DeSerialize_imp(std::istream &is, cv::Scalar& obj, int)
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
#include "MetaObject/Params/IO/TextPolicy.hpp"
INSTANTIATE_META_Param(cv::Scalar);
INSTANTIATE_META_Param(cv::Vec2f);
INSTANTIATE_META_Param(cv::Vec3f);
INSTANTIATE_META_Param(cv::Vec2b);
INSTANTIATE_META_Param(cv::Vec3b);
#endif
