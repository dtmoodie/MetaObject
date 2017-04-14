#ifdef HAVE_OPENCV
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/Map.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include "MetaObject/Parameters/IO/cvSpecializations.hpp"
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
#include "MetaObject/Parameters/IO/TextPolicy.hpp"
INSTANTIATE_META_PARAMETER(cv::Scalar);
INSTANTIATE_META_PARAMETER(cv::Vec2f);
INSTANTIATE_META_PARAMETER(cv::Vec3f);
INSTANTIATE_META_PARAMETER(cv::Vec2b);
INSTANTIATE_META_PARAMETER(cv::Vec3b);
#endif
