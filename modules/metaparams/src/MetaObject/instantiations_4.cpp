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
#include "MetaObject/serialization/cereal_map.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/string.hpp"
#include <cereal/types/vector.hpp>

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
#include "MetaObject/serialization/TextPolicy.hpp"
INSTANTIATE_META_PARAM(cv::Scalar);
INSTANTIATE_META_PARAM(cv::Vec2f);
INSTANTIATE_META_PARAM(cv::Vec3f);
INSTANTIATE_META_PARAM(cv::Vec2b);
INSTANTIATE_META_PARAM(cv::Vec3b);
INSTANTIATE_META_PARAM(std::vector<cv::Vec3b>);
typedef std::map<std::string, cv::Vec3b> ClassColormap_t;
INSTANTIATE_META_PARAM(ClassColormap_t);
#endif
