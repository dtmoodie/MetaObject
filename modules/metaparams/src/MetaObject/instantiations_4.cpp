#ifdef HAVE_OPENCV
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/metaparams/MetaParamsInclude.hpp"

#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#define MO_EXPORTS __attribute__((visibility("default")))
#else
#define MO_EXPORTS
#endif
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include <boost/lexical_cast.hpp>
#include <cereal/types/vector.hpp>

namespace mo
{
namespace IO
{
    namespace Text
    {
        namespace imp
        {
            void Serialize_imp(std::ostream& os, const cv::Scalar& obj, int) {
                os << obj.val[0] << ", " << obj.val[1] << ", " << obj.val[2] << ", " << obj.val[3];
            }
            void DeSerialize_imp(std::istream& is, cv::Scalar& obj, int) {
                char c;
                for (int i = 0; i < 4; ++i) {
                    is >> obj[i];
                    is >> c;
                }
            }
        }
    }
}
}

#include "MetaObject/serialization/TextPolicy.hpp"
#define ASSERT_SERIALIZABLE(TYPE) static_assert(mo::IO::Text::imp::stream_serializable<TYPE>::value, "Checking stream serializable for " #TYPE)

namespace cv
{
    template <class T>
    std::ostream& operator<<(std::ostream& os, const cv::Scalar_<T>& obj)
    {
        ASSERT_SERIALIZABLE(cv::Scalar);
        os << obj.val[0] << ", " << obj.val[1] << ", " << obj.val[2] << ", " << obj.val[3];
        return os;
    }
}

namespace cereal
{
    template<class AR, class T>
    void serialize(AR& ar, cv::Scalar_<T>& scalar)
    {

    }

    template<class AR, class T, int N>
    void serialize(AR& ar, cv::Vec<T, N>& vec)
    {

    }
}

INSTANTIATE_META_PARAM(cv::Scalar);
INSTANTIATE_META_PARAM(cv::Vec2f);
INSTANTIATE_META_PARAM(cv::Vec3f);
INSTANTIATE_META_PARAM(cv::Vec2b);
INSTANTIATE_META_PARAM(cv::Vec3b);
INSTANTIATE_META_PARAM(std::vector<cv::Vec3b>);
typedef std::map<std::string, cv::Vec3b> ClassColormap_t;
INSTANTIATE_META_PARAM(ClassColormap_t);
#endif
