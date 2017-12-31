#ifdef HAVE_OPENCV
#include "MetaObject/metaparams/MetaParamsInclude.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include <opencv2/core/types.hpp>
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

namespace cv
{
    template <class T>
    std::ostream& operator<<(std::ostream& os, const cv::Point_<T>& pt)
    {
        os << pt.x << ", " << pt.y;
        return os;
    }
    template <class T>
    std::ostream& operator<<(std::ostream& os, const cv::Point3_<T>& pt)
    {
        os << pt.x << ", " << pt.y << ", " << pt.z;
        return os;
    }
}

namespace cereal
{
    template <class AR, class T>
    void serialize(AR& ar, cv::Point_<T>& pt)
    {
        ar(cereal::make_nvp("x", pt.x), cereal::make_nvp("y", pt.y));
    }

    template <class AR, class T>
    void serialize(AR& ar, cv::Point3_<T>& pt)
    {
        ar(cereal::make_nvp("x", pt.x), cereal::make_nvp("y", pt.y), cereal::make_nvp("z", pt.z));
    }
}

#include "MetaObject/params/detail/MetaParamImpl.hpp"
using namespace cv;

namespace mo
{
    namespace reflect
    {
        REFLECT_TEMPLATED_DATA_START(Point_)
            ;
            REFLECT_DATA_MEMBER(x)
            REFLECT_DATA_MEMBER(y)
        REFLECT_DATA_END();

        REFLECT_TEMPLATED_DATA_START(Point3_)
            ;
            REFLECT_DATA_MEMBER(x)
            REFLECT_DATA_MEMBER(y)
            REFLECT_DATA_MEMBER(z)
        REFLECT_DATA_END();
    }
}
INSTANTIATE_META_PARAM(Point2f);
INSTANTIATE_META_PARAM(Point2d);
INSTANTIATE_META_PARAM(Point3d);
INSTANTIATE_META_PARAM(Point3f);
INSTANTIATE_META_PARAM(Point);
#endif
