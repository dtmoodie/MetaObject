#ifdef HAVE_OPENCV
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include <boost/lexical_cast.hpp>

namespace cereal
{
    template<class AR, class T> void serialize(AR& ar, cv::Point_<T>& pt)
    {
        ar(make_nvp("x", pt.x), make_nvp("y", pt.y));
    }
    template<class AR, class T> void serialize(AR& ar, cv::Point3_<T>& pt)
    {
        ar(make_nvp("x", pt.x), make_nvp("y", pt.y), make_nvp("z", pt.z));
    }
    template<class AR, class T> void serialize(AR& ar, cv::Scalar_<T>& pt)
    {
        ar(make_nvp("0", pt[0]), make_nvp("1", pt[1]), make_nvp("2", pt[2]), make_nvp("3", pt[3]));
    }
    template<class AR, class T> void serialize(AR& ar, cv::Rect_<T>& rect)
    {
        ar(make_nvp("x", rect.x), make_nvp("y", rect.y), make_nvp("width", rect.width), make_nvp("height", rect.height));
    }

    template<class AR, class T, int N> void serialize(AR& ar, cv::Vec<T, N>& vec)
    {
        for (int i = 0; i < N; ++i)
        {
            ar(make_nvp(std::to_string(i), vec[i]));
        }
    }
}


INSTANTIATE_META_PARAMETER(cv::Scalar);
INSTANTIATE_META_PARAMETER(cv::Vec2f);
INSTANTIATE_META_PARAMETER(cv::Vec3f);
INSTANTIATE_META_PARAMETER(cv::Vec2b);
INSTANTIATE_META_PARAMETER(cv::Vec3b);
#endif