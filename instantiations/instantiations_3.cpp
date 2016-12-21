#ifdef HAVE_OPENCV
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include "MetaObject/Parameters/IO/TextPolicy.hpp"
#include <boost/lexical_cast.hpp>
#include "cereal/types/vector.hpp"

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
        for(int i = 0; i < N; ++i)
        {
            ar(make_nvp(std::to_string(i), vec[i]));
        }
    }
}
template<typename T> std::ostream& operator<<(std::ostream& out, cv::Point_<T>& pt)
{
    out << pt.x << ", " << pt.y;
    return out;
}

template<typename T> std::istream& operator>>(std::istream& in, cv::Point_<T>& pt)
{
    char ch;
    in >> pt.x >> ch >> pt.y;
    return in;
}

template<typename T> std::ostream& operator<<(std::ostream& out, cv::Point3_<T>& pt)
{
    out << pt.x << ", " << pt.y << ", " << pt.z;
    return out;
}

template<typename T> std::istream& operator >> (std::istream& in, cv::Point3_<T>& pt)
{
    char ch;
    in >> pt.x >> ch >> pt.y >> ch >> pt.z;
    return in;
}
template<typename T> std::ostream& operator<<(std::ostream& out, cv::Rect_<T>& pt)
{
    out << pt.x << ", " << pt.y << ", " << pt.width << ", " << pt.height;
    return out;
}

template<typename T> std::istream& operator >> (std::istream& in, cv::Rect_<T>& pt)
{
    char ch;
    in >> pt.x >> ch >> pt.y >> ch >> pt.width >> ch >> pt.height;
    return in;
}

INSTANTIATE_META_PARAMETER(cv::Point2f);
INSTANTIATE_META_PARAMETER(cv::Point2d);
INSTANTIATE_META_PARAMETER(cv::Point3d);
INSTANTIATE_META_PARAMETER(cv::Point3f);
INSTANTIATE_META_PARAMETER(cv::Point);
INSTANTIATE_META_PARAMETER(cv::Rect);
INSTANTIATE_META_PARAMETER(cv::Rect2d);
INSTANTIATE_META_PARAMETER(cv::Rect2f);
INSTANTIATE_META_PARAMETER(std::vector<cv::Rect>);
#endif
