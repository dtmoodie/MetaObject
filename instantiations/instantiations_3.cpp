#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"

INSTANTIATE_META_PARAMETER(cv::Point2f);
INSTANTIATE_META_PARAMETER(cv::Point2d);
INSTANTIATE_META_PARAMETER(cv::Point3d);
INSTANTIATE_META_PARAMETER(cv::Point3f);
INSTANTIATE_META_PARAMETER(cv::Point);
INSTANTIATE_META_PARAMETER(cv::Rect);
INSTANTIATE_META_PARAMETER(cv::Rect2d);
INSTANTIATE_META_PARAMETER(cv::Rect2f);
INSTANTIATE_META_PARAMETER(cv::Scalar);
INSTANTIATE_META_PARAMETER(cv::Vec2f);
INSTANTIATE_META_PARAMETER(cv::Vec3f);
INSTANTIATE_META_PARAMETER(cv::Vec2b);
INSTANTIATE_META_PARAMETER(cv::Vec3b);