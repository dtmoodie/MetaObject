#ifdef HAVE_OPENCV
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include "MetaObject/Parameters/IO/cvSpecializations.hpp"
#include <boost/lexical_cast.hpp>


INSTANTIATE_META_PARAMETER(cv::Scalar);
INSTANTIATE_META_PARAMETER(cv::Vec2f);
INSTANTIATE_META_PARAMETER(cv::Vec3f);
INSTANTIATE_META_PARAMETER(cv::Vec2b);
INSTANTIATE_META_PARAMETER(cv::Vec3b);
#endif