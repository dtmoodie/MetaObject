#ifdef HAVE_OPENCV
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/Map.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include "MetaObject/Parameters/IO/TextPolicy.hpp"
#include <boost/lexical_cast.hpp>
#include "MetaObject/Parameters/IO/cvSpecializations.hpp"
#include "cereal/types/vector.hpp"
#include "MetaObject/Parameters/detail/MetaParametersDetail.hpp"
INSTANTIATE_META_PARAMETER(cv::Point2f);
INSTANTIATE_META_PARAMETER(cv::Point2d);
INSTANTIATE_META_PARAMETER(cv::Point3d);
INSTANTIATE_META_PARAMETER(cv::Point3f);
INSTANTIATE_META_PARAMETER(cv::Point);

#endif
