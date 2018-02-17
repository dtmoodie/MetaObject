#ifdef HAVE_OPENCV
#include "MetaObject/metaparams/MetaParamsInclude.hpp"
#include "MetaObject/metaparams/reflect/cv_types.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include "ct/reflect/cereal.hpp"
#include "ct/reflect/reflect_data.hpp"
#include <boost/lexical_cast.hpp>
#include <cereal/types/vector.hpp>
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

using namespace cv;
static_assert(ct::reflect::ReflectData<cv::Rect>::IS_SPECIALIZED, "Specialization not working for cv::Rect");

INSTANTIATE_META_PARAM(Rect);
INSTANTIATE_META_PARAM(Rect2d);
INSTANTIATE_META_PARAM(Rect2f);
INSTANTIATE_META_PARAM(std::vector<Rect>);
INSTANTIATE_META_PARAM(std::vector<Rect2f>);

#endif
