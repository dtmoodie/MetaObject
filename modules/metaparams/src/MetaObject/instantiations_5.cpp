#include "MetaObject/core/metaobject_config.hpp"
#include "common.hpp"

#if MO_HAVE_OPENCV
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>
#include <MetaObject/types/opencv.hpp>

#include <ct/reflect.hpp>

#include <boost/lexical_cast.hpp>

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
static_assert(ct::IsReflected<cv::Rect>::value, "Specialization not working for cv::Rect");

namespace mo
{
    namespace metaparams
    {
        void instCVRect(SystemTable* table)
        {
            registerTrait<Size>();
            registerTrait<Rect>();
            registerTrait<Rect2d>();
            registerTrait<Rect2f>();
            registerTrait<std::vector<Rect>>();
            registerTrait<std::vector<Rect2f>>();
        }
    } // namespace metaparams
} // namespace mo

#endif
