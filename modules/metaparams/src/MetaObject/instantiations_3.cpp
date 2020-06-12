#include "common.hpp"

#include "MetaObject/core/metaobject_config.hpp"
#ifdef MO_HAVE_OPENCV
#include <MetaObject/runtime_reflection/StructTraits.hpp>
#include <MetaObject/types/opencv.hpp>

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

#include <opencv2/core/types.hpp>
using namespace cv;

namespace mo
{
    namespace metaparams
    {
        void instCV(SystemTable* table)
        {
            registerTrait<Point2f>();
            registerTrait<Point2d>();
            registerTrait<Point3d>();
            registerTrait<Point3f>();
            registerTrait<Point>();
        }
    } // namespace metaparams
} // namespace mo

#endif
