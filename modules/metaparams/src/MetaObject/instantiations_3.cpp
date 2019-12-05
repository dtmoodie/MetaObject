#include "MetaObject/core/metaobject_config.hpp"
#ifdef MO_HAVE_OPENCV
#include "MetaObject/types/opencv.hpp"

#include "MetaObject/params/MetaParam.hpp"

#include "ct/reflect/cerealize.hpp"
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

#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include <opencv2/core/types.hpp>
using namespace cv;

namespace mo
{
    namespace MetaParams
    {
        void instCV(SystemTable* table)
        {
            INSTANTIATE_META_PARAM(Point2f, table);
            INSTANTIATE_META_PARAM(Point2d, table);
            INSTANTIATE_META_PARAM(Point3d, table);
            INSTANTIATE_META_PARAM(Point3f, table);
            INSTANTIATE_META_PARAM(Point, table);
        }
    } // namespace MetaParams
} // namespace mo

#endif
