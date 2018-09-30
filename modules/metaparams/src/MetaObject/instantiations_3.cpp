#include "MetaObject/core/metaobject_config.hpp"
#if MO_HAVE_OPENCV
#include "MetaObject/metaparams/MetaParamsInclude.hpp"
#include "MetaObject/metaparams/reflect/cv_types.hpp"
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
using namespace cv;

namespace mo
{
    namespace MetaParams
    {
        void instCV(SystemTable* table)
        {
            INSTANTIATE_META_PARAM(Point2f);
            INSTANTIATE_META_PARAM(Point2d);
            INSTANTIATE_META_PARAM(Point3d);
            INSTANTIATE_META_PARAM(Point3f);
            INSTANTIATE_META_PARAM(Point);
        }
    }
}
EXTERN_TYPE(Point2f);
EXTERN_TYPE(Point2d);
EXTERN_TYPE(Point3d);
EXTERN_TYPE(Point3f);
EXTERN_TYPE(Point);

#endif
