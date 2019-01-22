#include "MetaObject/core/metaobject_config.hpp"

#if MO_HAVE_OPENCV
#include "MetaObject/metaparams/MetaParamsInclude.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include "MetaObject/types/opencv.hpp"
#include "ct/reflect.hpp"
#include "ct/reflect/cerealize.hpp"
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>
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
static_assert(ct::Reflect<cv::Rect>::SPECIALIZED, "Specialization not working for cv::Rect");

namespace mo
{
    namespace MetaParams
    {
        void instCVRect(SystemTable* table)
        {
            INSTANTIATE_META_PARAM(Size, table);
            INSTANTIATE_META_PARAM(Rect, table);
            INSTANTIATE_META_PARAM(Rect2d, table);
            INSTANTIATE_META_PARAM(Rect2f, table);
            INSTANTIATE_META_PARAM(std::vector<Rect>, table);
            INSTANTIATE_META_PARAM(std::vector<Rect2f>, table);
        }
    }
}

#endif
