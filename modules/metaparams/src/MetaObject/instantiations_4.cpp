#include "MetaObject/core/metaobject_config.hpp"
#if MO_HAVE_OPENCV
#include "MetaObject/metaparams/reflect/cv_types.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "ct/reflect/cerealize.hpp"
#if MO_HAVE_PYTHON
#include <boost/python/object.hpp>
#endif

namespace ct
{
    template <class T, size_t N>
    struct ArrayAdapter;

}
#if MO_HAVE_PYTHON
namespace mo
{
    namespace python
    {
        template <class T, size_t N>
        inline void convertFromPython(const boost::python::object& obj, ct::ArrayAdapter<T, N> result);
    }
}
#endif

#include "MetaObject/metaparams/MetaParamsInclude.hpp"
#include "opencv2/core/types.hpp"
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
#include <boost/lexical_cast.hpp>
#include <cereal/types/vector.hpp>

#if MO_HAVE_PYTHON
namespace mo
{
    namespace python
    {
        template <class T, size_t N>
        inline void convertFromPython(const boost::python::object& obj, ct::ArrayAdapter<T, N> result)
        {
            if (result.ptr)
            {
                for (size_t i = 0; i < N; ++i)
                {
                    boost::python::extract<T> extractor(obj[i]);
                    result.ptr[i] = extractor();
                }
            }
        }
    }
}
#endif

static_assert(ct::Reflect<cv::Scalar>::SPECIALIZED, "Specialization not working for cv::Scalar");
static_assert(ct::Reflect<cv::Vec2f>::SPECIALIZED, "Specialization not working for cv::Vec2f");
static_assert(ct::Reflect<cv::Vec2b>::SPECIALIZED, "Specialization not working for cv::Vec2b");
static_assert(ct::Reflect<cv::Vec3f>::SPECIALIZED, "Specialization not working for cv::Vec3f");
static_assert(ct::Reflect<cv::Vec3b>::SPECIALIZED, "Specialization not working for cv::Vec3b");

using namespace cv;
namespace mo
{
    namespace MetaParams
    {
        void instCVVec(SystemTable* table)
        {
            INSTANTIATE_META_PARAM(Scalar);
            INSTANTIATE_META_PARAM(Vec2f);
            INSTANTIATE_META_PARAM(Vec3f);
            INSTANTIATE_META_PARAM(Vec2b);
            INSTANTIATE_META_PARAM(Vec3b);
            INSTANTIATE_META_PARAM(std::vector<Vec3b>);
            typedef std::map<std::string, Vec3b> ClassColormap_t;
            INSTANTIATE_META_PARAM(ClassColormap_t);
        }
    }
}
EXTERN_TYPE(Scalar);
EXTERN_TYPE(Vec2f);
EXTERN_TYPE(Vec3f);
EXTERN_TYPE(Vec2b);
EXTERN_TYPE(Vec3b);
EXTERN_TYPE(std::vector<Vec3b>);
typedef std::map<std::string, Vec3b> ClassColormap_t;
EXTERN_TYPE(ClassColormap_t);
#endif
