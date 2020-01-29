#include "common.hpp"

#ifdef MO_HAVE_OPENCV

#include <MetaObject/params/MetaParam.hpp>
#include <MetaObject/types/opencv.hpp>
#include <ct/reflect/cerealize.hpp>

#include <MetaObject/core/metaobject_config.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/array_adapter.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/map.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>
#include <MetaObject/types/cereal_map.hpp>

#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>

#ifdef MO_HAVE_PYTHON
#include <boost/python/object.hpp>
#endif

namespace ct
{
    template <class T, size_t N>
    struct ArrayAdapter;
}

namespace boost
{
    namespace python
    {
        namespace api
        {
            class object;
        }

        using api::object;
    } // namespace python
} // namespace boost

namespace ct
{
    template <class T, ssize_t N>
    inline bool convertFromPython(const boost::python::object& obj, ct::TArrayView<T, N> result);
}
#include <ct/types/opencv.hpp>

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

#include <boost/lexical_cast.hpp>
#include <cereal/types/vector.hpp>

#ifdef MO_HAVE_PYTHON
namespace ct
{
    template <class T, ssize_t N>
    inline bool convertFromPython(const boost::python::object& obj, ct::TArrayView<T, N> result)
    {
        if (result.ptr)
        {
            for (size_t i = 0; i < result.size(); ++i)
            {
                boost::python::extract<T> extractor(obj[i]);
                result.ptr[i] = extractor();
            }
            return true;
        }
        return false;
    }
} // namespace ct
#endif

static_assert(ct::IsReflected<cv::Scalar>::value, "Specialization not working for cv::Scalar");
static_assert(ct::IsReflected<cv::Vec2f>::value, "Specialization not working for cv::Vec2f");
static_assert(ct::IsReflected<cv::Vec2b>::value, "Specialization not working for cv::Vec2b");
static_assert(ct::IsReflected<cv::Vec3f>::value, "Specialization not working for cv::Vec3f");
static_assert(ct::IsReflected<cv::Vec3b>::value, "Specialization not working for cv::Vec3b");

using namespace cv;
namespace mo
{
    namespace metaparams
    {
        void instCVVec(SystemTable* table)
        {
            registerTrait<Scalar>();
            registerTrait<Vec2f>();
            registerTrait<Vec3f>();
            registerTrait<Vec2b>();
            registerTrait<Vec3b>();
            registerTrait<std::vector<Vec3b>>();
            typedef std::map<std::string, Vec3b> ClassColormap_t;
            registerTrait<ClassColormap_t>();
        }
    } // namespace metaparams
} // namespace mo

#endif // MO_HAVE_OPENCV
