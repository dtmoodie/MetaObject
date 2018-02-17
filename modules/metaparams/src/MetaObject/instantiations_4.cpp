#ifdef HAVE_OPENCV
#include "MetaObject/metaparams/reflect/cv_types.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "ct/reflect/cereal.hpp"
#include <boost/python/object.hpp>

namespace ct
{
    namespace reflect
    {
        template <class T, size_t N>
        struct ArrayAdapter;
    }
}

namespace mo
{
    namespace python
    {
        template <class T, size_t N>
        inline void convertFromPython(ct::reflect::ArrayAdapter<T, N> result, const boost::python::object& obj);
    }
}

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

namespace mo
{
    namespace python
    {
        template <class T, size_t N>
        struct PythonConverter<ct::reflect::ArrayAdapter<T, N>, void>
        {
            static inline void fromPython(const boost::python::object& obj, ct::reflect::ArrayAdapter<T, N> array) {}
        };
        template <class T, size_t N>
        inline void convertFromPython(ct::reflect::ArrayAdapter<T, N> result, const boost::python::object& obj)
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

static_assert(ct::reflect::ReflectData<cv::Scalar>::IS_SPECIALIZED, "Specialization not working for cv::Scalar");
static_assert(ct::reflect::ReflectData<cv::Vec2f>::IS_SPECIALIZED, "Specialization not working for cv::Vec2f");
static_assert(ct::reflect::ReflectData<cv::Vec2b>::IS_SPECIALIZED, "Specialization not working for cv::Vec2b");
static_assert(ct::reflect::ReflectData<cv::Vec3f>::IS_SPECIALIZED, "Specialization not working for cv::Vec3f");
static_assert(ct::reflect::ReflectData<cv::Vec3b>::IS_SPECIALIZED, "Specialization not working for cv::Vec3b");

using namespace cv;
INSTANTIATE_META_PARAM(Scalar);
INSTANTIATE_META_PARAM(Vec2f);
INSTANTIATE_META_PARAM(Vec3f);
INSTANTIATE_META_PARAM(Vec2b);
INSTANTIATE_META_PARAM(Vec3b);
INSTANTIATE_META_PARAM(std::vector<Vec3b>);
typedef std::map<std::string, Vec3b> ClassColormap_t;
INSTANTIATE_META_PARAM(ClassColormap_t);
#endif
