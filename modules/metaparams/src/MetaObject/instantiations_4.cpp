#ifdef HAVE_OPENCV
#include "MetaObject/params/MetaParam.hpp"
#include "ct/reflect/reflect_data.hpp"
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

using namespace cv;
namespace ct
{
    namespace reflect
    {
        template <class T, size_t N>
        struct ArrayAdapter
        {
            ArrayAdapter(T* ptr_ = nullptr) : ptr(ptr_) {}

            T* ptr;
            template <class AR>
            void serialize(AR& ar)
            {
                if (ptr)
                {
                    ar(cereal::make_size_tag(N));
                    for (size_t i = 0; i < N; ++i)
                    {
                        ar(ptr[i]);
                    }
                }
            }
        };

        template <class T, size_t N>
        std::ostream& operator<<(std::ostream& os, const ArrayAdapter<T, N>& array)
        {
            if (array.ptr)
            {
                os << "[";
                for (size_t i = 0; i < N; ++i)
                {
                    if (i != 0)
                        os << ',';
                    os << array.ptr[i];
                }
            }
            return os;
        }

        template <size_t N>
        std::ostream& operator<<(std::ostream& os, const ArrayAdapter<const unsigned char, N>& array)
        {
            if (array.ptr)
            {
                os << "[";
                for (size_t i = 0; i < N; ++i)
                {
                    if (i != 0)
                        os << ',';
                    os << int(array.ptr[i]);
                }
                os << "]";
            }
            return os;
        }

        template <class T, size_t Rows, size_t Cols>
        struct MatrixAdapter
        {
            MatrixAdapter(T* ptr_ = nullptr) : ptr(ptr_) {}

            T* ptr;

            template <class AR>
            void serialize(AR& ar)
            {
                if (ptr)
                {
                    const size_t N = Rows * Cols;
                    ar(cereal::make_size_tag(N));
                    for (int i = 0; i < N; ++i)
                    {
                        ar(ptr[i]);
                    }
                }
            }
        };

        template <class T, size_t R, size_t C>
        struct ReflectData<cv::Matx<T, R, C>>
        {
            static constexpr int N = 1;
            static constexpr int IS_SPECIALIZED = true;
            static constexpr MatrixAdapter<T, R, C> get(cv::Matx<T, R, C>& data, _counter_<0>) { return data.val; }
            static constexpr MatrixAdapter<const T, R, C> get(const cv::Matx<T, R, C>& data, _counter_<0>)
            {
                return data.val;
            }
            static constexpr const char* getName(_counter_<0>) { return "data"; }
        };

        template <class T, int R>
        struct ReflectData<cv::Vec<T, R>>
        {
            static constexpr int N = 1;
            static constexpr int IS_SPECIALIZED = true;
            static constexpr ArrayAdapter<T, R> get(cv::Vec<T, R>& data, _counter_<0>) { return data.val; }
            static constexpr ArrayAdapter<const T, R> get(const cv::Vec<T, R>& data, _counter_<0>)
            {
                return data.val;
            }
            static constexpr const char* getName(_counter_<0>) { return "data"; }
        };

        template <class T>
        struct ReflectData<cv::Scalar_<T>>
        {
            static constexpr int N = 1;
            static constexpr int IS_SPECIALIZED = true;
            static constexpr ArrayAdapter<T, 4> get(cv::Scalar_<T>& data, _counter_<0>) { return &data.val[0]; }
            static constexpr ArrayAdapter<const T, 4> get(const cv::Scalar_<T>& data, _counter_<0>)
            {
                return &data.val[0];
            }
            static constexpr const char* getName(_counter_<0>) { return "data"; }
        };
    }
}

namespace mo
{
    namespace python
    {
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

INSTANTIATE_META_PARAM(Scalar);
INSTANTIATE_META_PARAM(Vec2f);
INSTANTIATE_META_PARAM(Vec3f);
INSTANTIATE_META_PARAM(Vec2b);
INSTANTIATE_META_PARAM(Vec3b);
INSTANTIATE_META_PARAM(std::vector<Vec3b>);
typedef std::map<std::string, Vec3b> ClassColormap_t;
INSTANTIATE_META_PARAM(ClassColormap_t);
#endif
