#ifdef HAVE_OPENCV
#include "MetaObject/params/MetaParam.hpp"
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
namespace IO
{
    namespace Text
    {
        namespace imp
        {
            void Serialize_imp(std::ostream& os, const cv::Scalar& obj, int) {
                os << obj.val[0] << ", " << obj.val[1] << ", " << obj.val[2] << ", " << obj.val[3];
            }
            void DeSerialize_imp(std::istream& is, cv::Scalar& obj, int) {
                char c;
                for (int i = 0; i < 4; ++i) {
                    is >> obj[i];
                    is >> c;
                }
            }
        }
    }
}
}

#include "MetaObject/serialization/TextPolicy.hpp"
#define ASSERT_SERIALIZABLE(TYPE) static_assert(mo::IO::Text::imp::stream_serializable<TYPE>::value, "Checking stream serializable for " #TYPE)

namespace cv
{
    template <class T>
    std::ostream& operator<<(std::ostream& os, const cv::Scalar_<T>& obj)
    {
        ASSERT_SERIALIZABLE(cv::Scalar);
        os << obj.val[0] << ", " << obj.val[1] << ", " << obj.val[2] << ", " << obj.val[3];
        return os;
    }
}

namespace cereal
{
    template<class AR, class T>
    void serialize(AR& ar, cv::Scalar_<T>& scalar)
    {

    }

    template<class AR, class T, int N>
    void serialize(AR& ar, cv::Vec<T, N>& vec)
    {

    }
}

using namespace cv;
namespace mo
{
    namespace reflect
    {
        template<class T, size_t N>
        struct ArrayAdapter
        {
            ArrayAdapter(T* ptr_ = nullptr):
                ptr(ptr_)
            {
            }

            T* ptr;
        };

        template<class T, size_t N>
        std::ostream& operator<<(std::ostream& os, const ArrayAdapter<T, N>& array)
        {
            if(array.ptr)
            {
                os << "[";
                for(size_t i = 0; i < N; ++i)
                {
                    if(i != 0)
                        os << ',';
                    os << array.ptr[i];
                }
            }
            return os;
        }

        template<class T, size_t Rows, size_t Cols>
        struct MatrixAdapter
        {
            MatrixAdapter(T* ptr_ = nullptr):
                ptr(ptr_)
            {
            }

            T* ptr;
        };

        template<class T, size_t R, size_t C>
        struct ReflectData<cv::Matx<T, R, C>>
        {
            static constexpr int N = 1;
            static constexpr int IS_SPECIALIZED = true;
            static constexpr MatrixAdapter<T, R, C> get(cv::Matx<T, R, C>& data, mo::_counter_<0>){ return data.val; }
            static constexpr MatrixAdapter<const T, R, C> get(const cv::Matx<T, R, C>& data, mo::_counter_<0>){ return data.val; }
            static constexpr const char* getName(mo::_counter_<0>) { return "data"; }
        };

        template<class T, int R>
        struct ReflectData<cv::Vec<T, R>>
        {
            static constexpr int N = 1;
            static constexpr int IS_SPECIALIZED = true;
            static constexpr ArrayAdapter<T, R> get(cv::Vec<T, R>& data, mo::_counter_<0>){ return data.val; }
            static constexpr ArrayAdapter<const T, R> get(const cv::Vec<T, R>& data, mo::_counter_<0>){ return data.val; }
            static constexpr const char* getName(mo::_counter_<0>) { return "data"; }
        };

        template<class T>
        struct ReflectData<cv::Scalar_<T>>
        {
            static constexpr int N = 1;
            static constexpr int IS_SPECIALIZED = true;
            static constexpr ArrayAdapter<T, 4> get(cv::Scalar_<T>& data, mo::_counter_<0>){ return &data.val[0]; }
            static constexpr ArrayAdapter<const T, 4> get(const cv::Scalar_<T>& data, mo::_counter_<0>){ return &data.val[0]; }
            static constexpr const char* getName(mo::_counter_<0>) { return "data"; }
        };

    }
}

static_assert(mo::reflect::ReflectData<cv::Scalar>::IS_SPECIALIZED, "Specialization not working for cv::Scalar");
static_assert(mo::reflect::ReflectData<cv::Vec2f>::IS_SPECIALIZED, "Specialization not working for cv::Vec2f");
static_assert(mo::reflect::ReflectData<cv::Vec2b>::IS_SPECIALIZED, "Specialization not working for cv::Vec2b");
static_assert(mo::reflect::ReflectData<cv::Vec3f>::IS_SPECIALIZED, "Specialization not working for cv::Vec3f");
static_assert(mo::reflect::ReflectData<cv::Vec3b>::IS_SPECIALIZED, "Specialization not working for cv::Vec3b");

INSTANTIATE_META_PARAM(cv::Scalar);
INSTANTIATE_META_PARAM(cv::Vec2f);
INSTANTIATE_META_PARAM(cv::Vec3f);
INSTANTIATE_META_PARAM(cv::Vec2b);
INSTANTIATE_META_PARAM(cv::Vec3b);
INSTANTIATE_META_PARAM(std::vector<cv::Vec3b>);
typedef std::map<std::string, cv::Vec3b> ClassColormap_t;
INSTANTIATE_META_PARAM(ClassColormap_t);
#endif
