#include "MetaObject/metaparams/MetaParamsInclude.hpp"
#include <MetaObject/params/MetaParam.hpp>

#include "MetaObject/params/Types.hpp"
#include <MetaObject/params/AccessToken.hpp>
#include <MetaObject/params/ITAccessibleParam.hpp>
#include <boost/lexical_cast.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

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

using namespace mo;

namespace mo
{
    namespace reflect
    {
        template <class T>
        struct ReflectData<
            T,
            typename std::enable_if<std::is_same<ReadFile, T>::value || std::is_same<WriteFile, T>::value ||
                                        std::is_same<ReadDirectory, T>::value || std::is_same<WriteDirectory, T>::value,
                                    void>::type>
        {
            static constexpr int N = 1;
            static constexpr bool IS_SPECIALIZED = true;
            static std::string get(const T& data, mo::_counter_<0>) { return data.string(); }
            static constexpr const char* getName(mo::_counter_<0>) { return "path"; }
        };

        template <int I, class T>
        static constexpr inline
            typename std::enable_if<std::is_same<ReadFile, T>::value || std::is_same<WriteFile, T>::value ||
                                        std::is_same<ReadDirectory, T>::value || std::is_same<WriteDirectory, T>::value,
                                    std::string>::type
            getValue(const T& data)
        {
            return data.string();
        }

        template <int I, class T>
        static constexpr inline
            typename std::enable_if<std::is_same<ReadFile, T>::value || std::is_same<WriteFile, T>::value ||
                                        std::is_same<ReadDirectory, T>::value || std::is_same<WriteDirectory, T>::value,
                                    std::string>::type
            setValue(const T& data, const std::string& value)
        {
            data = T(value);
        }

        template <int I>
        std::string get(const ReadFile& data)
        {
            return data.string();
        }
        template <int I>
        std::string get(const WriteFile& data)
        {
            return data.string();
        }
        template <int I>
        std::string get(const ReadDirectory& data)
        {
            return data.string();
        }
        template <int I>
        std::string get(const WriteDirectory& data)
        {
            return data.string();
        }

        template <int I>
        std::string get(ReadFile& data)
        {
            return data.string();
        }
        template <int I>
        std::string get(WriteFile& data)
        {
            return data.string();
        }
        template <int I>
        std::string get(ReadDirectory& data)
        {
            return data.string();
        }
        template <int I>
        std::string get(WriteDirectory& data)
        {
            return data.string();
        }
    }
}

namespace std
{
    template <class T>
    ostream& operator<<(ostream& os, const std::vector<T>& data)
    {
        os << '[';
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (i != 0)
                os << ',';
            os << data[i];
        }
        os << ']';
        return os;
    }
}

// INSTANTIATE_META_PARAM(ReadFile);
// INSTANTIATE_META_PARAM(WriteFile);
// INSTANTIATE_META_PARAM(ReadDirectory);
// INSTANTIATE_META_PARAM(WriteDirectory);
INSTANTIATE_META_PARAM(EnumParam);
