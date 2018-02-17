#include "MetaObject/metaparams/MetaParamsInclude.hpp"
#include <MetaObject/params/MetaParam.hpp>

#include "MetaObject/types/file_types.hpp"
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
    namespace python
    {

        inline void convertFromPython(const boost::python::object& obj, EnumParam& param)
        {
            boost::python::extract<std::string> str_ext(obj);
            if (str_ext.check())
            {
                auto string = str_ext();
                auto itr = std::find(param.enumerations.begin(), param.enumerations.end(), string);
                if (itr != param.enumerations.end())
                {
                    param.current_selection = itr - param.enumerations.begin();
                }
            }
            else
            {
                boost::python::extract<int> int_ext(obj);
                MO_ASSERT(int_ext.check());
                int val = int_ext();
                param.current_selection = static_cast<size_t>(val);
            }
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

INSTANTIATE_META_PARAM(ReadFile);
INSTANTIATE_META_PARAM(WriteFile);
INSTANTIATE_META_PARAM(ReadDirectory);
INSTANTIATE_META_PARAM(WriteDirectory);
INSTANTIATE_META_PARAM(EnumParam);
