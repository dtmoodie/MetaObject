#include "MetaObject/core/metaobject_config.hpp"
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
#if MO_HAVE_PYTHON
#include <boost/python.hpp>
    namespace python
    {
        template <>
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
        template <>
        inline void convertFromPython(const boost::python::object& obj, ReadFile& result)
        {
            boost::python::extract<std::string> extractor(obj);
            result = extractor();
        }

        template <>
        inline void convertFromPython(const boost::python::object& obj, WriteFile& result)
        {
            boost::python::extract<std::string> extractor(obj);
            result = extractor();
        }

        template <>
        inline void convertFromPython(const boost::python::object& obj, WriteDirectory& result)
        {
            boost::python::extract<std::string> extractor(obj);
            result = extractor();
        }

        template <>
        inline void convertFromPython(const boost::python::object& obj, ReadDirectory& result)
        {
            boost::python::extract<std::string> extractor(obj);
            result = extractor();
        }

        template <>
        inline boost::python::object convertToPython(const ReadFile& file)
        {
            return boost::python::object(file.string());
        }

        template <>
        inline boost::python::object convertToPython(const WriteFile& file)
        {
            return boost::python::object(file.string());
        }

        template <>
        inline boost::python::object convertToPython(const ReadDirectory& file)
        {
            return boost::python::object(file.string());
        }

        template <>
        inline boost::python::object convertToPython(const WriteDirectory& file)
        {
            return boost::python::object(file.string());
        }
    }
#endif
}

namespace mo
{
    namespace MetaParams
    {
        void instMOTypes(SystemTable* table)
        {
            INSTANTIATE_META_PARAM(ReadFile);
            INSTANTIATE_META_PARAM(WriteFile);
            INSTANTIATE_META_PARAM(ReadDirectory);
            INSTANTIATE_META_PARAM(WriteDirectory);
            INSTANTIATE_META_PARAM(EnumParam);
        }
    }
}

EXTERN_TYPE(ReadFile);
EXTERN_TYPE(WriteFile);
EXTERN_TYPE(ReadDirectory);
EXTERN_TYPE(WriteDirectory);
EXTERN_TYPE(EnumParam);
