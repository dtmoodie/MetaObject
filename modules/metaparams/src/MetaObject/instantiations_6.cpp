#include "common.hpp"

#include <MetaObject/runtime_reflection/visitor_traits/string.hpp>

#include <MetaObject/runtime_reflection/visitor_traits/filesystem.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>

#include <MetaObject/runtime_reflection/StructTraits.hpp>

#include "MetaObject/core/metaobject_config.hpp"

#include "MetaObject/types/file_types.hpp"
#include <MetaObject/params/AccessToken.hpp>

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

using namespace mo;

#ifdef MO_HAVE_PYTHON2
#include <boost/python.hpp>
namespace ct
{

    template <>
    inline bool convertFromPython(const boost::python::object& obj, EnumParam& param)
    {
        boost::python::extract<std::string> str_ext(obj);
        if (str_ext.check())
        {
            auto string = str_ext();
            auto itr = std::find(param.enumerations.begin(), param.enumerations.end(), string);
            if (itr != param.enumerations.end())
            {
                param.current_selection = itr - param.enumerations.begin();
                return true;
            }
            return false;
        }
        boost::python::extract<int> int_ext(obj);

        if (int_ext.check())
        {
            int val = int_ext();
            param.current_selection = static_cast<size_t>(val);
            return true;
        }

        return false;
    }

    template <>
    inline bool convertFromPython(const boost::python::object& obj, ReadFile& result)
    {
        boost::python::extract<std::string> extractor(obj);
        result = extractor();
        return true;
    }

    template <>
    inline bool convertFromPython(const boost::python::object& obj, WriteFile& result)
    {
        boost::python::extract<std::string> extractor(obj);
        result = extractor();
        return true;
    }

    template <>
    inline bool convertFromPython(const boost::python::object& obj, WriteDirectory& result)
    {
        boost::python::extract<std::string> extractor(obj);
        result = extractor();
        return true;
    }

    template <>
    inline bool convertFromPython(const boost::python::object& obj, ReadDirectory& result)
    {
        boost::python::extract<std::string> extractor(obj);
        result = extractor();
        return true;
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
} // namespace ct

#endif

namespace mo
{
    namespace metaparams
    {
        void instMOTypes(SystemTable* table)
        {
            registerTrait<ReadFile>();
            registerTrait<WriteFile>();
            registerTrait<ReadDirectory>();
            registerTrait<WriteDirectory>();
            registerTrait<EnumParam>();
        }
    } // namespace metaparams
} // namespace mo
