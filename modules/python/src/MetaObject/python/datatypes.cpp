#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/core/TypeTable.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/params/ICoordinateSystem.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/python/DataConverter.hpp"
#include <boost/python.hpp>

#include <RuntimeObjectSystem/IObjectInfo.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace mo
{
    std::string printParam(const mo::ParamBase* param)
    {
        std::stringstream ss;
        param->print(ss);
        return ss.str();
    }

    std::string printInputParam(const mo::InputParam* param)
    {
        std::stringstream ss;
        ss << printParam(param);
        ss << "\n";
        auto input = param->getInputParam();
        if (input)
        {
            ss << printParam(input);
        }
        return ss.str();
    }

    std::string printStringVec(const std::vector<std::string>& strs)
    {
        std::stringstream ss;
        for (const auto& itr : strs)
        {
            ss << itr << "\n";
        }
        return ss.str();
    }

    std::string printTypes()
    {
        auto types = mo::python::DataConverterRegistry::instance()->listConverters();
        std::stringstream ss;
        for (auto& type : types)
        {
            ss << mo::Demangle::typeToName(type) << "\n";
        }
        return ss.str();
    }

    boost::python::object getData(const mo::ParamBase* param)
    {
        auto getter = mo::python::DataConverterRegistry::instance()->getGetter(param->getTypeInfo());
        if (getter)
        {
            return getter(param);
        }
        MO_LOG(trace) << "Accessor function not found for for " << mo::Demangle::typeToName(param->getTypeInfo())
                      << " Available converters:\n"
                      << printTypes();
        return boost::python::object(std::string("No to python converter registered for " +
                                                 mo::TypeTable::instance().typeToName(param->getTypeInfo())));
    }

    bool setData(mo::IParam* param, const boost::python::object& obj)
    {
        auto setter = mo::python::DataConverterRegistry::instance()->getSetter(param->getTypeInfo());
        if (setter)
        {
            return setter(param, obj);
        }
        MO_LOG(trace) << "Setter function not found for " << mo::Demangle::typeToName(param->getTypeInfo());
        return false;
    }

    std::string getDataTypeName(const mo::ParamBase* param) { return mo::Demangle::typeToName(param->getTypeInfo()); }

    /*void setDefaultBufferType(const std::shared_ptr<Context>& source,
                              const std::shared_ptr<Context>& dest,
                              mo::ParamType type)
    {
        setDefaultBufferType(source.get(), dest.get(), type);
    }*/

    void setupDataTypes(const std::string& module_name)
    {
        boost::python::object datatype_module(
            boost::python::handle<>(boost::python::borrowed(PyImport_AddModule((module_name + ".datatypes").c_str()))));
        boost::python::scope().attr("datatypes") = datatype_module;
        boost::python::scope datatype_scope = datatype_module;

        boost::python::class_<ParamBase, boost::noncopyable>("ParamBase", boost::python::no_init)
            .def("getName", &ParamBase::getTreeName)
            .def("getType", &getDataTypeName)
            .def("__repr__", &printParam)
            //.def("getContext", &ParamBase::getContext, boost::python::return_internal_reference<>())
            .add_property("data", &getData, &setData);

        boost::python::class_<IParam, boost::noncopyable, boost::python::bases<ParamBase>>("IParam",
                                                                                           boost::python::no_init);

        boost::python::implicitly_convertible<IParam*, ParamBase*>();

        boost::python::class_<std::vector<ParamBase*>> param_vec("ParamVec", boost::python::no_init);
        param_vec.def(boost::python::vector_indexing_suite<std::vector<ParamBase*>>());

        boost::python::class_<std::vector<IParam*>> iparam_vec("IParamVec", boost::python::no_init);
        iparam_vec.def(boost::python::vector_indexing_suite<std::vector<IParam*>>());

        boost::python::class_<InputParam, InputParam*, boost::python::bases<ParamBase>, boost::noncopyable> input_param(
            "InputParam", boost::python::no_init);
        input_param.def("__repr__", &printInputParam);

        boost::python::implicitly_convertible<InputParam*, IParam*>();

        boost::python::class_<std::vector<InputParam*>> input_param_vec("InputParamVec", boost::python::no_init);
        input_param_vec.def(boost::python::vector_indexing_suite<std::vector<InputParam*>>());

        // boost::python::class_<ICoordinateSystem, std::shared_ptr<ICoordinateSystem>, boost::noncopyable>
        // cs_obj("ICoordinateSystem", boost::python::no_init);
        // cs_obj.def("getName", &ICoordinateSystem::getName,
        // boost::python::return_value_policy<boost::python::reference_existing_object>());

        boost::python::class_<Context, std::shared_ptr<Context>, boost::noncopyable>("Context", boost::python::no_init)
            .add_property("name", &Context::getName)
            .add_property("thread_id", &Context::getThreadId);

        boost::python::def("setDefaultBufferType", &setDefaultBufferType);

        boost::python::class_<IMetaObject, boost::noncopyable> metaobject("IMetaObject", boost::python::no_init);

        boost::python::class_<std::vector<std::string>>("StringVec")
            .def(boost::python::vector_indexing_suite<std::vector<std::string>>())
            .def("__repr__", &printStringVec);

        boost::python::class_<IObjectInfo, boost::noncopyable>("IObjectInfo", boost::python::no_init)
            .def("getInterfaceId", &IObjectInfo::GetInterfaceId)
            .def("getInterfaceName", &IObjectInfo::GetInterfaceName)
            .def("__repr__", &IObjectInfo::Print, (boost::python::arg("verbosity") = IObjectInfo::INFO));
    }
}
