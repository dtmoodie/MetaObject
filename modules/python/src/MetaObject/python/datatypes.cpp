#include <boost/python.hpp>

#include "MetaObject/python/DataConverter.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/ICoordinateSystem.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/object/IMetaObject.hpp"

#include <RuntimeObjectSystem/IObjectInfo.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
namespace mo
{
    std::string printParam(const mo::ParamBase* param)
    {
        std::stringstream ss;
        ss << param->getTreeName();
        ss << " [" << mo::Demangle::typeToName(param->getTypeInfo()) << "]";
        auto ts = param->getTimestamp();
        if (ts)
        {
            ss << " " << *ts;
        }
        auto fn = param->getFrameNumber();
        if (fn != -1)
        {
            ss << " " << fn;
        }
        auto cs = param->getCoordinateSystem();
        if (cs)
        {
            ss << " " << cs->getName();
        }
        return ss.str();
    }
    
    std::string printStringVec(const std::vector<std::string>& strs)
    {
        std::stringstream ss;
        for (const auto& itr : strs) {
            ss << itr << "\n";
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
        return boost::python::object();
    }

    bool setData(mo::IParam* param, const boost::python::object& obj)
    {
        auto setter = mo::python::DataConverterRegistry::instance()->getSetter(param->getTypeInfo());
        if (setter)
        {
            return setter(param, obj);
        }
        return false;
    }

    std::string getDataTypeName(const mo::ParamBase* param)
    {
        return mo::Demangle::typeToName(param->getTypeInfo());
    }

    void setupDataTypes(const std::string& module_name)
    {
        boost::python::object datatype_module(boost::python::handle<>(boost::python::borrowed(PyImport_AddModule((module_name + ".datatypes").c_str()))));
        boost::python::scope().attr("datatypes") = datatype_module;
        boost::python::scope datatype_scope = datatype_module;

        boost::python::class_<ParamBase, boost::noncopyable>("ParamBase", boost::python::no_init)
            .def("getName", &ParamBase::getTreeName)
            .def("getType", &getDataTypeName)
            .add_property("data", &getData, &setData);

        boost::python::class_<std::vector<ParamBase*>> param_vec("ParamVec", boost::python::no_init);

        boost::python::class_<InputParam, boost::python::bases<ParamBase>, boost::noncopyable> input_param("InputParam", boost::python::no_init);

        boost::python::class_<std::vector<InputParam*>> input_param_vec("InputParamVec", boost::python::no_init);

        //boost::python::class_<ICoordinateSystem, std::shared_ptr<ICoordinateSystem>, boost::noncopyable> cs_obj("ICoordinateSystem", boost::python::no_init);
        //cs_obj.def("getName", &ICoordinateSystem::getName, boost::python::return_value_policy<boost::python::reference_existing_object>());

        boost::python::class_<Context, boost::noncopyable>("Context", boost::python::no_init)
            .add_property("getName", &Context::getName)
            .add_property("thread_id", &Context::getThreadId);

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