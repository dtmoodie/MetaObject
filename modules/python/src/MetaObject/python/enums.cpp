#include <RuntimeObjectSystem/IObjectInfo.h>
#include <boost/python.hpp>

namespace mo
{
    void setupEnums(const std::string& module_name)
    {
        boost::python::object enums_module(boost::python::handle<>(
            boost::python::borrowed(PyImport_AddModule((std::string(module_name) + ".enums").c_str()))));
        boost::python::scope().attr("enums") = enums_module;
        boost::python::scope enums_scope = enums_module;
        boost::python::enum_<IObjectInfo::Verbosity>("IObjectInfoVerbosity")
            .value("INFO", IObjectInfo::INFO)
            .value("DEBUG", IObjectInfo::DEBUG)
            .value("RCC", IObjectInfo::RCC);
    }
}