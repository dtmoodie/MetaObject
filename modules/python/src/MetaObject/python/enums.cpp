#include <MetaObject/core/detail/Enums.hpp>
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
        boost::python::enum_<mo::ParamType>("ParamType")
            .value("Default", mo::ParamType::Default_e)
            .value("TParam", mo::ParamType::TParam_e)
            .value("CircularBuffer", mo::ParamType::CircularBuffer_e)
            .value("ConstMap", mo::ParamType::ConstMap_e)
            .value("Map", mo::ParamType::Map_e)
            .value("StreamBuffer", mo::ParamType::StreamBuffer_e)
            .value("BlockingStreamBuffer", mo::ParamType::BlockingStreamBuffer_e)
            .value("NNStreamBuffer", mo::ParamType::NNStreamBuffer_e)
            .value("Queue", mo::ParamType::Queue_e)
            .value("BlockingQueue", mo::ParamType::BlockingQueue_e)
            .value("DroppingQueue", mo::ParamType::DroppingQueue_e)
            .value("ForceBufferedConnection", mo::ParamType::ForceBufferedConnection_e)
            .value("ForceDirectConnection", mo::ParamType::ForceDirectConnection_e);
    }
}
