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
        boost::python::enum_<mo::BufferFlags>("BufferFlags")
            .value("Default", mo::BufferFlags::Default_e)
            .value("TParam", mo::BufferFlags::TParam_e)
            .value("CircularBuffer", mo::BufferFlags::CircularBuffer_e)
            .value("ConstMap", mo::BufferFlags::ConstMap_e)
            .value("Map", mo::BufferFlags::Map_e)
            .value("StreamBuffer", mo::BufferFlags::StreamBuffer_e)
            .value("BlockingStreamBuffer", mo::BufferFlags::BlockingStreamBuffer_e)
            .value("NNStreamBuffer", mo::BufferFlags::NNStreamBuffer_e)
            .value("Queue", mo::BufferFlags::Queue_e)
            .value("BlockingQueue", mo::BufferFlags::BlockingQueue_e)
            .value("DroppingQueue", mo::BufferFlags::DroppingQueue_e)
            .value("ForceBufferedConnection", mo::BufferFlags::ForceBufferedConnection_e)
            .value("ForceDirectConnection", mo::BufferFlags::ForceDirectConnection_e);

        boost::python::enum_<mo::UpdateFlags>("ParamUpdateFlags")
                .value("ValueUpdated", mo::ValueUpdated_e)
                .value("InputSet", mo::InputSet_e)
                .value("InputCleared", mo::InputCleared_e)
                .value("InputUpdated", mo::InputUpdated_e)
                .value("BufferUpdated", mo::BufferUpdated_e);
    }
}
