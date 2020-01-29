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
        // TODO boost python enum from ct
        /*boost::python::enum_<mo::BufferFlags>("BufferFlags")
            .value("Default", mo::BufferFlags::DEFAULT)
            .value("CircularBuffer", mo::BufferFlags::CIRCULAR_BUFFER)
            .value("Map", mo::BufferFlags::MAP_BUFFER)
            .value("StreamBuffer", mo::BufferFlags::STREAM_BUFFER)
            .value("BlockingStreamBuffer", mo::BufferFlags::BLOCKING_STREAM_BUFFER)
            .value("NNStreamBuffer", mo::BufferFlags::NEAREST_NEIGHBOR_BUFFER)
            .value("Queue", mo::BufferFlags::QUEUE_BUFFER)
            .value("BlockingQueue", mo::BufferFlags::BLOCKING_QUEUE_BUFFER)
            .value("DroppingQueue", mo::BufferFlags::DROPPING_QUEUE_BUFFER)
            .value("ForceBufferedConnection", mo::BufferFlags::FORCE_BUFFERED)
            .value("ForceDirectConnection", mo::BufferFlags::FORCE_DIRECT);

        boost::python::enum_<mo::UpdateFlags>("ParamUpdateFlags")
            .value("ValueUpdated", mo::UpdateFlags::kVALUE_UPDATED)
            .value("InputSet", mo::UpdateFlags::kINPUT_SET)
            .value("InputCleared", mo::UpdateFlags::kINPUT_CLEARED)
            .value("InputUpdated", mo::UpdateFlags::kINPUT_UPDATED)
            .value("BufferUpdated", mo::UpdateFlags::kBUFFER_UPDATED);*/
    }
}
