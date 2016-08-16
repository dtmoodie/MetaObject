#include "MetaObject/Parameters/IO/Policy.hpp"
#include "MetaObject/Detail/TypeInfo.h"

#include <map>

using namespace mo;
struct SerializationFunctionRegistry::impl
{
    std::map<TypeInfo, SerializeBinary_f > _binary_serialize_map;
    std::map<TypeInfo, DeSerializeBinary_f > _binary_deserialize_map;

    std::map<TypeInfo, SerializeXml_f > _xml_serialize_map;
    std::map<TypeInfo, DeSerializeXml_f > _xml_deserialize_map;
};

SerializationFunctionRegistry::SerializationFunctionRegistry()
{
    _pimpl = new impl();    
}

SerializationFunctionRegistry::~SerializationFunctionRegistry()
{
    delete _pimpl;
}

SerializationFunctionRegistry* SerializationFunctionRegistry::Instance()
{
    static SerializationFunctionRegistry g_inst;
    return &g_inst;
}

SerializationFunctionRegistry::SerializeBinary_f SerializationFunctionRegistry::GetBinarySerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_binary_serialize_map.find(type);
    if(itr != _pimpl->_binary_serialize_map.end())
    {
        return itr->second;
    }
    return SerializeBinary_f();
}

SerializationFunctionRegistry::DeSerializeBinary_f SerializationFunctionRegistry::GetBinaryDeSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_binary_deserialize_map.find(type);
    if (itr != _pimpl->_binary_deserialize_map.end())
    {
        return itr->second;
    }
    return DeSerializeBinary_f ();
}

SerializationFunctionRegistry::SerializeXml_f SerializationFunctionRegistry::GetXmlSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_xml_serialize_map.find(type);
    if (itr != _pimpl->_xml_serialize_map.end())
    {
        return itr->second;
    }
    return SerializeXml_f();
}

SerializationFunctionRegistry::DeSerializeXml_f SerializationFunctionRegistry::GetXmlDeSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_xml_deserialize_map.find(type);
    if (itr != _pimpl->_xml_deserialize_map.end())
    {
        return itr->second;
    }
    return DeSerializeXml_f();
}

void SerializationFunctionRegistry::SetBinarySerializationFunction(const TypeInfo& type, SerializeBinary_f f)
{
    _pimpl->_binary_serialize_map[type] = f;
}

void SerializationFunctionRegistry::SetBinaryDeSerializationFunction(const TypeInfo& type, DeSerializeBinary_f f)
{
    _pimpl->_binary_deserialize_map[type] = f;
}

void SerializationFunctionRegistry::SetXmlSerializationFunction(const TypeInfo& type, SerializeXml_f f)
{
    _pimpl->_xml_serialize_map[type] = f;
}

void SerializationFunctionRegistry::SetXmlDeSerializationFunction(const TypeInfo& type, DeSerializeXml_f f)
{
    _pimpl->_xml_deserialize_map[type] = f;
}