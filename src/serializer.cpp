#include "MetaObject/IO/Serializer.hpp"
#include <MetaObject/IMetaObject.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <map>
using namespace mo;
std::map<std::string, SerializerFactory::BinarySerialize_f> _binary_serialization_functions;
std::map<std::string, SerializerFactory::BinaryDeSerialize_f> _binary_deserialization_functions;

void SerializerFactory::RegisterSerializationFunction(const char* obj_type, BinarySerialize_f f)
{
    _binary_serialization_functions[obj_type] = f;
}

void SerializerFactory::RegisterDeSerializationFunction(const char* obj_type, BinaryDeSerialize_f f)
{
    _binary_deserialization_functions[obj_type] = f;
}

void SerializerFactory::Serialize(IMetaObject* obj, std::ostream& os, SerializationType type)
{
    cereal::PortableBinaryOutputArchive ar(os);
    auto func_itr = _binary_serialization_functions.find(obj->GetTypeName());
    if(func_itr != _binary_serialization_functions.end())
    {
        func_itr->second(obj, ar);
    }
}

void SerializerFactory::DeSerialize(IMetaObject* obj, std::istream& is, SerializationType type)
{
    cereal::PortableBinaryInputArchive ar(is);
    auto func_itr = _binary_deserialization_functions.find(obj->GetTypeName());
    if (func_itr != _binary_deserialization_functions.end())
    {
        func_itr->second(obj, ar);
    }
}

IMetaObject* SerializerFactory::DeSerialize(std::istream& os, SerializationType type)
{
    return nullptr;
}