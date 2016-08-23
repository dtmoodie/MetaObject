#include "MetaObject/IO/Serializer.hpp"
#include <MetaObject/IMetaObject.hpp>
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <map>
using namespace mo;
std::map<std::string, SerializerFactory::BinarySerialize_f> _binary_serialization_functions;
std::map<std::string, SerializerFactory::BinaryDeSerialize_f> _binary_deserialization_functions;

std::map<std::string, SerializerFactory::XMLSerialize_f> _xml_serialization_functions;
std::map<std::string, SerializerFactory::XMLDeSerialize_f> _xml_deserialization_functions;

void SerializerFactory::RegisterSerializationFunctionBinary(const char* obj_type, BinarySerialize_f f)
{
    _binary_serialization_functions[obj_type] = f;
}

void SerializerFactory::RegisterDeSerializationFunctionBinary(const char* obj_type, BinaryDeSerialize_f f)
{
    _binary_deserialization_functions[obj_type] = f;
}

void SerializerFactory::RegisterSerializationFunctionXML(const char* obj_type, XMLSerialize_f f)
{
    _xml_serialization_functions[obj_type] = f;
}

void SerializerFactory::RegisterDeSerializationFunctionXML(const char* obj_type, XMLDeSerialize_f f)
{
    _xml_deserialization_functions[obj_type] = f;
}

SerializerFactory::BinarySerialize_f SerializerFactory::GetSerializationFunctionBinary(const char* obj_type)
{
	auto itr = _binary_serialization_functions.find(obj_type);
	if (itr != _binary_serialization_functions.end())
	{
		return itr->second;
	}
	return SerializerFactory::BinarySerialize_f();
}

SerializerFactory::BinaryDeSerialize_f SerializerFactory::GetDeSerializationFunctionBinary(const char* obj_type)
{
	auto itr = _binary_deserialization_functions.find(obj_type);
	if (itr != _binary_deserialization_functions.end())
	{
		return itr->second;
	}
	return SerializerFactory::BinaryDeSerialize_f();
}

SerializerFactory::XMLSerialize_f SerializerFactory::GetSerializationFunctionXML(const char* obj_type)
{
	auto itr = _xml_serialization_functions.find(obj_type);
	if (itr != _xml_serialization_functions.end())
	{
		return itr->second;
	}
	return SerializerFactory::XMLSerialize_f();
}

SerializerFactory::XMLDeSerialize_f SerializerFactory::GetDeSerializationFunctionXML(const char* obj_type)
{
	auto itr = _xml_deserialization_functions.find(obj_type);
	if (itr != _xml_deserialization_functions.end())
	{
		return itr->second;
	}
	return SerializerFactory::XMLDeSerialize_f();
}


void SerializerFactory::Serialize(const rcc::shared_ptr<IMetaObject>& obj, std::ostream& os, SerializationType type)
{
    if(type == Binary_e)
    {
        cereal::BinaryOutputArchive ar(os);
        ar(cereal::make_nvp("ObjectType", std::string(obj->GetTypeName())));
        auto func_itr = _binary_serialization_functions.find(obj->GetTypeName());
        if (func_itr != _binary_serialization_functions.end())
        {
            func_itr->second(obj.Get(), ar);
        }
    }else if(type == xml_e)
    {
        cereal::XMLOutputArchive ar(os);
        std::string ObjectType(obj->GetTypeName());
        ar(CEREAL_NVP(ObjectType));
        auto func_itr = _xml_serialization_functions.find(obj->GetTypeName());
        if (func_itr != _xml_serialization_functions.end())
        {
            func_itr->second(obj.Get(), ar);
        }
    }
}

void SerializerFactory::DeSerialize(IMetaObject* obj, std::istream& is, SerializationType type)
{
    
    if(type == Binary_e)
    {
        cereal::BinaryInputArchive ar(is);
        std::string ObjectType;
        ar(CEREAL_NVP(ObjectType));
        auto func_itr = _binary_deserialization_functions.find(obj->GetTypeName());
        if (func_itr != _binary_deserialization_functions.end())
        {
            func_itr->second(obj, ar);
        }
    }else if(type == xml_e)
    {
        cereal::XMLInputArchive ar(is);
        std::string ObjectType;
        ar(CEREAL_NVP(ObjectType));
        ASSERT_EQ(ObjectType, obj->GetTypeName());
        auto func_itr = _xml_deserialization_functions.find(obj->GetTypeName());
        if (func_itr != _xml_deserialization_functions.end())
        {
            func_itr->second(obj, ar);
        }
    }
}

rcc::shared_ptr<IMetaObject> SerializerFactory::DeSerialize(std::istream& os, SerializationType type)
{
    IMetaObject* obj = nullptr;
    if (type == Binary_e)
    {
        cereal::BinaryInputArchive ar(os);
        std::string ObjectType;
        ar(CEREAL_NVP(ObjectType));
        if(ObjectType.size())
        {
            obj = MetaObjectFactory::Instance()->Create(ObjectType.c_str());
            auto func_itr = _binary_deserialization_functions.find(obj->GetTypeName());
            if (func_itr != _binary_deserialization_functions.end())
            {
                func_itr->second(obj, ar);
            }
        }
    }
    else if (type == xml_e)
    {
        cereal::XMLInputArchive ar(os);
        std::string ObjectType;
        ar(CEREAL_NVP(ObjectType));
        obj = MetaObjectFactory::Instance()->Create(ObjectType.c_str());
        if(obj)
        {
            auto func_itr = _xml_deserialization_functions.find(obj->GetTypeName());
            if (func_itr != _xml_deserialization_functions.end())
            {
                func_itr->second(obj, ar);
            }
        }
    }
    return obj;
}