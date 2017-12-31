#include "MetaObject/logging/logging.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/serialization/SerializationFactory.hpp"
#include "MetaObject/serialization/Serializer.hpp"
#include <MetaObject/object/IMetaObject.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>

#include <list>
#include <map>
using namespace mo;
std::map<std::string, SerializerFactory::BinarySerialize_f> _binary_serialization_functions;
std::map<std::string, SerializerFactory::BinaryDeSerialize_f> _binary_deserialization_functions;

std::map<std::string, SerializerFactory::XMLSerialize_f> _xml_serialization_functions;
std::map<std::string, SerializerFactory::XMLDeSerialize_f> _xml_deserialization_functions;

std::map<std::string, SerializerFactory::JSONSerialize_f> _json_serialization_functions;
std::map<std::string, SerializerFactory::JSONDeSerialize_f> _json_deserialization_functions;

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

void SerializerFactory::RegisterSerializationFunctionJSON(const char* obj_type, JSONSerialize_f f)
{
    _json_serialization_functions[obj_type] = f;
}
void SerializerFactory::RegisterDeSerializationFunctionJSON(const char* obj_type, JSONDeSerialize_f f)
{
    _json_deserialization_functions[obj_type] = f;
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

SerializerFactory::XMLSerialize_f SerializerFactory::getSerializationFunctionXML(const char* obj_type)
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

SerializerFactory::JSONSerialize_f SerializerFactory::GetSerializationFunctionJSON(const char* obj_type)
{
    auto itr = _json_serialization_functions.find(obj_type);
    if (itr != _json_serialization_functions.end())
    {
        return itr->second;
    }
    return SerializerFactory::JSONSerialize_f();
}

SerializerFactory::JSONDeSerialize_f SerializerFactory::GetDeSerializationFunctionJSON(const char* obj_type)
{
    auto itr = _json_deserialization_functions.find(obj_type);
    if (itr != _json_deserialization_functions.end())
    {
        return itr->second;
    }
    return SerializerFactory::JSONDeSerialize_f();
}

void SerializerFactory::Serialize(const rcc::shared_ptr<IMetaObject>& obj, std::ostream& os, SerializationType type)
{
    if (type == Binary_e)
    {
        cereal::BinaryOutputArchive ar(os);
        ar(cereal::make_nvp("ObjectType", std::string(obj->GetTypeName())));
        auto func_itr = _binary_serialization_functions.find(obj->GetTypeName());
        if (func_itr != _binary_serialization_functions.end())
        {
            func_itr->second(obj.get(), ar);
        }
    }
    else if (type == xml_e)
    {
        cereal::XMLOutputArchive ar(os);
        std::string ObjectType(obj->GetTypeName());
        ar(CEREAL_NVP(ObjectType));
        auto func_itr = _xml_serialization_functions.find(obj->GetTypeName());
        if (func_itr != _xml_serialization_functions.end())
        {
            func_itr->second(obj.get(), ar);
        }
    }
}

void SerializerFactory::DeSerialize(IMetaObject* obj, std::istream& is, SerializationType type)
{

    if (type == Binary_e)
    {
        cereal::BinaryInputArchive ar(is);
        std::string ObjectType;
        ar(CEREAL_NVP(ObjectType));
        auto func_itr = _binary_deserialization_functions.find(obj->GetTypeName());
        if (func_itr != _binary_deserialization_functions.end())
        {
            func_itr->second(obj, ar);
        }
    }
    else if (type == xml_e)
    {
        cereal::XMLInputArchive ar(is);
        std::string ObjectType;
        ar(CEREAL_NVP(ObjectType));
        MO_ASSERT_EQ(ObjectType, obj->GetTypeName());
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
        if (ObjectType.size())
        {
            obj = MetaObjectFactory::instance()->create(ObjectType.c_str());
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
        obj = MetaObjectFactory::instance()->create(ObjectType.c_str());
        if (obj)
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

bool mo::Serialize(cereal::BinaryOutputArchive& ar, const IMetaObject* obj)
{
    if (auto func = SerializerFactory::GetSerializationFunctionBinary(obj->GetTypeName()))
    {
        func(obj, ar);
    }
    else
    {
        MO_LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->getParams();
        std::string type = obj->GetTypeName();
        ar(cereal::make_nvp("TypeName", type));
        for (auto& param : params) {
            auto func1 = SerializationFactory::instance()->getBinarySerializationFunction(param->getTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    MO_LOG(debug) << "Unable to serialize " << param->getTreeName();
                }
            }
            else
            {
                MO_LOG(debug) << "No serialization function found for " << param->getTypeInfo().name();
            }
        }
    }
    return true;
}

bool mo::DeSerialize(cereal::BinaryInputArchive& ar, IMetaObject* obj)
{
    (void)ar;
    (void)obj;
    return false;
}

bool mo::Serialize(cereal::XMLOutputArchive& ar, const IMetaObject* obj)
{
    if (auto func = SerializerFactory::getSerializationFunctionXML(obj->GetTypeName()))
    {
        func(obj, ar);
        return true;
    }
    else
    {
        MO_LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->getParams();
        std::string type = obj->GetTypeName();
        ar(cereal::make_nvp("TypeName", type));
        for (auto& param : params) {
            auto func1 = SerializationFactory::instance()->getXmlSerializationFunction(param->getTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    MO_LOG(debug) << "Unable to serialize " << param->getTreeName();
                }
            }
            else
            {
                MO_LOG(debug) << "No serialization function found for " << param->getTypeInfo().name();
            }
        }
        return true;
    }
}

bool mo::DeSerialize(cereal::XMLInputArchive& ar, IMetaObject* obj)
{
    (void)ar;
    (void)obj;
    return false;
}

bool mo::Serialize(cereal::JSONOutputArchive& ar, const IMetaObject* obj)
{
    if (auto func = SerializerFactory::GetSerializationFunctionJSON(obj->GetTypeName()))
    {
        func(obj, ar);
        return true;
    }
    else
    {
        MO_LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->getParams();
        std::string type = obj->GetTypeName();
        ObjectId id = obj->GetObjectId();
        ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
        ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
        ar(cereal::make_nvp("TypeName", type));
        for (auto& param : params) {
            if (param->checkFlags(mo::ParamFlags::Input_e))
            {
                InputParam* input = dynamic_cast<InputParam*>(param);
                if (input)
                {
                    auto input_source_param = input->getInputParam();
                    if (input_source_param)
                    {
                        std::string input_source = input_source_param->getTreeName();
                        std::string param_name = param->getName();
                        ar(cereal::make_nvp(param_name, input_source));
                    }
                }
            }
            if (param->checkFlags(mo::ParamFlags::Output_e))
                continue;
            auto func1 = SerializationFactory::instance()->getJsonSerializationFunction(param->getTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    MO_LOG(debug) << "Unable to serialize " << param->getTreeName();
                }
            }
            else
            {
                MO_LOG(debug) << "No serialization function found for " << param->getTypeInfo().name();
            }
        }
        return true;
    }
}

bool mo::DeSerialize(cereal::JSONInputArchive& ar, IMetaObject* obj)
{
    if (obj == nullptr)
        return false;
    if (auto func = SerializerFactory::GetDeSerializationFunctionJSON(obj->GetTypeName()))
    {
        func(obj, ar);
        return true;
    }
    else
    {
        MO_LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->getParams();
        for (auto& param : params) {
            if (param->checkFlags(mo::ParamFlags::Output_e))
                continue;
            auto func1 = SerializationFactory::instance()->getJsonDeSerializationFunction(param->getTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    MO_LOG(debug) << "Unable to serialize " << param->getTreeName();
                }
            }
            else
            {
                MO_LOG(debug) << "No serialization function found for " << param->getTypeInfo().name();
            }
        }
        return true;
    }
}
static std::list<ObjectId> serialized_objects;
void mo::StartSerialization()
{
    serialized_objects.clear();
}

void mo::SetHasBeenSerialized(ObjectId id)
{
    serialized_objects.push_back(id);
}

bool mo::CheckHasBeenSerialized(ObjectId id)
{
    return std::find(serialized_objects.begin(), serialized_objects.end(), id) != serialized_objects.end();
}

void mo::EndSerialization()
{
    serialized_objects.clear();
}
