#include "MetaObject/serialization/SerializationFactory.hpp"
#include "MetaObject/detail/TypeInfo.hpp"

#include <map>

using namespace mo;
struct SerializationFactory::impl
{
    std::map<TypeInfo, std::pair<SerializeBinary_f, DeSerializeBinary_f>> _binary_map;
    std::map<TypeInfo, std::pair<SerializeXml_f, DeSerializeXml_f>> _xml_map;
    std::map<TypeInfo, std::pair<SerializeJson_f, DeSerializeJson_f>> _json_map;
    std::map<TypeInfo, std::pair<SerializeText_f, DeSerializeText_f>> _text_map;
};

SerializationFactory::SerializationFactory()
{
    _pimpl = new impl();
}

SerializationFactory::~SerializationFactory()
{
    delete _pimpl;
}

SerializationFactory* SerializationFactory::instance()
{
    static SerializationFactory* g_inst = nullptr;
    if (g_inst == nullptr)
        g_inst = new SerializationFactory();
    return g_inst;
}

SerializationFactory::SerializeBinary_f SerializationFactory::getBinarySerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_binary_map.find(type);
    if (itr != _pimpl->_binary_map.end())
    {
        return itr->second.first;
    }
    return SerializeBinary_f();
}

SerializationFactory::DeSerializeBinary_f SerializationFactory::getBinaryDeSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_binary_map.find(type);
    if (itr != _pimpl->_binary_map.end())
    {
        return itr->second.second;
    }
    return DeSerializeBinary_f();
}

SerializationFactory::SerializeXml_f SerializationFactory::getXmlSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_xml_map.find(type);
    if (itr != _pimpl->_xml_map.end())
    {
        return itr->second.first;
    }
    return SerializeXml_f();
}

SerializationFactory::DeSerializeXml_f SerializationFactory::getXmlDeSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_xml_map.find(type);
    if (itr != _pimpl->_xml_map.end())
    {
        return itr->second.second;
    }
    return DeSerializeXml_f();
}

SerializationFactory::SerializeJson_f SerializationFactory::getJsonSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_json_map.find(type);
    if (itr != _pimpl->_json_map.end())
    {
        return itr->second.first;
    }
    return SerializeJson_f();
}
SerializationFactory::DeSerializeJson_f SerializationFactory::getJsonDeSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_json_map.find(type);
    if (itr != _pimpl->_json_map.end())
    {
        return itr->second.second;
    }
    return DeSerializeJson_f();
}

SerializationFactory::SerializeText_f SerializationFactory::getTextSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_text_map.find(type);
    if (itr != _pimpl->_text_map.end())
    {
        return itr->second.first;
    }
    return SerializeText_f();
}

SerializationFactory::DeSerializeText_f SerializationFactory::getTextDeSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_text_map.find(type);
    if (itr != _pimpl->_text_map.end())
    {
        return itr->second.second;
    }
    return DeSerializeText_f();
}

void SerializationFactory::setBinarySerializationFunctions(const TypeInfo& type,
                                                           SerializeBinary_f s,
                                                           DeSerializeBinary_f l)
{
    if (_pimpl->_binary_map.find(type) == _pimpl->_binary_map.end())
        _pimpl->_binary_map[type] = std::make_pair(s, l);
}

void SerializationFactory::setXmlSerializationFunctions(const TypeInfo& type, SerializeXml_f s, DeSerializeXml_f l)
{
    if (_pimpl->_xml_map.find(type) == _pimpl->_xml_map.end())
        _pimpl->_xml_map[type] = std::make_pair(s, l);
}

void SerializationFactory::setJsonSerializationFunctions(const TypeInfo& type,
                                                         SerializeJson_f serialize,
                                                         DeSerializeJson_f deserialize)
{
    if (_pimpl->_json_map.find(type) == _pimpl->_json_map.end())
        _pimpl->_json_map[type] = std::make_pair(serialize, deserialize);
}

void SerializationFactory::setTextSerializationFunctions(const TypeInfo& type,
                                                         SerializeText_f serialize,
                                                         DeSerializeText_f deserialize)
{
    if (_pimpl->_text_map.find(type) == _pimpl->_text_map.end())
        _pimpl->_text_map[type] = std::make_pair(serialize, deserialize);
}

SerializationFactory::SerializeBinary_f SerializationFactory::getSaveFunction(const TypeInfo& type,
                                                                              cereal::BinaryOutputArchive& ar)
{
    (void)ar;
    return getBinarySerializationFunction(type);
}

SerializationFactory::DeSerializeBinary_f SerializationFactory::getLoadFunction(const TypeInfo& type,
                                                                                cereal::BinaryInputArchive& ar)
{
    (void)ar;
    return getBinaryDeSerializationFunction(type);
}

SerializationFactory::SerializeXml_f SerializationFactory::getSaveFunction(const TypeInfo& type,
                                                                           cereal::XMLOutputArchive& ar)
{
    (void)ar;
    return getXmlSerializationFunction(type);
}

SerializationFactory::DeSerializeXml_f SerializationFactory::getLoadFunction(const TypeInfo& type,
                                                                             cereal::XMLInputArchive& ar)
{
    (void)ar;
    return getXmlDeSerializationFunction(type);
}

SerializationFactory::SerializeJson_f SerializationFactory::getSaveFunction(const TypeInfo& type,
                                                                            cereal::JSONOutputArchive& ar)
{
    (void)ar;
    return getJsonSerializationFunction(type);
}

SerializationFactory::DeSerializeJson_f SerializationFactory::getLoadFunction(const TypeInfo& type,
                                                                              cereal::JSONInputArchive& ar)
{
    (void)ar;
    return getJsonDeSerializationFunction(type);
}
