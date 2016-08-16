#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <functional>

namespace cereal
{
    class PortableBinaryInputArchive;
    class PortableBinaryOutputArchive;
    class XMLOutputArchive;
    class XMLInputArchive;
}

namespace mo
{
    class IParameter;
    class TypeInfo;
    class MO_EXPORTS SerializationFunctionRegistry
    {
    public:
        static SerializationFunctionRegistry* Instance();

        typedef std::function<bool(IParameter*, cereal::PortableBinaryOutputArchive&)> SerializeBinary_f;
        typedef std::function<bool(IParameter*, cereal::PortableBinaryInputArchive&)> DeSerializeBinary_f;
        typedef std::function<bool(IParameter*, cereal::XMLOutputArchive&)> SerializeXml_f;
        typedef std::function<bool(IParameter*, cereal::XMLInputArchive&)> DeSerializeXml_f;

        SerializeBinary_f GetBinarySerializationFunction(const TypeInfo& type);
        DeSerializeBinary_f GetBinaryDeSerializationFunction(const TypeInfo& type);

        SerializeXml_f GetXmlSerializationFunction(const TypeInfo& type);
        DeSerializeXml_f GetXmlDeSerializationFunction(const TypeInfo& type);

        void SetBinarySerializationFunction(const TypeInfo& type, SerializeBinary_f f);
        void SetBinaryDeSerializationFunction(const TypeInfo& type, DeSerializeBinary_f f);

        void SetXmlSerializationFunction(const TypeInfo& type, SerializeXml_f f);
        void SetXmlDeSerializationFunction(const TypeInfo& type, DeSerializeXml_f f);


    private:
        SerializationFunctionRegistry();
        ~SerializationFunctionRegistry();
        struct impl;
        impl* _pimpl;
    };
}