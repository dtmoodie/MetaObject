#pragma once
#include "MetaObject/serialization.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace cereal
{
    class BinaryInputArchive;
    class BinaryOutputArchive;
    class XMLOutputArchive;
    class XMLInputArchive;
    class JSONOutputArchive;
    class JSONInputArchive;
}

namespace mo
{
    class IParam;
    class TypeInfo;
    class Context;

    struct Binary;
    struct JSON;
    struct XML;

    template <class T>
    struct ArchivePairs
    {
    };
    template <>
    struct ArchivePairs<Binary>
    {
        using Input = cereal::BinaryInputArchive;
        using Output = cereal::BinaryOutputArchive;
    };
    template <>
    struct ArchivePairs<JSON>
    {
        using Input = cereal::JSONInputArchive;
        using Output = cereal::JSONOutputArchive;
    };
    template <>
    struct ArchivePairs<XML>
    {
        using Input = cereal::XMLInputArchive;
        using Output = cereal::XMLOutputArchive;
    };

    class MO_EXPORTS SerializationFactory
    {
      public:
        static SerializationFactory* instance();

        template <class AR>
        using Serializer_f = std::function<bool(const IParam*, AR&)>;
        template <class AR>
        using Deserializer_f = std::function<bool(IParam*, AR&)>;

        using SerializeBinary_f = Serializer_f<cereal::BinaryOutputArchive>;
        using SerializeXml_f = Serializer_f<cereal::XMLOutputArchive>;
        using SerializeJson_f = Serializer_f<cereal::JSONOutputArchive>;
        using SerializeText_f = Serializer_f<std::stringstream>;

        using DeSerializeBinary_f = Deserializer_f<cereal::BinaryInputArchive>;
        using DeSerializeXml_f = Deserializer_f<cereal::XMLInputArchive>;
        using DeSerializeJson_f = Deserializer_f<cereal::JSONInputArchive>;
        using DeSerializeText_f = Deserializer_f<std::stringstream>;

        template <class AR>
        Serializer_f<AR> getSerializer(const TypeInfo& type);
        template <class AR>
        Deserializer_f<AR> getDeserializer(const TypeInfo& type);

        SerializeBinary_f getBinarySerializationFunction(const TypeInfo& type);
        DeSerializeBinary_f getBinaryDeSerializationFunction(const TypeInfo& type);

        SerializeXml_f getXmlSerializationFunction(const TypeInfo& type);
        DeSerializeXml_f getXmlDeSerializationFunction(const TypeInfo& type);

        SerializeJson_f getJsonSerializationFunction(const TypeInfo& type);
        DeSerializeJson_f getJsonDeSerializationFunction(const TypeInfo& type);

        SerializeText_f getTextSerializationFunction(const TypeInfo& type);
        DeSerializeText_f getTextDeSerializationFunction(const TypeInfo& type);

        SerializeBinary_f getSaveFunction(const TypeInfo& type, cereal::BinaryOutputArchive& ar);
        DeSerializeBinary_f getLoadFunction(const TypeInfo& type, cereal::BinaryInputArchive& ar);

        SerializeXml_f getSaveFunction(const TypeInfo& type, cereal::XMLOutputArchive& ar);
        DeSerializeXml_f getLoadFunction(const TypeInfo& type, cereal::XMLInputArchive& ar);

        SerializeJson_f getSaveFunction(const TypeInfo& type, cereal::JSONOutputArchive& ar);
        DeSerializeJson_f getLoadFunction(const TypeInfo& type, cereal::JSONInputArchive& ar);

        void setBinarySerializationFunctions(const TypeInfo& type,
                                             SerializeBinary_f serialize,
                                             DeSerializeBinary_f deserialize);

        void setXmlSerializationFunctions(const TypeInfo& type, SerializeXml_f serialize, DeSerializeXml_f deserialize);

        void
        setJsonSerializationFunctions(const TypeInfo& type, SerializeJson_f serialize, DeSerializeJson_f deserialize);

        void
        setTextSerializationFunctions(const TypeInfo& type, SerializeText_f serialize, DeSerializeText_f deserialize);

        std::vector<TypeInfo> listSerializableTypes();

      private:
        SerializationFactory();
        ~SerializationFactory();
        struct impl;
        std::unique_ptr<impl> _pimpl;
    };
}
