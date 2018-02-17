#pragma once
#include "MetaObject/serialization.hpp"
#include <functional>

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

    class MO_EXPORTS IBinarySerializer
    {
      public:
        virtual ~IBinarySerializer() {}
        virtual void setInput(IParam* input) = 0;
    };

    class MO_EXPORTS IXmlSerializer
    {
    };

    class MO_EXPORTS IJsonSerializer
    {
    };

    class MO_EXPORTS ITextSerializer
    {
    };

    class MO_EXPORTS SerializationFactory
    {
      public:
        static SerializationFactory* instance();

        typedef std::function<bool(IParam*, cereal::BinaryOutputArchive&)> SerializeBinary_f;
        typedef std::function<bool(IParam*, cereal::BinaryInputArchive&)> DeSerializeBinary_f;
        typedef std::function<bool(IParam*, cereal::XMLOutputArchive&)> SerializeXml_f;
        typedef std::function<bool(IParam*, cereal::XMLInputArchive&)> DeSerializeXml_f;
        typedef std::function<bool(IParam*, cereal::JSONOutputArchive&)> SerializeJson_f;
        typedef std::function<bool(IParam*, cereal::JSONInputArchive&)> DeSerializeJson_f;
        typedef std::function<bool(IParam*, std::stringstream&)> SerializeText_f;
        typedef std::function<bool(IParam*, std::stringstream&)> DeSerializeText_f;

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

      private:
        SerializationFactory();
        ~SerializationFactory();
        struct impl;
        impl* _pimpl;
    };
}
