#pragma once

#include "ISerializer.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <cereal/cereal.hpp>
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
    class MO_EXPORTS SerializerFactory: public ISerializer
    {
    public:
        static void Serialize(IMetaObject* obj, std::ostream& os, SerializationType type);
        static void DeSerialize(IMetaObject* obj, std::istream& os, SerializationType type);
        static IMetaObject* DeSerialize(std::istream& os, SerializationType type);

        typedef std::function<void(IMetaObject*, cereal::PortableBinaryOutputArchive&)> BinarySerialize_f;
        typedef std::function<void(IMetaObject*, cereal::PortableBinaryInputArchive&)> BinaryDeSerialize_f;

        typedef std::function<void(IMetaObject*, cereal::XMLOutputArchive&)> XMLSerialize_f;
        typedef std::function<void(IMetaObject*, cereal::XMLInputArchive&)> XMLDeSerialize_f;

        static void RegisterSerializationFunctionBinary(const char* obj_type, BinarySerialize_f f);
        static void RegisterDeSerializationFunctionBinary(const char* obj_type, BinaryDeSerialize_f f);
        static void RegisterSerializationFunctionXML(const char* obj_type, XMLSerialize_f f);
        static void RegisterDeSerializationFunctionXML(const char* obj_type, XMLDeSerialize_f f);
    };
}