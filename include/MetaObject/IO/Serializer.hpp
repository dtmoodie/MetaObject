#pragma once

#include "ISerializer.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <cereal/cereal.hpp>

namespace cereal
{
    class PortableBinaryInputArchive ;
    class PortableBinaryOutputArchive ;
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
        
        static void RegisterSerializationFunction(const char* obj_type, BinarySerialize_f f);
        static void RegisterDeSerializationFunction(const char* obj_type, BinaryDeSerialize_f f);
    };
}