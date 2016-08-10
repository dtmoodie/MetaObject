#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <ostream>
#include <istream>
namespace mo
{
    class IMetaObject;
    class MO_EXPORTS ISerializer
    {
    public:
        enum SerializationType
        {
            Binary_e = 0,
            xml_e,
            json_e
        };

        // populate a message stream from an object
        virtual void         Serialize(IMetaObject* obj, std::ostream& os, SerializationType type) = 0;

        // Create an object from a message stream
        virtual IMetaObject* DeSerialize(std::istream& os, SerializationType type) = 0;

        // Populate a created object from a message stream
        virtual void         DeSerialize(IMetaObject* obj, std::istream& os, SerializationType type) = 0;
    };
}