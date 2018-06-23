#pragma once
#include "MetaObject/detail/Export.hpp"

#include <ostream>
#include <string>

struct SystemTable;

namespace mo
{
    class TypeInfo;
    class MO_EXPORTS Demangle
    {
      public:
        static std::string typeToName(const TypeInfo& type);
        static const TypeInfo& nameToType(const std::string& name);
        static void registerName(const TypeInfo& type, const char* name);
        static void registerType(const TypeInfo& type);
        // This returns a type map that is stored in cereal binary format that
        // can be used to reconstruct this database
        static void getTypeMapBinary(std::ostream& stream);
        static void saveTypeMap(const std::string& filename);
    };
}
