#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "TypeTable.hpp"
#include <map>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <fstream>

namespace mo
{
    std::string Demangle::typeToName(const TypeInfo& type) { return TypeTable::instance().typeToName(type); }

    const TypeInfo Demangle::nameToType(const std::string& name) { return TypeTable::instance().nameToType(name); }

    void Demangle::registerName(const TypeInfo& type, const char* name)
    {
        TypeTable::instance().registerType(type, name);
    }

    void Demangle::registerType(const TypeInfo& type) { TypeTable::instance().registerType(type, ""); }

    void Demangle::getTypeMapBinary(std::ostream& /*stream*/) {}

    void Demangle::saveTypeMap(const std::string& /*filename*/) {}
}
