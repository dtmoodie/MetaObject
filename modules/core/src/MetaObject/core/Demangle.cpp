#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <map>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <fstream>
using namespace mo;
std::map<TypeInfo, std::string>& registry()
{
    static std::map<TypeInfo, std::string> inst;
    return inst;
}

std::string Demangle::typeToName(const TypeInfo& type)
{
    std::map<TypeInfo, std::string>& reg = registry();
    auto itr = reg.find(type);
    if (itr != reg.end())
    {
        return itr->second;
    }
    return type.name();
}

const TypeInfo& Demangle::nameToType(const std::string& name)
{
    static TypeInfo default_type = TypeInfo(typeid(void));
    std::map<TypeInfo, std::string>& reg = registry();
    for (const auto& itr : reg) {
        if (itr.second == name)
        {
            return itr.first;
        }
    }
    return default_type;
}

void Demangle::registerName(const TypeInfo& type, const char* name)
{
    std::map<TypeInfo, std::string>& reg = registry();
    auto itr = reg.find(type);
    if (itr == reg.end())
    {
        reg[type] = name;
    }
    else
    {
        if (itr->second.empty())
            itr->second = std::string(name);
    }
}

void Demangle::registerType(const TypeInfo& type)
{
    std::map<TypeInfo, std::string>& reg = registry();
    auto itr = reg.find(type);
    if (itr == reg.end())
    {
        reg[type] = "";
    }
}

void Demangle::getTypeMapBinary(std::ostream& stream)
{

    cereal::BinaryOutputArchive ar(stream);
    std::map<std::string, size_t> lut;
    auto& reg = registry();
    for (auto& itr : reg) {
        if (itr.second.size())
        {
            lut[itr.second] = itr.first.get().hash_code();
        }
        else
        {
            lut[itr.first.name()] = itr.first.get().hash_code();
        }
    }
    ar(lut);
}

void Demangle::saveTypeMap(const std::string& filename)
{
    std::ofstream ofs(filename);
    getTypeMapBinary(ofs);
}
