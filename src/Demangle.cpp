#include "MetaObject/Parameters/Demangle.hpp"

#include <map>

using namespace mo;
std::map<TypeInfo, const char*>& Registry()
{
    static std::map<TypeInfo, const char*> inst;
    return inst;
}
std::string Demangle::TypeToName(TypeInfo type)
{
    std::map<TypeInfo, const char*>& reg = Registry();
    auto itr = reg.find(type);
    if(itr != reg.end())
    {
        return itr->second;
    }
    return type.name();
}

void Demangle::RegisterName(TypeInfo type, const char* name)
{
    std::map<TypeInfo, const char*>& reg = Registry();
    auto itr = reg.find(type);
    if(itr == reg.end())
    {
        reg[type] = name;
    }
}