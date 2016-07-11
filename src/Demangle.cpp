#include "parameters/Demangle.hpp"
#include <map>

using namespace Parameters;
std::map<Loki::TypeInfo, const char*>& Registry()
{
    static std::map<Loki::TypeInfo, const char*> inst;
    return inst;
}
std::string Demangle::TypeToName(Loki::TypeInfo type)
{
    std::map<Loki::TypeInfo, const char*>& reg = Registry();
    auto itr = reg.find(type);
    if(itr != reg.end())
    {
        return itr->second;
    }
    return type.name();
}

void Demangle::RegisterName(Loki::TypeInfo type, const char* name)
{
    std::map<Loki::TypeInfo, const char*>& reg = Registry();
    auto itr = reg.find(type);
    if(itr == reg.end())
    {
        reg[type] = name;
    }
}