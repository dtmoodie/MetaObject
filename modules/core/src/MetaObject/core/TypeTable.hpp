#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"

#include <map>
#include <string>
#include <vector>

struct SystemTable;

namespace mo
{
    class MO_EXPORTS TypeTable
    {
      public:
        static TypeTable& instance();
        static TypeTable& instance(SystemTable* table);

        std::string typeToName(const TypeInfo& type);
        const TypeInfo nameToType(const std::string& name);

        template <class T>
        void registerType(const char* name);

        void registerType(const TypeInfo& info, const char* name);

        std::vector<TypeInfo> listKnownTypes();

      private:
        std::map<TypeInfo, std::string> m_types;

    }; // class TypeTable

    template <class T>
    void TypeTable::registerType(const char* name)
    {
        registerType(TypeInfo(typeid(T)), name);
    }
} // namespace mo
