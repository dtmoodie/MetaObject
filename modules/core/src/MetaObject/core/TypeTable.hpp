#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <memory>
#include <string>
#include <vector>
struct SystemTable;

namespace mo
{
    class MO_EXPORTS TypeTable
    {
      public:
        static std::shared_ptr<TypeTable> instance();
        static std::shared_ptr<TypeTable> instance(SystemTable* table);

        virtual ~TypeTable();
        virtual std::string typeToName(const TypeInfo& type) = 0;
        virtual const TypeInfo nameToType(const std::string& name) const = 0;
        virtual void registerType(const TypeInfo& info, const char* name) = 0;
        virtual std::vector<TypeInfo> listKnownTypes() const = 0;

        template <class T>
        void registerType(const char* name);

    }; // class TypeTable

    template <class T>
    void TypeTable::registerType(const char* name)
    {
        registerType(TypeInfo(typeid(T)), name);
    }
} // namespace mo
