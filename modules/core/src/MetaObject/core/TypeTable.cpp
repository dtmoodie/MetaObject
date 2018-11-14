#include "TypeTable.hpp"

#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/logging/logging.hpp"

#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

namespace mo
{
    TypeTable& TypeTable::instance()
    {
        auto module = PerModuleInterface::GetInstance();
        auto table = module->GetSystemTable();
        return instance(table);
    }

    TypeTable& TypeTable::instance(SystemTable* table)
    {
        if (table == nullptr)
        {
            table = PerModuleInterface::GetInstance()->GetSystemTable();
        }
        MO_ASSERT(table);
        auto inst = table->getSingleton<TypeTable>();
        if (!inst)
        {
            inst = table->setSingleton(std::unique_ptr<TypeTable>(new TypeTable));
        }
        return *inst;
    }

    std::string TypeTable::typeToName(const TypeInfo& type)
    {
        auto itr = m_types.find(type);
        if (itr == m_types.end())
        {
            return type.name();
        }
        else
        {
            return itr->second;
        }
    }

    const TypeInfo TypeTable::nameToType(const std::string& name)
    {
        for (const auto& pair : m_types)
        {
            if (pair.second == name)
            {
                return pair.first;
            }
        }
        THROW(warning, "{} not a registered type", name);
        return {};
    }

    void TypeTable::registerType(const TypeInfo& info, const char* name)
    {
        m_types[info] = name;
    }

    std::vector<TypeInfo> TypeTable::listKnownTypes()
    {
        std::vector<TypeInfo> types;
        for (const auto& itr : m_types)
        {
            types.push_back(itr.first);
        }
        return types;
    }
}
