

#include "TypeTable.hpp"

#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/logging/logging.hpp"

#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

namespace mo
{
    TypeTable::~TypeTable() = default;

    class TypeTableImpl : public TypeTable
    {
        std::string typeToName(const TypeInfo& type) override;
        const TypeInfo nameToType(const std::string& name) const override;

        void registerType(const TypeInfo& info, const char* name) override;

        std::vector<TypeInfo> listKnownTypes() const override;

      private:
        std::mutex m_mtx;
        std::unordered_map<TypeInfo, std::string> m_types;
    };

    std::shared_ptr<TypeTable> TypeTable::instance()
    {
        return mo::singleton<TypeTable>();
    }

    std::shared_ptr<TypeTable> TypeTable::instance(SystemTable* table)
    {
        MO_ASSERT(table);
        auto inst = table->getSingleton<TypeTable, TypeTableImpl>();
        return inst;
    }

    std::string TypeTableImpl::typeToName(const TypeInfo& type)
    {
        auto itr = m_types.find(type);
        if (itr == m_types.end())
        {
            m_types[type] = type.name();
            return type.name();
        }
        return itr->second;
    }

    const TypeInfo TypeTableImpl::nameToType(const std::string& name) const
    {
        for (const auto& pair : m_types)
        {
            if (pair.second == name)
            {
                return pair.first;
            }
        }
        THROW(warn, "{} not a registered type", name);
        return {};
    }

    void TypeTableImpl::registerType(const TypeInfo& info, const char* name)
    {
        m_types[info] = name;
    }

    std::vector<TypeInfo> TypeTableImpl::listKnownTypes() const
    {
        std::vector<TypeInfo> types;
        for (const auto& itr : m_types)
        {
            types.push_back(itr.first);
        }
        return types;
    }
} // namespace mo
