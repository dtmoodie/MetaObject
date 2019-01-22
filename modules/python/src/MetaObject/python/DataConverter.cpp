#include "DataConverter.hpp"
#include "MetaObject/core/detail/singleton.hpp"
#include <MetaObject/core/SystemTable.hpp>

namespace mo
{
    namespace python
    {
        DataConverterRegistry* DataConverterRegistry::instance()
        {
            return singleton<DataConverterRegistry>();
        }

        DataConverterRegistry* DataConverterRegistry::instance(SystemTable* table)
        {
            return singleton<DataConverterRegistry>(table);
        }

        void
        DataConverterRegistry::registerConverters(const mo::TypeInfo& type, const Set_t& setter, const Get_t& getter)
        {
            m_registered_converters[type] = std::make_pair(setter, getter);
        }

        DataConverterRegistry::Set_t DataConverterRegistry::getSetter(const mo::TypeInfo& type)
        {
            auto itr = m_registered_converters.find(type);
            if (itr != m_registered_converters.end())
            {
                return itr->second.first;
            }
            return {};
        }

        DataConverterRegistry::Get_t DataConverterRegistry::getGetter(const mo::TypeInfo& type)
        {
            auto itr = m_registered_converters.find(type);
            if (itr != m_registered_converters.end())
            {
                return itr->second.second;
            }
            return {};
        }

        std::vector<mo::TypeInfo> DataConverterRegistry::listConverters()
        {
            std::vector<mo::TypeInfo> types;
            for (const auto& itr : m_registered_converters)
            {
                types.push_back(itr.first);
            }
            return types;
        }

    } // namespace mo::python
} // namespace mo
