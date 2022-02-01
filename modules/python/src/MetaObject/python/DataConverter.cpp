#include "DataConverter.hpp"
#include <MetaObject/core/SystemTable.hpp>

namespace mo
{
    namespace python
    {
        DataConversionTable* DataConversionTable::instance()
        {
            return singleton<DataConversionTable>().get();
        }

        DataConversionTable* DataConversionTable::instance(SystemTable* table)
        {
            return table->getSingleton<DataConversionTable>().get();
        }

        void DataConversionTable::registerConverters(const mo::TypeInfo& type,
                                                     const FromPython_t& from,
                                                     const ToPython_t& to)
        {
            m_registered_converters[type] = std::make_pair(to, from);
        }

        DataConversionTable::FromPython_t DataConversionTable::getConverterFromPython(const mo::TypeInfo& type) const
        {
            auto itr = m_registered_converters.find(type);
            if (itr != m_registered_converters.end())
            {
                return itr->second.second;
            }
            return {};
        }

        DataConversionTable::ToPython_t DataConversionTable::getConverterToPython(const mo::TypeInfo& type) const
        {
            auto itr = m_registered_converters.find(type);
            if (itr != m_registered_converters.end())
            {
                return itr->second.first;
            }
            return {};
        }

        std::vector<mo::TypeInfo> DataConversionTable::listConverters() const
        {
            std::vector<mo::TypeInfo> types;
            for (const auto& itr : m_registered_converters)
            {
                types.push_back(itr.first);
            }
            return types;
        }

    } // namespace python
} // namespace mo
