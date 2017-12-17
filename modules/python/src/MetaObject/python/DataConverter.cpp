#include "DataConverter.hpp"
#include "MetaObject/core/detail/singleton.hpp"

namespace mo
{
namespace python
{
    DataConverterRegistry* DataConverterRegistry::instance()
    {
        return uniqueSingleton<DataConverterRegistry>();
    }

    void DataConverterRegistry::registerConverters(const mo::TypeInfo& type, Set_t&& setter, const Get_t&& getter)
    {
        m_registered_converters[type] = std::make_pair(std::move(setter), std::move(getter));
    }

    DataConverterRegistry::Set_t DataConverterRegistry::getSetter(const mo::TypeInfo& type)
    {
        return m_registered_converters[type].first;
    }
    
    DataConverterRegistry::Get_t DataConverterRegistry::getGetter(const mo::TypeInfo& type)
    {
        return m_registered_converters[type].second;
    }


} // namespace mo::python
} // namespace mo