#include "DataConverter.hpp"
#include <MetaObject/core/SystemTable.hpp>

#include <cstdint>
#include <string>

namespace mo
{
    namespace python
    {
        // Register converters for the standard POD types. Object param accessors
        // (addParamAccessors) are only added for types present in the
        // DataConverterRegistry, and the metaparams instantiations no longer
        // populate it, so the python module registers its own basic coverage.
        void registerPodConverters(SystemTable* table)
        {
            // Named locals: ParamConverter<T>(table) would parse as a function
            // declaration (most-vexing-parse), not a constructor call.
            ParamConverter<bool> bool_converter(table);
            ParamConverter<uint8_t> uint8_converter(table);
            ParamConverter<int8_t> int8_converter(table);
            ParamConverter<uint16_t> uint16_converter(table);
            ParamConverter<int16_t> int16_converter(table);
            ParamConverter<uint32_t> uint32_converter(table);
            ParamConverter<int32_t> int32_converter(table);
            ParamConverter<uint64_t> uint64_converter(table);
            ParamConverter<int64_t> int64_converter(table);
            ParamConverter<float> float_converter(table);
            ParamConverter<double> double_converter(table);
            ParamConverter<std::string> string_converter(table);
        }

        DataConverterRegistry* DataConverterRegistry::instance()
        {
            return singleton<DataConverterRegistry>().get();
        }

        DataConverterRegistry* DataConverterRegistry::instance(SystemTable* table)
        {
            return table->getSingleton<DataConverterRegistry>().get();
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
