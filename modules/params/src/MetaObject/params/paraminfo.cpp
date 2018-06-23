#include "ParamInfo.hpp"

namespace mo
{
    ParamInfo::ParamInfo(const mo::TypeInfo& type_,
                         const std::string& name_,
                         const std::string& tooltip_,
                         const std::string description_,
                         ParamFlags type_flags_,
                         const std::string& init)
        : data_type(type_), name(name_), tooltip(tooltip_), description(description_), initial_value(init),
          type_flags(type_flags_)

    {
    }

    std::string ParamInfo::print() const { return ""; }

    const std::string& ParamInfo::getName() const { return name; }

    const std::string& ParamInfo::getTooltip() const { return tooltip; }

    const std::string& ParamInfo::getDescription() const { return description; }

    const std::string& ParamInfo::getInitialization() const { return initial_value; }

    const EnumClassBitset<ParamFlags> ParamInfo::getParamType() const { return type_flags; }

    mo::TypeInfo ParamInfo::getDataType() const { return data_type; }
}
