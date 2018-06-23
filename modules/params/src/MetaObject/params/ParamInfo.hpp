#pragma once
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/params/IParam.hpp"
#include <string>

namespace mo
{
    struct ParamInfo
    {
        ParamInfo(const mo::TypeInfo& type_,
                  const std::string& name_,
                  const std::string& tooltip_ = "",
                  const std::string description_ = "",
                  ParamFlags type_flags_ = ParamFlags::Control_e,
                  const std::string& init = "");
        std::string print() const;
        const std::string& getName() const;
        const std::string& getTooltip() const;
        const std::string& getDescription() const;
        const std::string& getInitialization() const;
        const EnumClassBitset<ParamFlags> getParamType() const;
        mo::TypeInfo getDataType() const;

      private:
        mo::TypeInfo data_type;
        std::string name;
        std::string tooltip;
        std::string description;
        std::string initial_value;
        EnumClassBitset<ParamFlags> type_flags;
    };
}
