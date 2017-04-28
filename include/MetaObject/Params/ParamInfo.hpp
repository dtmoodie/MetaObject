#pragma once
#include "MetaObject/Params/IParam.hpp"
#include "MetaObject/Detail/TypeInfo.hpp"
#include <string>

namespace mo {
struct ParamInfo {
    ParamInfo(const mo::TypeInfo& type_,
                  const std::string& name_,
                  const std::string& tooltip_ = "",
                  const std::string description_ = "",
                  ParamFlags type_flags_ = Control_e,
                  const std::string& init = ""):
        data_type(type_), name(name_),
        tooltip(tooltip_), description(description_),
        type_flags(type_flags_), initial_value(init)
    { }
    std::string Print();
    mo::TypeInfo data_type;
    std::string name;
    std::string tooltip;
    std::string description;
    std::string initial_value;
    ParamFlags type_flags;
};
}
