#pragma once
#include "MetaObject/Parameters/IParameter.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <string>

namespace mo
{
	struct ParameterInfo
	{
        ParameterInfo(const mo::TypeInfo& type_, const std::string& name_, const std::string& tooltip_ = "", const std::string description_ = "", ParameterType type_flags_ = Control_e):
            data_type(type_), name(name_), tooltip(tooltip_), description(description_), type_flags(type_flags_) { }

		mo::TypeInfo data_type;
		std::string name;
		std::string tooltip;
		std::string description;
		ParameterType type_flags;
	};
}