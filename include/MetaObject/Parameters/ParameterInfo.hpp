#pragma once
#include "MetaObject/Parameters/IParameter.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <string>

namespace mo
{
	struct ParameterInfo
	{
		mo::TypeInfo data_type;
		std::string name;
		std::string tooltip;
		std::string description;
		ParameterType type_flags;
	};
}