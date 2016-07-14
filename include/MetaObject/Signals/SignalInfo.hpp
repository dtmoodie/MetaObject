#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
namespace mo
{
    struct MO_EXPORTS SignalInfo
    {
        TypeInfo signature;
        std::string name;
        std::string description;
        std::string tooltip;
    };
}