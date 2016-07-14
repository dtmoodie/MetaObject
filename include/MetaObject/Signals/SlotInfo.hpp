#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
namespace mo
{
    struct MO_EXPORTS SlotInfo
    {
        TypeInfo signature;
        std::string name;
        std::string description;
        std::string tooltip;
    };
}