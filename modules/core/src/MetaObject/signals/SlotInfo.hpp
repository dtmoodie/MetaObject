#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
namespace mo
{
    struct MO_EXPORTS SlotInfo
    {
        std::string Print();
        TypeInfo signature;
        std::string name;
        std::string description;
        std::string tooltip;
    };
}
