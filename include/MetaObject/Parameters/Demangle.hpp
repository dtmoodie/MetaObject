#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"

namespace mo
{
    class MO_EXPORTS Demangle
    {
    public:
        static std::string TypeToName(TypeInfo type);
        static void RegisterName(TypeInfo type, const char* name);
    };
}