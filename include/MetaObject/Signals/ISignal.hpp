#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include <memory>

namespace mo
{
    class Context;
    struct MO_EXPORTS SignalInfo
    {
        TypeInfo signature;
        std::string name;
        std::string description;
    };
    
    struct MO_EXPORTS SlotInfo
    {
        TypeInfo signature;
        std::string name;
        std::string description;
    };

    class MO_EXPORTS ISignal
    {
	public:
        virtual ~ISignal(){}
        virtual TypeInfo GetSignature() const = 0;

        Context* _ctx = nullptr;
    };
}
