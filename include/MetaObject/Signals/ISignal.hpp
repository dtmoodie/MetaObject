#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include <memory>

namespace mo
{
    class Context;
    struct SignalInfo
    {
        TypeInfo signature;
        std::string name;
        std::string description;
    };
    struct SlotInfo
    {
        TypeInfo signature;
        std::string name;
        std::string description;
    };

    class MO_EXPORTS ISignal
    {
	public:
        virtual ~ISignal(){}
        virtual TypeInfo GetSignature() = 0;

        Context* _ctx = nullptr;
    };
}
