#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include <memory>

namespace mo
{
    class Context;
    class MO_EXPORTS ISignal
    {
	public:
        virtual ~ISignal(){}
        virtual TypeInfo GetSignature() const = 0;

        Context* _ctx = nullptr;
    };
}
