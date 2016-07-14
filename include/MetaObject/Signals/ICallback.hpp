#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
namespace mo
{
    class IMetaObject;
    class Context;
    // ICallback wraps std::function.
    // ICallbacks cannot be called, only the explicit TypedCallback
    // Calls of TypedCallback call slots in other objects
    class MO_EXPORTS ICallback
    {
    public:
        virtual ~ICallback();
        virtual TypeInfo GetSignature() const = 0;
        virtual void Disconnect() = 0;
        IMetaObject* receiver = nullptr;
        Context* ctx = nullptr;
    };
}