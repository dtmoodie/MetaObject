#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
namespace mo
{
    class IMetaObject;
    class MO_EXPORTS ICallback
    {
    public:
        virtual ~ICallback();
        virtual TypeInfo GetSignature() const = 0;
        virtual void Disconnect() = 0;
        IMetaObject* receiver = nullptr;
    };
}