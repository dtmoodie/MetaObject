#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <memory>
namespace mo
{
    class ISignal;
    class ICallback;
    class Context;
    class Connection;
    class IMetaObject;
    class MO_EXPORTS ISlot
    {
    public:
        virtual ~ISlot();
        virtual std::shared_ptr<Connection> Connect(std::weak_ptr<ISignal>& signal) = 0;
        virtual bool Connect(std::weak_ptr<ICallback>& cb) const = 0;
        virtual bool Connect(ICallback* cb) const = 0;
        virtual TypeInfo GetSignature() const = 0;
        Context* _ctx = nullptr;
    };
}