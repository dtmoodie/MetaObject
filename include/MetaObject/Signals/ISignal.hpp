#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include <memory>

namespace mo
{
    class Context;
    class Connection;
    class SignalManager;
    class ISlot;
    class MO_EXPORTS ISignal
    {
	public:
        virtual ~ISignal(){}
        virtual TypeInfo GetSignature() const = 0;
        
        virtual std::shared_ptr<Connection> Connect(const std::string& name, SignalManager* mgr) = 0;
        virtual std::shared_ptr<Connection> Connect(ISlot* slot) = 0;

        virtual void Disconnect(const std::string& name, SignalManager* mgr) = 0;
        virtual void Disconnect(ISlot* slot) = 0;

        Context* _ctx = nullptr;
    };
}
