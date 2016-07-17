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
    // Signals are simply relays for function calls
    // When a void return callback is connected to a signal, any call of that callback will 
    // be sent to any slot that is connected to the signal.
    // The connecting callback has contextual information that is used by the signal to determine
    // How to relay the function call to any connected slots.
    // Slots are receivers of signals and callbacks.  
    class MO_EXPORTS ISignal
    {
	public:
        virtual ~ISignal(){}
        virtual TypeInfo GetSignature() const = 0;
        
        //virtual std::shared_ptr<Connection> Connect(const std::string& name, SignalManager* mgr) = 0;
        virtual std::shared_ptr<Connection> Connect(ISlot* slot) = 0;

        //virtual void Disconnect(const std::string& name, SignalManager* mgr) = 0;
        virtual void Disconnect(ISlot* slot) = 0;
    };
}
