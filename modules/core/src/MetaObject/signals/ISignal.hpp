#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include <memory>

namespace mo
{
    class Context;
    class Connection;
    class SignalManager;
    class ISlot;
    class ISignalRelay;
    class IMetaObject;
    // Signals are simply relays for function calls
    // When a void return callback is Connected to a signal, any call of that callback will
    // be sent to any slot that is Connected to the signal.
    // The Connecting callback has contextual information that is used by the signal to determine
    // How to relay the function call to any Connected slots.
    // Slots are receivers of signals and callbacks.
    class MO_EXPORTS ISignal
    {
    public:
        virtual TypeInfo getSignature() const = 0;
        virtual std::shared_ptr<Connection> connect(ISlot* slot) = 0;
        virtual std::shared_ptr<Connection> connect(std::shared_ptr<ISignalRelay>& relay) = 0;
        virtual bool disConnect() = 0;
        virtual bool disConnect(ISlot* slot) = 0;
        virtual bool disConnect(std::weak_ptr<ISignalRelay> relay) = 0;

        IMetaObject* getParent() const;
        const Context* getContext() const;
        void setContext(Context* ctx);
    protected:
        friend class IMetaObject;
        void setParent(IMetaObject* parent);
        IMetaObject* _parent = nullptr;
        Context* _ctx = nullptr;
    };
}