#pragma once
#include "MetaObject/detail/Export.hpp"

#include "MetaObject/core/detail/Forward.hpp"
#include "MetaObject/detail/TypeInfo.hpp"

#include <memory>

namespace mo
{
    class Connection;
    class SignalManager;
    class ISlot;
    class ISignalRelay;
    class IMetaObject;
    class MetaObject;

    // Signals are simply relays for function calls
    // When a void return callback is Connected to a signal, any call of that callback will
    // be sent to any slot that is Connected to the signal.
    // The Connecting callback has contextual information that is used by the signal to determine
    // How to relay the function call to any Connected slots.
    // Slots are receivers of signals and callbacks.
    class MO_EXPORTS ISignal
    {
      public:
        virtual ~ISignal();
        virtual const TypeInfo& getSignature() const = 0;
        virtual std::shared_ptr<Connection> connect(ISlot* slot) = 0;
        virtual std::shared_ptr<Connection> connect(std::shared_ptr<ISignalRelay>& relay) = 0;
        virtual bool disconnect() = 0;
        virtual bool disconnect(ISlot* slot) = 0;
        virtual bool disconnect(std::weak_ptr<ISignalRelay> relay) = 0;

        IAsyncStream* getStream() const;
        void setStream(IAsyncStream* ctx);

      protected:
        IAsyncStream* m_stream = nullptr;
    };
}
