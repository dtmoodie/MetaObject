#ifndef MO_SIGNALS_ISIGNAL_HPP
#define MO_SIGNALS_ISIGNAL_HPP
#include "MetaObject/detail/Export.hpp"

#include "MetaObject/core/detail/forward.hpp"
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
        virtual TypeInfo getSignature() const = 0;
        virtual ConnectionPtr_t connect(ISlot& slot) = 0;
        virtual ConnectionPtr_t connect(std::shared_ptr<ISignalRelay>& relay) = 0;
        virtual bool disconnect() = 0;
        virtual bool disconnect(const ISlot& slot) = 0;
        virtual bool disconnect(const ISignalRelay& relay) = 0;
        virtual bool isConnected() const = 0;

        IAsyncStream* getStream() const;
        void setStream(IAsyncStream& stream);

      protected:
        IAsyncStream* m_stream = nullptr;
    };
} // namespace mo

#endif // MO_SIGNALS_ISIGNAL_HPP