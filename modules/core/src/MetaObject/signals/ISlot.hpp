#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <memory>
namespace mo
{
    class ISignal;
    class IAsyncStream;
    class Connection;
    class IMetaObject;
    class MetaObject;
    class ISignalRelay;
    class MO_EXPORTS ISlot
    {
      public:
        virtual ~ISlot();
        virtual std::shared_ptr<Connection> connect(ISignal* sig) = 0;
        virtual std::shared_ptr<Connection> connect(std::shared_ptr<ISignalRelay>& relay) = 0;
        virtual bool disconnect(std::weak_ptr<ISignalRelay> relay) = 0;
        virtual void clear() = 0;
        virtual const TypeInfo& getSignature() const = 0;
        IMetaObject* getParent() const;
        IAsyncStream* getStream() const;
        void setStream(IAsyncStream* ctx);

      protected:
        IAsyncStream* m_stream = nullptr;
    };
}
