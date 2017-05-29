#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <memory>
namespace mo
{
    class ISignal;
    class Context;
    class Connection;
    class IMetaObject;
    class ISignalRelay;
    class MO_EXPORTS ISlot
    {
    public:
        virtual ~ISlot();
        virtual std::shared_ptr<Connection> connect(ISignal* sig) = 0;
        virtual std::shared_ptr<Connection> connect(std::shared_ptr<ISignalRelay>& relay) = 0;
        virtual bool disConnect(std::weak_ptr<ISignalRelay> relay) = 0;
        virtual void clear() = 0;
        virtual TypeInfo getSignature() const = 0;
        IMetaObject* getParent() const;
        Context* getContext() const;
        void setContext(Context* ctx);

    protected:
        friend class IMetaObject;
        void setParent(IMetaObject* parent);
        IMetaObject* _parent = nullptr;
        Context* _ctx;
    };
}
