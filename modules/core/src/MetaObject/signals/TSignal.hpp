#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/signals/ISignal.hpp"
#include <memory>
#include <mutex>
#include <vector>
namespace mo
{
    class IMetaObject;
    class Context;
    class Connection;
    template <class Sig>
    class TSignalRelay;
    template <class Sig>
    class TSignal
    {
    };
    template <class... T>
    class MO_EXPORTS TSignal<void(T...)> : public ISignal
    {
      public:
        TSignal();
        void operator()(T... args);
        void operator()(Context* ctx, T... args);
        virtual const TypeInfo& getSignature() const;

        std::shared_ptr<Connection> connect(ISlot* slot);
        std::shared_ptr<Connection> connect(std::shared_ptr<ISignalRelay>& relay);
        std::shared_ptr<Connection> connect(std::shared_ptr<TSignalRelay<void(T...)>>& relay);

        bool disconnect();
        bool disconnect(ISlot* slot);
        bool disconnect(std::weak_ptr<ISignalRelay> relay);

      protected:
        std::recursive_mutex mtx;
        std::vector<std::shared_ptr<TSignalRelay<void(T...)>>> _typed_relays;
    };

    template <class R, class... T>
    class MO_EXPORTS TSignal<R(T...)> : public ISignal
    {
      public:
        TSignal();
        R operator()(T... args);
        R operator()(Context* ctx, T... args);
        virtual const TypeInfo& getSignature() const;

        std::shared_ptr<Connection> connect(ISlot* slot);
        std::shared_ptr<Connection> connect(std::shared_ptr<ISignalRelay>& relay);
        std::shared_ptr<Connection> connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay);

        bool disconnect();
        bool disconnect(ISlot* slot);
        bool disconnect(std::weak_ptr<ISignalRelay> relay);

      protected:
        std::recursive_mutex mtx;
        std::shared_ptr<TSignalRelay<R(T...)>> _typed_relay;
    };
}
#include "detail/TSignalImpl.hpp"
