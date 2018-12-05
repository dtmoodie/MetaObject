#pragma once
#include <MetaObject/core/detail/forward.hpp>

#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/logging/logging.hpp"

#include "Connection.hpp"
#include "ISignal.hpp"
#include "ISignalRelay.hpp"
#include "TSlot.hpp"

#include <memory>
#include <mutex>
#include <vector>

namespace mo
{
    template <class Sig, class Mutex>
    class TSignalRelay;

    template <class Sig>
    class TSignal;

    template <class... T>
    class MO_EXPORTS TSignal<void(T...)> : public ISignal
    {
      public:
        TSignal();
        void operator()(T... args);
        void operator()(IAsyncStream* ctx, T... args);
        virtual const TypeInfo& getSignature() const;

        std::shared_ptr<Connection> connect(ISlot* slot);
        std::shared_ptr<Connection> connect(std::shared_ptr<ISignalRelay>& relay);
        std::shared_ptr<Connection> connect(std::shared_ptr<TSignalRelay<void(T...)>>& relay);

        bool disconnect();
        bool disconnect(ISlot* slot);
        bool disconnect(std::weak_ptr<ISignalRelay> relay);

      protected:
        std::recursive_mutex mtx;
        std::vector<typename TSignalRelay<void(T...)>::Ptr> _typed_relays;
    };

    template <class R, class... T>
    class MO_EXPORTS TSignal<R(T...)> : public ISignal
    {
      public:
        TSignal();
        R operator()(T... args);
        R operator()(IAsyncStream* ctx, T... args);
        virtual const TypeInfo& getSignature() const;

        ConnectionPtr_t connect(ISlot* slot);
        ConnectionPtr_t connect(ISignalRelay::Ptr& relay);
        ConnectionPtr_t connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay);

        bool disconnect();
        bool disconnect(ISlot* slot);
        bool disconnect(std::weak_ptr<ISignalRelay> relay);

      protected:
        std::recursive_mutex mtx;
        typename TSignalRelay<R(T...)>::Ptr _typed_relay;
    };

    /////////////////////////////////////////////////////////////////////////
    ///  Implementation
    /////////////////////////////////////////////////////////////////////////
    template <class Sig>
    class TSignal;
    template <class R, class... T>
    TSignal<R(T...)>::TSignal()
    {
    }

    template <class R, class... T>
    R TSignal<R(T...)>::operator()(T... args)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (_typed_relay)
        {
            return (*_typed_relay)(this, args...);
        }
        THROW(debug, "Not Connected to a signal relay");
        return R();
    }

    template <class R, class... T>
    R TSignal<R(T...)>::operator()(IAsyncStream* stream, T... args)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (_typed_relay)
        {
            return (*_typed_relay)(stream, args...);
        }
        THROW(debug, "Not Connected to a signal relay");
        return R();
    }

    template <class R, class... T>
    const TypeInfo& TSignal<R(T...)>::getSignature() const
    {
        static TypeInfo type(typeid(R(T...)));
        return type;
    }

    template <class R, class... T>
    std::shared_ptr<Connection> TSignal<R(T...)>::connect(ISlot* slot)
    {
        return slot->connect(this);
    }

    template <class R, class... T>
    std::shared_ptr<Connection> TSignal<R(T...)>::connect(std::shared_ptr<ISignalRelay>& relay)
    {
        if (relay == nullptr)
        {
            relay.reset(new TSignalRelay<R(T...)>());
        }
        auto typed = std::dynamic_pointer_cast<TSignalRelay<R(T...)>>(relay);
        if (typed)
            return connect(typed);
        return std::shared_ptr<Connection>();
    }

    template <class R, class... T>
    std::shared_ptr<Connection> TSignal<R(T...)>::connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay)
    {
        if (relay == nullptr)
        {
            relay.reset(new TSignalRelay<R(T...)>());
        }
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (relay != _typed_relay)
        {
            _typed_relay = relay;
            return std::make_shared<SignalConnection>(this, relay);
        }
        return std::shared_ptr<Connection>();
    }

    template <class R, class... T>
    bool TSignal<R(T...)>::disconnect()
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (_typed_relay)
        {
            _typed_relay.reset();
            return true;
        }
        return false;
    }

    template <class R, class... T>
    bool TSignal<R(T...)>::disconnect(ISlot* slot)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (_typed_relay)
        {
            if (_typed_relay->m_slot == slot)
            {
                _typed_relay.reset();
                return true;
            }
        }
        return false;
    }

    template <class R, class... T>
    bool TSignal<R(T...)>::disconnect(std::weak_ptr<ISignalRelay> relay_)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        auto relay = relay_.lock();
        if (_typed_relay == relay)
        {
            _typed_relay.reset();
            return true;
        }
        return false;
    }

    // ---------------------------------------------------------------------
    // void specialization
    template <class... T>
    TSignal<void(T...)>::TSignal()
    {
    }

    template <class... T>
    void TSignal<void(T...)>::operator()(T... args)
    {
        std::unique_lock<std::recursive_mutex> lock(mtx);
        auto relays = _typed_relays;
        lock.unlock();
        for (auto& relay : relays)
        {
            if (relay)
            {
                (*relay)(this, args...);
            }
        }
    }
    template <class... T>
    void TSignal<void(T...)>::operator()(IAsyncStream* stream, T... args)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        for (auto& relay : _typed_relays)
        {
            if (relay)
            {
                (*relay)(stream, args...);
            }
        }
    }

    template <class... T>
    const TypeInfo& TSignal<void(T...)>::getSignature() const
    {
        static TypeInfo type(typeid(void(T...)));
        return type;
    }

    template <class... T>
    std::shared_ptr<Connection> TSignal<void(T...)>::connect(ISlot* slot)
    {
        return slot->connect(this);
    }

    template <class... T>
    std::shared_ptr<Connection> TSignal<void(T...)>::connect(std::shared_ptr<ISignalRelay>& relay)
    {
        if (relay == nullptr)
        {
            relay.reset(new TSignalRelay<void(T...)>());
        }
        auto typed = std::dynamic_pointer_cast<TSignalRelay<void(T...)>>(relay);
        if (typed)
            return connect(typed);
        return std::shared_ptr<Connection>();
    }

    template <class... T>
    std::shared_ptr<Connection> TSignal<void(T...)>::connect(std::shared_ptr<TSignalRelay<void(T...)>>& relay)
    {
        if (relay == nullptr)
        {
            relay.reset(new TSignalRelay<void(T...)>());
        }
        std::lock_guard<std::recursive_mutex> lock(mtx);
        auto itr = std::find(_typed_relays.begin(), _typed_relays.end(), relay);
        if (itr == _typed_relays.end())
        {
            _typed_relays.push_back(relay);
            return std::shared_ptr<Connection>(new SignalConnection(this, relay));
        }
        return std::shared_ptr<Connection>();
    }

    template <class... T>
    bool TSignal<void(T...)>::disconnect()
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (_typed_relays.size())
        {
            _typed_relays.clear();
            return true;
        }
        return false;
    }

    template <class... T>
    bool TSignal<void(T...)>::disconnect(ISlot* slot_)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        for (auto relay = _typed_relays.begin(); relay != _typed_relays.end(); ++relay)
        {
            for (auto& slot : (*relay)->m_slots)
            {
                if (slot == slot_)
                {
                    _typed_relays.erase(relay);
                    return true;
                }
            }
        }
        return false;
    }

    template <class... T>
    bool TSignal<void(T...)>::disconnect(std::weak_ptr<ISignalRelay> relay_)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        auto relay = relay_.lock();
        auto itr = std::find(_typed_relays.begin(), _typed_relays.end(), relay);
        if (itr != _typed_relays.end())
        {
            _typed_relays.erase(itr);
            return true;
        }
        return false;
    }
}
