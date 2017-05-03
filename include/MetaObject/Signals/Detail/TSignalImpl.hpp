#pragma once
#include "MetaObject/Detail/Placeholders.hpp"
#include "MetaObject/Signals/Connection.hpp"
#include "MetaObject/Signals/TSlot.hpp"
#include "MetaObject/Signals/TSignalRelay.hpp"
#include "MetaObject/Logging/Log.hpp"

namespace mo
{
    template<class Sig> class TSignal;
    template<class R, class...T>
    TSignal<R(T...)>::TSignal()
    {

    }

    template<class R, class...T>
    R TSignal<R(T...)>::operator()(T... args)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (_typed_relay)
        {
            return (*_typed_relay)(this, args...);
        }
        THROW(debug) << "Not Connected to a signal relay";
        return R();
    }

    template<class R, class...T>
    R TSignal<R(T...)>::operator()(Context* ctx, T... args)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (_typed_relay)
        {
            return (*_typed_relay)(ctx, args...);
        }
        THROW(debug) << "Not Connected to a signal relay";
        return R();
    }

    template<class R, class...T>
    TypeInfo TSignal<R(T...)>::getSignature() const
    {
        return TypeInfo(typeid(R(T...)));
    }

    template<class R, class...T>
    std::shared_ptr<Connection> TSignal<R(T...)>::connect(ISlot* slot)
    {
        return slot->connect(this);
    }

    template<class R, class...T>
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

    template<class R, class...T>
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
            return std::shared_ptr<Connection>(new SignalConnection(this, relay));
        }
        return std::shared_ptr<Connection>();
    }


    template<class R, class...T>
    bool TSignal<R(T...)>::disConnect()
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (_typed_relay)
        {
            _typed_relay.reset();
            return true;
        }
        return false;
    }

    template<class R, class...T>
    bool TSignal<R(T...)>::disConnect(ISlot* slot)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (_typed_relay)
        {
            if (_typed_relay->_slot == slot)
            {
                _typed_relay.reset();
                return true;
            }
        }
        return false;
    }

    template<class R, class...T>
    bool TSignal<R(T...)>::disConnect(std::weak_ptr<ISignalRelay> relay_)
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
    template<class...T>
    TSignal<void(T...)>::TSignal()
    {

    }

    template<class...T>
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
    template<class...T>
    void TSignal<void(T...)>::operator()(Context* ctx, T... args)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        for (auto& relay : _typed_relays)
        {
            if (relay)
            {
                (*relay)(ctx, args...);
            }
        }
    }

    template<class...T>
    TypeInfo TSignal<void(T...)>::getSignature() const
    {
        return TypeInfo(typeid(void(T...)));
    }

    template<class...T>
    std::shared_ptr<Connection> TSignal<void(T...)>::connect(ISlot* slot)
    {
        return slot->connect(this);
    }

    template<class...T>
    std::shared_ptr<Connection> TSignal<void(T...)>::connect(std::shared_ptr<ISignalRelay>& relay){
        if (relay == nullptr){
            relay.reset(new TSignalRelay<void(T...)>());
        }
        auto typed = std::dynamic_pointer_cast<TSignalRelay<void(T...)>>(relay);
        if (typed)
            return connect(typed);
        return std::shared_ptr<Connection>();
    }

    template<class...T>
    std::shared_ptr<Connection> TSignal<void(T...)>::connect(std::shared_ptr<TSignalRelay<void(T...)>>& relay){
        if (relay == nullptr){
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


    template<class...T>
    bool TSignal<void(T...)>::disConnect()
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if (_typed_relays.size())
        {
            _typed_relays.clear();
            return true;
        }
        return false;
    }

    template<class...T>
    bool TSignal<void(T...)>::disConnect(ISlot* slot_)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx);
        for (auto relay = _typed_relays.begin(); relay != _typed_relays.end(); ++relay)
        {
            for (auto& slot : (*relay)->_slots)
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

    template<class...T>
    bool TSignal<void(T...)>::disConnect(std::weak_ptr<ISignalRelay> relay_)
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