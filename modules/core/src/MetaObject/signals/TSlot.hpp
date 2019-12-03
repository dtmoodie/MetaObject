#pragma once
#include "Connection.hpp"
#include "ISlot.hpp"
#include "MetaObject/core/AsyncStream.hpp"
#include "TSignal.hpp"
#include "TSignalRelay.hpp"

#include <functional>

namespace mo
{
    template <typename Sig>
    class TSlot
    {
    };
    template <typename Sig, class Mutex>
    class TSignalRelay;

    template <typename R, typename... T>
    class TSlot<R(T...)> : public std::function<R(T...)>, public ISlot
    {
      public:
        TSlot();
        TSlot(const std::function<R(T...)>& other);
        TSlot(std::function<R(T...)>&& other);
        ~TSlot() override;

        TSlot& operator=(const std::function<R(T...)>& other);
        TSlot& operator=(const TSlot& other);

        std::shared_ptr<Connection> connect(ISignal* sig) override;
        std::shared_ptr<Connection> connect(TSignal<R(T...)>* signal);
        std::shared_ptr<Connection> connect(std::shared_ptr<ISignalRelay>& relay) override;
        std::shared_ptr<Connection> connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay);
        virtual bool disconnect(std::weak_ptr<ISignalRelay> relay) override;
        void clear() override;
        const TypeInfo& getSignature() const override;
        operator bool() const;

      protected:
        std::vector<std::shared_ptr<TSignalRelay<R(T...)>>> _relays;
    };

    ///////////////////////////////////////////////////////////////////////
    /// IMPLEMENTATION
    ///////////////////////////////////////////////////////////////////////

    template <class R, class... T>
    TSlot<R(T...)>::TSlot()
    {
    }

    template <class R, class... T>
    TSlot<R(T...)>::TSlot(const std::function<R(T...)>& other)
        : std::function<R(T...)>(other)
    {
    }

    template <class R, class... T>
    TSlot<R(T...)>::TSlot(std::function<R(T...)>&& other)
        : std::function<R(T...)>(other)
    {
    }

    template <class R, class... T>
    TSlot<R(T...)>::~TSlot()
    {
        clear();
    }

    template <class R, class... T>
    TSlot<R(T...)>& TSlot<R(T...)>::operator=(const std::function<R(T...)>& other)
    {
        std::function<R(T...)>::operator=(other);
        return *this;
    }

    template <class R, class... T>
    TSlot<R(T...)>& TSlot<R(T...)>::operator=(const TSlot<R(T...)>& other)
    {
        this->_relays = other._relays;
        return *this;
    }

    template <class R, class... T>
    std::shared_ptr<Connection> TSlot<R(T...)>::connect(ISignal* sig)
    {
        auto typed = dynamic_cast<TSignal<R(T...)>*>(sig);
        if (typed)
        {
            return connect(typed);
        }
        return std::shared_ptr<Connection>();
    }

    template <class R, class... T>
    std::shared_ptr<Connection> TSlot<R(T...)>::connect(TSignal<R(T...)>* typed)
    {
        std::shared_ptr<TSignalRelay<R(T...)>> relay(new TSignalRelay<R(T...)>());
        relay->connect(this);
        typed->connect(relay);
        _relays.push_back(relay);
        return std::shared_ptr<Connection>(new SlotConnection(this, relay));
    }

    template <class R, class... T>
    std::shared_ptr<Connection> TSlot<R(T...)>::connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay)
    {
        relay->connect(this);
        _relays.push_back(relay);
        return std::shared_ptr<Connection>(new SlotConnection(this, std::dynamic_pointer_cast<ISignalRelay>(relay)));
    }

    template <class R, class... T>
    std::shared_ptr<Connection> TSlot<R(T...)>::connect(std::shared_ptr<ISignalRelay>& relay)
    {
        if (relay == nullptr)
        {
            relay.reset(new TSignalRelay<R(T...)>());
        }
        auto typed = std::dynamic_pointer_cast<TSignalRelay<R(T...)>>(relay);
        if (typed)
        {
            _relays.push_back(typed);
            if (relay->connect(this))
            {
                return std::shared_ptr<Connection>(new SlotConnection(this, relay));
            }
        }
        return std::shared_ptr<Connection>();
    }

    template <class R, class... T>
    bool TSlot<R(T...)>::disconnect(std::weak_ptr<ISignalRelay> relay_)
    {
        auto relay = relay_.lock();
        for (auto itr = _relays.begin(); itr != _relays.end(); ++itr)
        {
            if ((*itr) == relay)
            {
                (*itr)->disconnect(this);
                _relays.erase(itr);
                return true;
            }
        }
        return false;
    }

    template <class R, class... T>
    void TSlot<R(T...)>::clear()
    {
        for (auto& relay : _relays)
        {
            relay->disconnect(this);
        }
    }

    template <class R, class... T>
    const TypeInfo& TSlot<R(T...)>::getSignature() const
    {
        static TypeInfo type(typeid(R(T...)));
        return type;
    }

    template <class R, class... T>
    TSlot<R(T...)>::operator bool() const
    {
        return std::function<R(T...)>::operator bool();
    }
}
