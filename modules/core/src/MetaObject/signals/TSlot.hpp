#ifndef MO_SIGNALS_TSLOT_HPP
#define MO_SIGNALS_TSLOT_HPP
#include "ArgumentPack.hpp"
#include "Connection.hpp"
#include "ISlot.hpp"
#include "MetaObject/core/AsyncStream.hpp"
#include "TSignal.hpp"
#include "TSignalRelay.hpp"

#include <ct/bind.hpp>

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
        template <class... ARGS>
        TSlot(ARGS&&... args);
        ~TSlot() override;

        TSlot& operator=(const std::function<R(T...)>& other);
        TSlot& operator=(const TSlot& other);

        // R invokeArgpack(const ArgumentPack<T...>&);

        std::shared_ptr<Connection> connect(ISignal& sig) override;
        std::shared_ptr<Connection> connect(TSignal<R(T...)>& signal);
        std::shared_ptr<Connection> connect(std::shared_ptr<ISignalRelay>& relay) override;
        std::shared_ptr<Connection> connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay);
        bool disconnect(std::weak_ptr<ISignalRelay> relay) override;
        void clear() override;
        const TypeInfo& getSignature() const override;
        operator bool() const;

        template <class U, class V>
        void bind(R (U::*fptr)(T...), V* ptr)
        {
            (*this) = ct::variadicBind(fptr, ptr);
        }

      private:
        // template <int... Is>
        // R invokeArgpack(const ArgumentPack<T...>& arg_pack, ct::int_sequence<Is...>);
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
    template <class... ARGS>
    TSlot<R(T...)>::TSlot(ARGS&&... args)
        : std::function<R(T...)>(std::forward<ARGS>(args)...)
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

    /*template <class R, class... T>
    R TSlot<R(T...)>::invokeArgpack(const ArgumentPack<T...>& arg_pack)
    {
        return invokeArgpack(arg_pack, ct::make_int_sequence<sizeof...(T)>{});
    }

    template <class R, class... T>
    template <int... Is>
    R TSlot<R(T...)>::invokeArgpack(const ArgumentPack<T...>& arg_pack, ct::int_sequence<Is...>)
    {
        return (*this)(std::get<Is>(arg_pack.data)...);
    }*/

    template <class R, class... T>
    std::shared_ptr<Connection> TSlot<R(T...)>::connect(ISignal& sig)
    {
        auto typed = dynamic_cast<TSignal<R(T...)>*>(&sig);
        if (typed)
        {
            return connect(*typed);
        }
        return std::shared_ptr<Connection>();
    }

    template <class R, class... T>
    std::shared_ptr<Connection> TSlot<R(T...)>::connect(TSignal<R(T...)>& typed)
    {
        std::shared_ptr<TSignalRelay<R(T...)>> relay(new TSignalRelay<R(T...)>());
        relay->connect(*this);
        typed.connect(relay);
        _relays.push_back(relay);
        return std::shared_ptr<Connection>(new SlotConnection(this, relay));
    }

    template <class R, class... T>
    std::shared_ptr<Connection> TSlot<R(T...)>::connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay)
    {
        relay->connect(*this);
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
            if (relay->connect(*this))
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
                (*itr)->disconnect(*this);
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
            relay->disconnect(*this);
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
} // namespace mo
#endif // MO_SIGNALS_TSLOT_HPP