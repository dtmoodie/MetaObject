#pragma once
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include "MetaObject/signals/Connection.hpp"

namespace mo
{
	template<class Sig> class TSlot;

	template<class R, class...T>
	TSlot<R(T...)>::TSlot()
	{
		
	}

	template<class R, class...T>
	TSlot<R(T...)>::TSlot(const std::function<R(T...)>& other) :
		std::function<R(T...)>(other)
	{
		
	}
    
    template<class R, class...T>
    TSlot<R(T...)>::TSlot(std::function<R(T...)>&& other):
        std::function<R(T...)>(other)
    {
    }

	template<class R, class...T> 
	TSlot<R(T...)>::~TSlot()
	{
        clear();
	}

	template<class R, class...T>
	TSlot<R(T...)>& TSlot<R(T...)>::operator=(const std::function<R(T...)>& other)
	{
		std::function<R(T...)>::operator=(other);
		return *this;
	}

	template<class R, class...T>
	TSlot<R(T...)>& TSlot<R(T...)>::operator=(const TSlot<R(T...)>& other)
	{
		this->_relays = other._relays;
		return *this;
	}


	template<class R, class...T> 
	std::shared_ptr<Connection> TSlot<R(T...)>::connect(ISignal* sig)
	{
		auto typed = dynamic_cast<TSignal<R(T...)>*>(sig);
		if (typed)
		{
			return connect(typed);
		}
		return std::shared_ptr<Connection>();
	}

	template<class R, class...T>
	std::shared_ptr<Connection> TSlot<R(T...)>::connect(TSignal<R(T...)>* typed)
	{
		std::shared_ptr<TSignalRelay<R(T...)>> relay(new TSignalRelay<R(T...)>());
		relay->connect(this);
        typed->connect(relay);
		_relays.push_back(relay);
		return std::shared_ptr<Connection>(new SlotConnection(this, relay));
	}
    template<class R, class...T>
    std::shared_ptr<Connection> TSlot<R(T...)>::connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay)
    {
        relay->connect(this);
        _relays.push_back(relay);
        return std::shared_ptr<Connection>(new SlotConnection(this, std::dynamic_pointer_cast<ISignalRelay>(relay)));
    }
	template<class R, class...T> 
	std::shared_ptr<Connection> TSlot<R(T...)>::connect(std::shared_ptr<ISignalRelay>& relay){
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

	template<class R, class...T>
	bool TSlot<R(T...)>::disConnect(std::weak_ptr<ISignalRelay> relay_)
	{
		auto relay = relay_.lock();
		for (auto itr = _relays.begin(); itr != _relays.end(); ++itr)
		{
			if ((*itr) == relay)
			{
				(*itr)->disConnect(this);
				_relays.erase(itr);
				return true;
			}
		}
		return false;
	}
    template<class R, class... T>
    void TSlot<R(T...)>::clear()
    {
        for (auto& relay : _relays)
        {
            relay->disConnect(this);
        }
    }

	template<class R, class...T> 
	TypeInfo TSlot<R(T...)>::getSignature() const
	{
		return TypeInfo(typeid(R(T...)));
	}
    template<class R, class...T>
    TSlot<R(T...)>::operator bool() const{
        return std::function<R(T...)>::operator bool();
    }
}
