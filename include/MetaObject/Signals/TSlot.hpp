#pragma once
#include "ISlot.hpp"
#include "TSignal.hpp"
#include "MetaObject/Context.hpp"
#include "TSignalRelay.hpp"
#include <functional>
#include <future>
namespace mo
{
    template<typename Sig> class TSlot{};
	template<typename Sig> class TSignalRelay;

    template<typename R, typename... T> 
    class TSlot<R(T...)>: public std::function<R(T...)>, public ISlot{
    public:
		TSlot();
		TSlot(const std::function<R(T...)>& other);
        TSlot(std::function<R(T...)>&& other);
		~TSlot();

		TSlot& operator=(const std::function<R(T...)>& other);
		TSlot& operator=(const TSlot& other);

		std::shared_ptr<Connection> connect(ISignal* sig);
		std::shared_ptr<Connection> connect(TSignal<R(T...)>* signal);
		std::shared_ptr<Connection> connect(std::shared_ptr<ISignalRelay>& relay);
        std::shared_ptr<Connection> connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay);
		virtual bool disConnect(std::weak_ptr<ISignalRelay> relay);
        void clear();
		TypeInfo getSignature() const;
	protected:
		std::vector< std::shared_ptr< TSignalRelay<R(T...)> > > _relays;
		
    };
}
#include "detail/TSlotImpl.hpp"
