#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <MetaObject/Detail/TypeInfo.h>
#include "MetaObject/Detail/Placeholders.h"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include "MetaObject/Thread/InterThread.hpp"
#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/Signals/Connection.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Context.hpp"
#include "MetaObject/Signals/TypedSignalRelay.hpp"


namespace mo
{
    class IMetaObject;
	class Context;
    template<class Sig> class TypedSignal{};

	template<class...T> class TypedSignal<void(T...)> : public ISignal
	{
	public:
		TypedSignal();
		void operator()(T... args);
		TypeInfo GetSignature() const;

		std::shared_ptr<Connection> Connect(ISlot* slot);
		std::shared_ptr<Connection> Connect(std::shared_ptr<ISignalRelay>& relay);
		std::shared_ptr<Connection> Connect(std::shared_ptr<TypedSignalRelay<void(T...)>>& relay);
		bool Disconnect();
		bool Disconnect(ISlot* slot);
		bool Disconnect(std::weak_ptr<ISignalRelay> relay);
	protected:
		std::vector<std::shared_ptr<TypedSignalRelay<void(T...)>>> _typed_relays;
	private:
		static SignalRelayFactory<void(T...)> _relay_factory;
	};

	template<class R, class...T> class TypedSignal<R(T...)> : public ISignal
    {
    public:
		TypedSignal();
		R operator()(T... args);
		TypeInfo GetSignature() const;

		std::shared_ptr<Connection> Connect(ISlot* slot);
		std::shared_ptr<Connection> Connect(std::shared_ptr<ISignalRelay>& relay);
		std::shared_ptr<Connection> Connect(std::shared_ptr<TypedSignalRelay<R(T...)>>& relay);
		bool Disconnect();
		bool Disconnect(ISlot* slot);
		bool Disconnect(std::weak_ptr<ISignalRelay> relay);
	protected:
		std::shared_ptr<TypedSignalRelay<R(T...)>> _typed_relay;
	private:
		static SignalRelayFactory<R(T...)> _relay_factory;
    };
}
#include "detail/TypedSignalImpl.hpp"