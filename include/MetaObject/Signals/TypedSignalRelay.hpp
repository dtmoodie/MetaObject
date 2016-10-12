#pragma once
#include "ISignalRelay.hpp"
#include <set>
namespace mo
{
	template<class Sig> class TypedSlot;
	template<class Sig> class TypedSignal;
	template<class Sig> class TypedSignalRelay{	};

	template<class...T> class TypedSignalRelay<void(T...)>: public ISignalRelay
	{
	public:
		void operator()(TypedSignal<void(T...)>* sig, T&... args);
		void operator()(T&... args);
        void operator()(Context* ctx, T&... args);
		TypeInfo GetSignature() const;
		bool HasSlots() const;
	protected:
		friend TypedSignal<void(T...)>;
		friend TypedSlot<void(T...)>;

		bool Connect(ISlot* slot);
		bool  Connect(ISignal* signal);

		bool Connect(TypedSlot<void(T...)>* slot);
		bool Connect(TypedSignal<void(T...)>* sig);

		bool Disconnect(ISlot* slot);
		bool Disconnect(ISignal* signal);
		
		std::set<TypedSlot<void(T...)>*> _slots;
	};
	// Specialization for return value
	template<class R, class...T> class TypedSignalRelay<R(T...)>: public ISignalRelay
	{
	public:
		TypedSignalRelay();
		R operator()(TypedSignal<R(T...)>* sig, T&... args);
		R operator()(T&... args);
        R operator()(Context* ctx, T&... args);
		TypeInfo GetSignature() const;
		bool HasSlots() const;
	protected:
		friend TypedSignal<R(T...)>;
		friend TypedSlot<R(T...)>;

		bool Connect(ISlot* slot);
		bool Connect(ISignal* signal);

		bool Connect(TypedSlot<R(T...)>* slot);
		bool Connect(TypedSignal<R(T...)>* sig);

		bool Disconnect(ISlot* slot);
		bool Disconnect(ISignal* signal);
		
		TypedSlot<R(T...)>* _slot;
	};
}
#include "detail/TypedSignalRelayImpl.hpp"