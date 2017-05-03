#pragma once
#include "ISignalRelay.hpp"
#include <set>
#include <mutex>
namespace mo
{
	template<class Sig> class TSlot;
	template<class Sig> class TSignal;
	template<class Sig> class TSignalRelay{	};
    class Context;
	template<class...T> class TSignalRelay<void(T...)>: public ISignalRelay
	{
	public:
		void operator()(TSignal<void(T...)>* sig, T&... args);
		void operator()(T&... args);
        void operator()(Context* ctx, T&... args);
		TypeInfo getSignature() const;
		bool HasSlots() const;
	protected:
		friend TSignal<void(T...)>;
		friend TSlot<void(T...)>;

		bool connect(ISlot* slot);
		bool  connect(ISignal* signal);

		bool connect(TSlot<void(T...)>* slot);
		bool connect(TSignal<void(T...)>* sig);

		bool disConnect(ISlot* slot);
		bool disConnect(ISignal* signal);
		
		std::set<TSlot<void(T...)>*> _slots;
        std::mutex mtx;
	};
	// Specialization for return value
	template<class R, class...T> class TSignalRelay<R(T...)>: public ISignalRelay
	{
	public:
		TSignalRelay();
		R operator()(TSignal<R(T...)>* sig, T&... args);
		R operator()(T&... args);
        R operator()(Context* ctx, T&... args);
		TypeInfo getSignature() const;
		bool HasSlots() const;
	protected:
		friend TSignal<R(T...)>;
		friend TSlot<R(T...)>;

		bool connect(ISlot* slot);
		bool connect(ISignal* signal);

		bool connect(TSlot<R(T...)>* slot);
		bool connect(TSignal<R(T...)>* sig);

		bool disConnect(ISlot* slot);
		bool disConnect(ISignal* signal);
		
		TSlot<R(T...)>* _slot;
        std::mutex mtx;
	};
}
#include "detail/TSignalRelayImpl.hpp"
