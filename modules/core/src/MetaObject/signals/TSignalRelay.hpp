#pragma once
#include "ISignalRelay.hpp"
#include <mutex>
#include <set>
namespace mo
{
    template <class Sig>
    class TSlot;
    template <class Sig>
    class TSignal;
    template <class Sig>
    class TSignalRelay
    {
    };
    class Context;
    template <class... T>
    class TSignalRelay<void(T...)> : public ISignalRelay
    {
      public:
        void operator()(TSignal<void(T...)>* sig, const T&... args);
        void operator()(const T&... args);
        void operator()(Context* ctx, const T&... args);
        const TypeInfo& getSignature() const;
        bool hasSlots() const;

      protected:
        friend TSignal<void(T...)>;
        friend TSlot<void(T...)>;

        bool connect(ISlot* slot);
        bool connect(ISignal* signal);

        bool connect(TSlot<void(T...)>* slot);
        bool connect(TSignal<void(T...)>* sig);

        bool disconnect(ISlot* slot);
        bool disconnect(ISignal* signal);

        std::set<TSlot<void(T...)>*> _slots;
        std::mutex mtx;
    };

    // Specialization for return value
    template <class R, class... T>
    class TSignalRelay<R(T...)> : public ISignalRelay
    {
      public:
        TSignalRelay();
        R operator()(TSignal<R(T...)>* sig, const T&... args);
        R operator()(const T&... args);
        R operator()(Context* ctx, const T&... args);
        const TypeInfo& getSignature() const;
        bool hasSlots() const;

      protected:
        friend TSignal<R(T...)>;
        friend TSlot<R(T...)>;

        bool connect(ISlot* slot);
        bool connect(ISignal* signal);

        bool connect(TSlot<R(T...)>* slot);
        bool connect(TSignal<R(T...)>* sig);

        bool disconnect(ISlot* slot);
        bool disconnect(ISignal* signal);

        TSlot<R(T...)>* _slot;
        std::mutex mtx;
    };
}
#include "detail/TSignalRelayImpl.hpp"
