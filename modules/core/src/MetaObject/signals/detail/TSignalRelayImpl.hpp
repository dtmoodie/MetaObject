#pragma once
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/signals/Connection.hpp"
#include "MetaObject/thread/InterThread.hpp"
#include <MetaObject/core/Context.hpp>
namespace mo
{
    template <class Sig>
    class TSignalRelay;

    template <class... T>
    void TSignalRelay<void(T...)>::operator()(TSignal<void(T...)>* sig, const T&... args)
    {
        std::unique_lock<std::mutex> lock(mtx);
        auto my_slots = _slots;
        lock.unlock();
        for (auto slot : my_slots)
        {
            auto slot_ctx = slot->getContext();
            auto sig_ctx = sig->getContext();
            if (slot_ctx && sig_ctx)
            {
                if (slot_ctx->processId() == sig_ctx->processId())
                {
                    if (slot_ctx->threadId() != sig_ctx->threadId())
                    {
                        ThreadSpecificQueue::push(
                            std::bind([slot](T... args) { (*slot)(args...); }, args...), slot_ctx->threadId(), slot);
                        continue;
                    }
                }
                else
                {
                    THROW(error) << "Not implemented yet";
                }
            }
            if (slot)
            {
                (*slot)(args...);
            }
        }
    }

    template <class... T>
    void TSignalRelay<void(T...)>::operator()(const T&... args)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto slot : _slots)
        {
            (*slot)(args...);
        }
    }

    template <class... T>
    void TSignalRelay<void(T...)>::operator()(Context* ctx, const T&... args)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto slot : _slots)
        {
            auto slot_ctx = slot->getContext();
            if (slot_ctx)
            {
                if (slot_ctx->processId() == ctx->processId())
                {
                    if (slot_ctx->threadId() != ctx->threadId())
                    {
                        ThreadSpecificQueue::push(
                            std::bind([slot](T... args) { (*slot)(args...); }, args...), slot_ctx->threadId(), slot);
                    }

                    continue;
                }
                else
                {
                    THROW(error) << "Not implemented yet";
                }
            }
            (*slot)(args...);
        }
    }

    template <class... T>
    bool TSignalRelay<void(T...)>::connect(ISignal* signal)
    {
        auto typed = dynamic_cast<TSignal<void(T...)>*>(signal);
        if (typed)
        {
            return connect(typed);
        }
        return false;
    }

    template <class... T>
    bool TSignalRelay<void(T...)>::connect(TSignal<void(T...)>* signal)
    {
        return true;
    }

    template <class... T>
    bool TSignalRelay<void(T...)>::connect(ISlot* slot)
    {
        auto typed = dynamic_cast<TSlot<void(T...)>*>(slot);
        if (typed)
        {
            return connect(typed);
        }
        return false;
    }

    template <class... T>
    bool TSignalRelay<void(T...)>::connect(TSlot<void(T...)>* slot)
    {
        std::lock_guard<std::mutex> lock(mtx);
        _slots.insert(slot);
        return true;
    }

    template <class... T>
    bool TSignalRelay<void(T...)>::disconnect(ISlot* slot)
    {
        std::lock_guard<std::mutex> lock(mtx);
        return _slots.erase(static_cast<TSlot<void(T...)>*>(slot)) > 0;
    }

    template <class... T>
    bool TSignalRelay<void(T...)>::disconnect(ISignal* signal)
    {
        return false; // Currently not storing signal information to cache the Connection types
    }

    template <class... T>
    const TypeInfo& TSignalRelay<void(T...)>::getSignature() const
    {
        static TypeInfo type(typeid(void(T...)));
        return type;
    }
    template <class... T>
    bool TSignalRelay<void(T...)>::hasSlots() const
    {
        return _slots.size() != 0;
    }

    // ------------------------------------------------------------------
    // Return value specialization

    template <class R, class... T>
    TSignalRelay<R(T...)>::TSignalRelay()
        : _slot(nullptr)
    {
    }

    template <class R, class... T>
    R TSignalRelay<R(T...)>::operator()(TSignal<R(T...)>* sig, const T&... args)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (_slot)
        {
            return (*_slot)(args...);
        }
        THROW(debug) << "Slot not Connected";
        return R();
    }

    template <class R, class... T>
    R TSignalRelay<R(T...)>::operator()(Context* ctx, const T&... args)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (_slot)
            return (*_slot)(args...);
        THROW(debug) << "Slot not Connected";
        return R();
    }

    template <class R, class... T>
    R TSignalRelay<R(T...)>::operator()(const T&... args)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (_slot && *_slot)
            return (*_slot)(args...);
        THROW(debug) << "Slot not Connected";
        return R();
    }

    template <class R, class... T>
    bool TSignalRelay<R(T...)>::connect(ISlot* slot)
    {
        auto typed = dynamic_cast<TSlot<R(T...)>*>(slot);
        if (typed)
        {
            return connect(typed);
        }
        return true;
    }

    template <class R, class... T>
    bool TSignalRelay<R(T...)>::connect(TSlot<R(T...)>* slot)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (_slot == slot)
            return false;
        _slot = slot;
        return true;
    }

    template <class R, class... T>
    bool TSignalRelay<R(T...)>::connect(ISignal* signal)
    {
        return true;
    }

    template <class R, class... T>
    bool TSignalRelay<R(T...)>::connect(TSignal<R(T...)>* sig)
    {
        return true;
    }

    template <class R, class... T>
    bool TSignalRelay<R(T...)>::disconnect(ISlot* slot)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (_slot == slot)
        {
            _slot = nullptr;
            return true;
        }
        return false;
    }

    template <class R, class... T>
    bool TSignalRelay<R(T...)>::disconnect(ISignal* signal)
    {
        return false;
    }

    template <class R, class... T>
    const TypeInfo& TSignalRelay<R(T...)>::getSignature() const
    {
        static TypeInfo type(typeid(R(T...)));
        return type;
    }

    template <class R, class... T>
    bool TSignalRelay<R(T...)>::hasSlots() const
    {
        if (_slot == nullptr)
            return false;
        return *_slot;
    }
}
