#pragma once
#include "ISignalRelay.hpp"

#include "MetaObject/logging/logging.hpp"
#include <MetaObject/core/AsyncStream.hpp>
#include <boost/fiber/mutex.hpp>

#include <set>

namespace mo
{
    template <class Sig>
    class TSlot;
    template <class Sig>
    class TSignal;

    template <class Sig, class Mutex>
    class TSignalRelay
    {
    };

    class IAsyncStream;
    template <class... T, class Mutex>
    class TSignalRelay<void(T...), Mutex> : public ISignalRelay
    {
      public:
        using Ptr = std::shared_ptr<TSignalRelay<void(T...)>>;
        using ConstPtr = std::shared_ptr<const TSignalRelay<void(T...)>>;

        void operator()(TSignal<void(T...)>* sig, const T&... args);
        void operator()(const T&... args);
        void operator()(IAsyncStream* ctx, const T&... args);
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

        std::set<TSlot<void(T...)>*> m_slots;
        Mutex m_mtx;
    };

    // Specialization for return value
    template <class R, class... T, class Mutex>
    class TSignalRelay<R(T...), Mutex> : public ISignalRelay
    {
      public:
        using Ptr = std::shared_ptr<TSignalRelay<R(T...)>>;
        using ConstPtr = std::shared_ptr<const TSignalRelay<R(T...)>>;

        TSignalRelay();
        R operator()(TSignal<R(T...)>* sig, const T&... args);
        R operator()(const T&... args);
        R operator()(IAsyncStream* ctx, const T&... args);
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

        TSlot<R(T...)>* m_slot;
        Mutex m_mtx;
    };

    //////////////////////////////////////////////////////////////////
    ///                   Implementation
    //////////////////////////////////////////////////////////////////

    template <class... T, class Mutex>
    void TSignalRelay<void(T...), Mutex>::operator()(TSignal<void(T...)>* sig, const T&... args)
    {
        std::unique_lock<Mutex> lock(m_mtx);
        auto mym_slots = m_slots;
        lock.unlock();
        for (auto slot : mym_slots)
        {
            auto slot_ctx = slot->getStream();
            auto sig_ctx = sig->getStream();
            if (slot_ctx && sig_ctx)
            {
                if (slot_ctx->processId() == sig_ctx->processId())
                {
                    if (slot_ctx->threadId() != sig_ctx->threadId())
                    {
                        // TODO fiber implementation
                        // ThreadSpecificQueue::push(
                        //    std::bind([slot](T... args) { (*slot)(args...); }, args...), slot_ctx->threadId(), slot);
                        continue;
                    }
                }
                else
                {
                    THROW(error, "Not implemented yet");
                }
            }
            if (slot)
            {
                (*slot)(args...);
            }
        }
    }

    template <class... T, class Mutex>
    void TSignalRelay<void(T...), Mutex>::operator()(const T&... args)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        for (auto slot : m_slots)
        {
            (*slot)(args...);
        }
    }

    template <class... T, class Mutex>
    void TSignalRelay<void(T...), Mutex>::operator()(IAsyncStream* ctx, const T&... args)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        for (auto slot : m_slots)
        {
            auto slot_stream = slot->getStream();
            if (slot_stream)
            {
                if (slot_stream->processId() == ctx->processId())
                {
                    if (slot_stream->threadId() != ctx->threadId())
                    {
                        // TODO fibers
                        // ThreadSpecificQueue::push(
                        //    std::bind([slot](T... args) { (*slot)(args...); }, args...), slot_ctx->threadId(), slot);
                    }

                    continue;
                }
                else
                {
                    THROW(error, "Not implemented yet");
                }
            }
            (*slot)(args...);
        }
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::connect(ISignal* signal)
    {
        auto typed = dynamic_cast<TSignal<void(T...)>*>(signal);
        if (typed)
        {
            return connect(typed);
        }
        return false;
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::connect(TSignal<void(T...)>*)
    {
        return true;
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::connect(ISlot* slot)
    {
        auto typed = dynamic_cast<TSlot<void(T...)>*>(slot);
        if (typed)
        {
            return connect(typed);
        }
        return false;
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::connect(TSlot<void(T...)>* slot)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        m_slots.insert(slot);
        return true;
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::disconnect(ISlot* slot)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        return m_slots.erase(static_cast<TSlot<void(T...)>*>(slot)) > 0;
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::disconnect(ISignal*)
    {
        return false; // Currently not storing signal information to cache the Connection types
    }

    template <class... T, class Mutex>
    const TypeInfo& TSignalRelay<void(T...), Mutex>::getSignature() const
    {
        static TypeInfo type(typeid(void(T...)));
        return type;
    }
    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::hasSlots() const
    {
        return m_slots.size() != 0;
    }

    // ------------------------------------------------------------------
    // Return value specialization

    template <class R, class... T, class Mutex>
    TSignalRelay<R(T...), Mutex>::TSignalRelay()
        : m_slot(nullptr)
    {
    }

    template <class R, class... T, class Mutex>
    R TSignalRelay<R(T...), Mutex>::operator()(TSignal<R(T...)>* sig, const T&... args)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        if (m_slot)
        {
            return (*m_slot)(args...);
        }
        THROW(debug, "Slot not connected");
        return R();
    }

    template <class R, class... T, class Mutex>
    R TSignalRelay<R(T...), Mutex>::operator()(IAsyncStream* ctx, const T&... args)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        if (m_slot)
        {
            return (*m_slot)(args...);
        }
        THROW(debug, "Slot not connected");
        return R();
    }

    template <class R, class... T, class Mutex>
    R TSignalRelay<R(T...), Mutex>::operator()(const T&... args)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        if (m_slot && *m_slot)
            return (*m_slot)(args...);
        THROW(debug, "Slot not connected");
        return R();
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::connect(ISlot* slot)
    {
        auto typed = dynamic_cast<TSlot<R(T...)>*>(slot);
        if (typed)
        {
            return connect(typed);
        }
        return true;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::connect(TSlot<R(T...)>* slot)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        if (m_slot == slot)
            return false;
        m_slot = slot;
        return true;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::connect(ISignal*)
    {
        return true;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::connect(TSignal<R(T...)>*)
    {
        return true;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::disconnect(ISlot* slot)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        if (m_slot == slot)
        {
            m_slot = nullptr;
            return true;
        }
        return false;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::disconnect(ISignal*)
    {
        return false;
    }

    template <class R, class... T, class Mutex>
    const TypeInfo& TSignalRelay<R(T...), Mutex>::getSignature() const
    {
        static TypeInfo type(typeid(R(T...)));
        return type;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::hasSlots() const
    {
        if (m_slot == nullptr)
            return false;
        return *m_slot;
    }
}
#include "detail/TSignalRelayImpl.hpp"
