#ifndef MO_SIGNALS_TSIGNALRELAY_HPP
#define MO_SIGNALS_TSIGNALRELAY_HPP

#include <MetaObject/core/detail/forward.hpp>

#include "ISignalRelay.hpp"
#include <MetaObject/core/AsyncStream.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/signals/ArgumentPack.hpp>
#include <MetaObject/thread/Mutex.hpp>

#include <set>

namespace mo
{
    template <class Sig, class Mutex>
    class TSignalRelay
    {
    };

    struct IAsyncStream;
    template <class... T, class Mutex>
    class TSignalRelay<void(T...), Mutex> : public ISignalRelay
    {
      public:
        using Ptr_t = std::shared_ptr<TSignalRelay<void(T...)>>;
        using ConstPtr_t = std::shared_ptr<const TSignalRelay<void(T...)>>;

        template <class... U>
        void operator()(IAsyncStream& src_stream, U&&... args);
        TypeInfo getSignature() const override;
        bool hasSlots() const override;
        uint32_t numSlots() const override;

        bool connect(ISlot& slot) override;
        bool connect(ISignal& signal) override;

        bool connect(TSlot<void(T...)>& slot);
        bool connect(TSignal<void(T...)>& sig);

        bool disconnect(const ISlot& slot) override;
        bool disconnect(const ISignal& signal) override;

      private:
        std::set<TSlot<void(T...)>*> m_slots;
        mutable Mutex m_mtx;
    };

    // Specialization for return value
    template <class R, class... T, class Mutex>
    class TSignalRelay<R(T...), Mutex> : public ISignalRelay
    {
      public:
        using Ptr_t = std::shared_ptr<TSignalRelay<R(T...)>>;
        using ConstPtr_t = std::shared_ptr<const TSignalRelay<R(T...)>>;

        TSignalRelay();

        template <class... U>
        R operator()(IAsyncStream& src_stream, U&&... args);

        TypeInfo getSignature() const override;
        bool hasSlots() const override;
        uint32_t numSlots() const override;

        bool connect(ISlot& slot) override;
        bool connect(ISignal& signal) override;

        bool connect(TSlot<R(T...)>& slot);
        bool connect(TSignal<R(T...)>& sig);

        bool disconnect(const ISlot& slot) override;
        bool disconnect(const ISignal& signal) override;

      private:
        TSlot<R(T...)>* m_slot;
        mutable Mutex m_mtx;
    };

    //////////////////////////////////////////////////////////////////
    ///                   Implementation
    //////////////////////////////////////////////////////////////////

    template <class... T, class Mutex>
    template <class... U>
    void TSignalRelay<void(T...), Mutex>::operator()(IAsyncStream& src, U&&... args)
    {
        std::unique_lock<Mutex> lock(m_mtx);
        auto slots = m_slots;
        lock.unlock();
        for (auto slot : slots)
        {
            auto dst_stream = slot->getStream();
            if (dst_stream)
            {
                if (dst_stream->processId() == src.processId())
                {
                    if (dst_stream->threadId() != src.threadId())
                    {
                        // push as a task to the slot stream
                        // TODO argpack?
                        /*auto arg_pack =
                            std::make_shared<ArgumentPack<T...>>(src, *dst_stream, std::forward<T>(args)...);
                        dst_stream->pushWork([arg_pack, slot]() { slot->invokeArgpack(*arg_pack); });*/
                    }
                }
                else
                {
                    THROW(error, "IPC not implemented yet");
                }
            }
            else
            {
                // slot_stream == nullptr
                if (*slot)
                {
                    (*slot)(std::forward<U>(args)...);
                }
            }
        }
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::connect(ISignal& signal)
    {
        auto typed = dynamic_cast<TSignal<void(T...)>*>(&signal);
        if (typed)
        {
            return connect(*typed);
        }
        return false;
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::connect(TSignal<void(T...)>&)
    {
        return true;
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::connect(ISlot& slot)
    {
        auto typed = dynamic_cast<TSlot<void(T...)>*>(&slot);
        if (typed)
        {
            return connect(*typed);
        }
        return false;
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::connect(TSlot<void(T...)>& slot)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        m_slots.insert(&slot);
        return true;
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::disconnect(const ISlot& slot)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        for (auto itr = m_slots.begin(); itr != m_slots.end(); ++itr)
        {
            if (*itr == &slot)
            {
                m_slots.erase(itr);
                return true;
            }
        }
        return false;
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::disconnect(const ISignal&)
    {
        return false; // Currently not storing signal information to cache the Connection types
    }

    template <class... T, class Mutex>
    TypeInfo TSignalRelay<void(T...), Mutex>::getSignature() const
    {
        return TypeInfo::create<void(T...)>();
    }

    template <class... T, class Mutex>
    bool TSignalRelay<void(T...), Mutex>::hasSlots() const
    {
        std::lock_guard<Mutex> lock(m_mtx);
        return m_slots.size() != 0;
    }

    template <class... T, class Mutex>
    uint32_t TSignalRelay<void(T...), Mutex>::numSlots() const
    {
        std::lock_guard<Mutex> lock(m_mtx);
        return m_slots.size();
    }

    // ------------------------------------------------------------------
    // Return value specialization

    template <class R, class... T, class Mutex>
    TSignalRelay<R(T...), Mutex>::TSignalRelay()
        : m_slot(nullptr)
    {
    }

    // TODO the stream thing
    template <class R, class... T, class Mutex>
    template <class... U>
    R TSignalRelay<R(T...), Mutex>::operator()(IAsyncStream&, U&&... args)
    {
        std::unique_lock<Mutex> lock(m_mtx);
        auto slot = m_slot;
        lock.unlock();
        if (slot && *slot)
        {
            return (*slot)(std::forward<U>(args)...);
        }
        THROW(debug, "Slot not connected");
        return R();
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::connect(ISlot& slot)
    {
        auto typed = dynamic_cast<TSlot<R(T...)>*>(&slot);
        if (typed)
        {
            return connect(*typed);
        }
        return true;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::connect(TSlot<R(T...)>& slot)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        if (m_slot == &slot)
            return false;
        m_slot = &slot;
        return true;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::connect(ISignal&)
    {
        return true;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::connect(TSignal<R(T...)>&)
    {
        return true;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::disconnect(const ISlot& slot)
    {
        std::lock_guard<Mutex> lock(m_mtx);
        if (m_slot == &slot)
        {
            m_slot = nullptr;
            return true;
        }
        return false;
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::disconnect(const ISignal&)
    {
        return false;
    }

    template <class R, class... T, class Mutex>
    TypeInfo TSignalRelay<R(T...), Mutex>::getSignature() const
    {
        return TypeInfo::create<R(T...)>();
    }

    template <class R, class... T, class Mutex>
    bool TSignalRelay<R(T...), Mutex>::hasSlots() const
    {
        std::lock_guard<Mutex> lock(m_mtx);
        if (m_slot == nullptr)
            return false;
        return *m_slot;
    }

    template <class R, class... T, class Mutex>
    uint32_t TSignalRelay<R(T...), Mutex>::numSlots() const
    {
        std::lock_guard<Mutex> lock(m_mtx);
        return (m_slot != nullptr) ? 1 : 0;
    }
} // namespace mo

#endif // MO_SIGNALS_TSIGNALRELAY_HPP
