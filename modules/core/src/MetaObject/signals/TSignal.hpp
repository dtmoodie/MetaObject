#ifndef MO_SIGNALS_TSIGNAL_HPP
#define MO_SIGNALS_TSIGNAL_HPP
#include <MetaObject/core/detail/forward.hpp>

#include "MetaObject/core/IAsyncStream.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/logging/logging.hpp"

#include "Connection.hpp"
#include "ISignal.hpp"
#include "ISignalRelay.hpp"
#include "TSlot.hpp"

#include <ct/VariadicTypedef.hpp>
#include <ct/type_traits.hpp>
#include <ct/variadic_indexing.hpp>

#include <memory>
#include <mutex>
#include <vector>

namespace mo
{
    template <class Sig, class Mutex>
    class TSignalRelay;

    template <class Sig>
    class TSignalImpl;

    template <class... T>
    class MO_EXPORTS TSignalImpl<void(T...)> : public ISignal
    {
      public:
        TSignalImpl();
        template <class... U>
        void operator()(U&&... args) const;
        template <class... U>
        void operator()(IAsyncStream* stream, U&&... args) const;

        TypeInfo getSignature() const override;

        ConnectionPtr_t connect(ISlot& slot) override;
        ConnectionPtr_t connect(ISignalRelay::Ptr_t& relay) override;
        ConnectionPtr_t connect(std::shared_ptr<TSignalRelay<void(T...)>>& relay);

        bool disconnect() override;
        bool disconnect(const ISlot& slot_) override;
        bool disconnect(const ISignalRelay& relay_) override;
        bool isConnected() const override;
        virtual uint32_t numSlots() const override;

      protected:
        mutable std::recursive_mutex m_mtx;
        std::vector<typename TSignalRelay<void(T...)>::Ptr_t> m_typed_relays;
    };

    template <class R, class... T>
    class MO_EXPORTS TSignalImpl<R(T...)> : public ISignal
    {
      public:
        TSignalImpl();
        template <class... U>
        R operator()(U&&... args) const;
        template <class... U>
        R operator()(IAsyncStream* ctx, U&&... args) const;

        TypeInfo getSignature() const override;

        ConnectionPtr_t connect(ISlot& slot) override;
        ConnectionPtr_t connect(ISignalRelay::Ptr_t& relay) override;
        ConnectionPtr_t connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay);

        bool disconnect() override;
        bool disconnect(const ISlot& slot) override;
        bool disconnect(const ISignalRelay& relay) override;
        bool isConnected() const override;
        virtual uint32_t numSlots() const override;

      protected:
        mutable std::recursive_mutex m_mtx;
        typename TSignalRelay<R(T...)>::Ptr_t m_typed_relay;
    };

    template <class T, class E>
    class TSignal;

    template <class T>
    struct IsStream : std::is_same<T, IAsyncStream*>
    {
    };

    template <class... T>
    using EnableIfContainsStream =
        ct::EnableIf<ct::VariadicTypedef<ct::decay_t<T>...>::template contains<IAsyncStream>()>;

    template <class... T>
    using DisableIfContainsStream =
        ct::EnableIf<!ct::VariadicTypedef<ct::decay_t<T>...>::template contains<IAsyncStream>()>;

    template <class R, class... T>
    class TSignal<R(T...), EnableIfContainsStream<T...>> : public TSignalImpl<R(T...)>
    {
        static constexpr int32_t STREAM_INDEX = ct::VariadicTypedef<ct::decay_t<T>...>::template indexOf<IsStream>();

      public:
        template <class... U>
        R operator()(U&&... args) const
        {
            IAsyncStream* stream = ct::get<STREAM_INDEX>(std::forward<U>(args)...);
            return TSignalImpl<R(T...)>::operator()(stream, std::forward<U>(args)...);
        }
    };

    template <class... T>
    class TSignal<void(T...), EnableIfContainsStream<T...>> : public TSignalImpl<void(T...)>
    {
        static constexpr int32_t STREAM_INDEX = ct::VariadicTypedef<ct::decay_t<T>...>::template indexOf<IsStream>();

      public:
        template <class... U>
        void operator()(U&&... args) const
        {
            IAsyncStream* stream = ct::get<STREAM_INDEX>(std::forward<U>(args)...);
            TSignalImpl<void(T...)>::operator()(stream, std::forward<U>(args)...);
        }
    };

    template <class R, class... T>
    class TSignal<R(T...), DisableIfContainsStream<T...>> : public TSignalImpl<R(T...)>
    {
      public:
        template <class... U>
        R operator()(U&&... args) const
        {
            return TSignalImpl<R(T...)>::operator()(std::forward<U>(args)...);
        }

        template <class... U>
        R operator()(IAsyncStream* stream, U&&... args) const
        {
            return TSignalImpl<R(T...)>::operator()(stream, std::forward<U>(args)...);
        }
    };

    template <class... T>
    class TSignal<void(T...), DisableIfContainsStream<T...>> : public TSignalImpl<void(T...)>
    {
      public:
        template <class... U>
        void operator()(U&&... args) const
        {
            TSignalImpl<void(T...)>::operator()(std::forward<U>(args)...);
        }

        template <class... U>
        void operator()(IAsyncStream* stream, U&&... args) const
        {
            TSignalImpl<void(T...)>::operator()(stream, std::forward<U>(args)...);
        }
    };

    /////////////////////////////////////////////////////////////////////////
    ///  Implementation
    /////////////////////////////////////////////////////////////////////////

    template <class R, class... T>
    TSignalImpl<R(T...)>::TSignalImpl()
    {
    }

    template <class R, class... T>
    template <class... U>
    R TSignalImpl<R(T...)>::operator()(U&&... args) const
    {
        auto stream = this->getStream();
        if (!stream)
        {
            stream = IAsyncStream::current().get();
        }
        return (*this)(stream, std::forward<U>(args)...);
    }

    template <class R, class... T>
    template <class... U>
    R TSignalImpl<R(T...)>::operator()(IAsyncStream* stream, U&&... args) const
    {
        typename TSignalRelay<R(T...)>::Ptr_t relay;
        {
            std::lock_guard<std::recursive_mutex> lock(m_mtx);
            relay = m_typed_relay;
        }
        if (relay)
        {
            return (*relay)(stream, std::forward<U>(args)...);
        }
        THROW(debug, "Not Connected to a signal relay");
        return R();
    }

    template <class R, class... T>
    TypeInfo TSignalImpl<R(T...)>::getSignature() const
    {
        return TypeInfo::create<R(T...)>();
    }

    template <class R, class... T>
    ConnectionPtr_t TSignalImpl<R(T...)>::connect(ISlot& slot)
    {
        return slot.connect(*this);
    }

    template <class R, class... T>
    ConnectionPtr_t TSignalImpl<R(T...)>::connect(std::shared_ptr<ISignalRelay>& relay)
    {
        if (relay == nullptr)
        {
            relay.reset(new TSignalRelay<R(T...)>());
        }
        auto typed = std::dynamic_pointer_cast<TSignalRelay<R(T...)>>(relay);
        if (typed)
        {
            return connect(typed);
        }
        return ConnectionPtr_t();
    }

    template <class R, class... T>
    ConnectionPtr_t TSignalImpl<R(T...)>::connect(std::shared_ptr<TSignalRelay<R(T...)>>& relay)
    {
        if (relay == nullptr)
        {
            relay.reset(new TSignalRelay<R(T...)>());
        }
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        if (relay != m_typed_relay)
        {
            m_typed_relay = relay;
            return std::make_shared<SignalConnection>(this, relay);
        }
        return ConnectionPtr_t();
    }

    template <class R, class... T>
    bool TSignalImpl<R(T...)>::disconnect()
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        if (m_typed_relay)
        {
            m_typed_relay.reset();
            return true;
        }
        return false;
    }

    template <class R, class... T>
    bool TSignalImpl<R(T...)>::disconnect(const ISlot& slot)
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        if (m_typed_relay)
        {
            return m_typed_relay->disconnect(slot);
        }
        return false;
    }

    template <class R, class... T>
    bool TSignalImpl<R(T...)>::disconnect(const ISignalRelay& relay)
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        if (m_typed_relay.get() == &relay)
        {
            m_typed_relay.reset();
            return true;
        }
        return false;
    }

    template <class R, class... T>
    bool TSignalImpl<R(T...)>::isConnected() const
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        return m_typed_relay != nullptr;
    }

    template <class R, class... T>
    uint32_t TSignalImpl<R(T...)>::numSlots() const
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        if (m_typed_relay)
        {
            return m_typed_relay->numSlots();
        }
        return 0;
    }

    // ---------------------------------------------------------------------
    // void specialization
    template <class... T>
    TSignalImpl<void(T...)>::TSignalImpl()
    {
    }

    template <class... T>
    template <class... U>
    void TSignalImpl<void(T...)>::operator()(U&&... args) const
    {
        auto stream = this->getStream();
        if (stream == nullptr)
        {
            stream = IAsyncStream::current().get();
        }
        (*this)(stream, std::forward<U>(args)...);
    }
    template <class... T>
    template <class... U>
    void TSignalImpl<void(T...)>::operator()(IAsyncStream* stream, U&&... args) const
    {
        std::unique_lock<std::recursive_mutex> lock(m_mtx);
        auto relays = m_typed_relays;
        lock.unlock();
        for (auto& relay : m_typed_relays)
        {
            if (relay)
            {
                (*relay)(stream, std::forward<U>(args)...);
            }
        }
    }

    template <class... T>
    TypeInfo TSignalImpl<void(T...)>::getSignature() const
    {
        return TypeInfo::create<void(T...)>();
    }

    template <class... T>
    std::shared_ptr<Connection> TSignalImpl<void(T...)>::connect(ISlot& slot)
    {
        return slot.connect(*this);
    }

    template <class... T>
    std::shared_ptr<Connection> TSignalImpl<void(T...)>::connect(std::shared_ptr<ISignalRelay>& relay)
    {
        if (relay == nullptr)
        {
            relay.reset(new TSignalRelay<void(T...)>());
        }
        auto typed = std::dynamic_pointer_cast<TSignalRelay<void(T...)>>(relay);
        if (typed)
        {
            return connect(typed);
        }

        return std::shared_ptr<Connection>();
    }

    template <class... T>
    std::shared_ptr<Connection> TSignalImpl<void(T...)>::connect(std::shared_ptr<TSignalRelay<void(T...)>>& relay)
    {
        if (relay == nullptr)
        {
            relay.reset(new TSignalRelay<void(T...)>());
        }
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        auto itr = std::find(m_typed_relays.begin(), m_typed_relays.end(), relay);
        if (itr == m_typed_relays.end())
        {
            m_typed_relays.push_back(relay);
            return std::shared_ptr<Connection>(new SignalConnection(this, relay));
        }
        return std::shared_ptr<Connection>();
    }

    template <class... T>
    bool TSignalImpl<void(T...)>::disconnect()
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        if (m_typed_relays.size())
        {
            m_typed_relays.clear();
            return true;
        }
        return false;
    }

    template <class... T>
    bool TSignalImpl<void(T...)>::disconnect(const ISlot& slot_)
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        for (auto relay = m_typed_relays.begin(); relay != m_typed_relays.end(); ++relay)
        {
            if ((*relay)->disconnect(slot_))
            {
                return true;
            }
        }
        return false;
    }

    template <class... T>
    bool TSignalImpl<void(T...)>::disconnect(const ISignalRelay& relay)
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        auto ptr = &relay;
        auto itr = std::find_if(
            m_typed_relays.begin(),
            m_typed_relays.end(),
            [ptr](const typename TSignalRelay<void(T...)>::Ptr_t& relay) -> bool { return relay.get() == ptr; });
        if (itr != m_typed_relays.end())
        {
            m_typed_relays.erase(itr);
            return true;
        }
        return false;
    }

    template <class... T>
    bool TSignalImpl<void(T...)>::isConnected() const
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        return !m_typed_relays.empty();
    }

    template <class... T>
    uint32_t TSignalImpl<void(T...)>::numSlots() const
    {
        std::lock_guard<std::recursive_mutex> lock(m_mtx);
        uint32_t slots = 0;
        for (const auto& relay : m_typed_relays)
        {
            slots += relay->numSlots();
        }
        return slots;
    }
} // namespace mo

namespace ct
{
    REFLECT_TEMPLATED_DERIVED(mo::TSignal, mo::ISignal)

    REFLECT_END;
} // namespace ct

#endif // MO_SIGNALS_TSIGNAL_HPP
