#ifndef MO_PARAMS_TPublisher_HPP
#define MO_PARAMS_TPublisher_HPP

#include <MetaObject/params/IPublisher.hpp>
#include <MetaObject/params/ParamAllocator.hpp>
#include <MetaObject/params/TParam.hpp>
#include <MetaObject/runtime_reflection/StructTraits.hpp>
#include <MetaObject/runtime_reflection/VisitorTraits.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/memory.hpp>

namespace mo
{

    template <class T>
    struct MO_EXPORTS TPublisher<T, 0> : TParam<IPublisher>
    {
        using TContainerPtr_t = std::shared_ptr<TDataContainer<T>>;
        using TContainerConstPtr_t = std::shared_ptr<const TDataContainer<T>>;

        TPublisher();

        bool providesOutput(TypeInfo type) const override;
        std::vector<TypeInfo> getOutputTypes() const override;
        std::vector<Header> getAvailableHeaders() const override;
        boost::optional<Header> getNewestHeader() const override;

        IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* = nullptr) override;

        uint32_t getNumSubscribers() const override;

        void setAllocator(Allocator::Ptr_t alloc) override;

        // TODO create a few overloads of publish
        void publish(const TContainerConstPtr_t& data, IAsyncStream* = nullptr);
        void publish(const TContainerPtr_t& data, IAsyncStream* = nullptr);
        void publish(const T& data, Header, IAsyncStream* = nullptr);
        void publish(T&& data, Header, IAsyncStream* = nullptr);
        template <class... Ts>
        auto publish(T data, Ts&&... args)
            -> void; // ct::EnableIf<ct::IsBase<ct::Base<T>, ct::Derived<ct::decay_t<U>>>::value ||
                     // std::is_same<ct::decay_t<U>, T>::value>;

        // create data using m_allocator for publishing
        // This is more optimal than creating the data on your own because it can use
        // an allocator that is aware of how the data is used after publishing
        template <class... Args>
        TContainerPtr_t create(Args&&... args);

        std::ostream& print(std::ostream&) const override;

        ConnectionPtr_t registerUpdateNotifier(ISlot& f);

        ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr_t& relay_);

        void load(ILoadVisitor& visitor) override;
        void save(ISaveVisitor& visitor) const override;
        void visit(StaticVisitor& visitor) const override;

      protected:
        TContainerConstPtr_t getCurrentData() const;
        void setCurrentData(const TContainerConstPtr_t);

      private:
        TContainerConstPtr_t m_data;
        uint32_t m_num_subscribers = 0;
        ParamAllocator::Ptr_t m_allocator;

        TSignal<DataUpdate_s> m_update_signal;
        FrameNumber m_update_counter = 0;
    };

    template <class T, uint8_t P>
    struct MO_EXPORTS TPublisher : TPublisher<T, P - 1>
    {
    };

    template <class T, uint64_t FLAG = ParamFlags::kOUTPUT>
    struct MO_EXPORTS TFPublisher : TPublisher<T>
    {
        TFPublisher()
        {
            this->setFlags(FLAG);
        }
    };
    ////////////////////////////////////////////////
    // implementation
    ////////////////////////////////////////////////

    template <class T>
    TPublisher<T, 0>::TPublisher()
    {
        this->setFlags(ParamFlags(ParamFlags::kOUTPUT));
        setAllocator(Allocator::getDefault());
    }

    template <class T>
    bool TPublisher<T, 0>::providesOutput(const TypeInfo type) const
    {
        return type == TypeInfo::create<T>();
    }

    template <class T>
    std::vector<TypeInfo> TPublisher<T, 0>::getOutputTypes() const
    {
        return {TypeInfo::create<T>()};
    }

    template <class T>
    std::vector<Header> TPublisher<T, 0>::getAvailableHeaders() const
    {
        std::vector<Header> out;
        Mutex_t::Lock_t lock(this->mtx());
        if (m_data)
        {
            out.push_back(m_data->getHeader());
        }
        return out;
    }

    template <class T>
    boost::optional<Header> TPublisher<T, 0>::getNewestHeader() const
    {
        Mutex_t::Lock_t lock(this->mtx());
        if (m_data)
        {
            return m_data->getHeader();
        }
        return {};
    }

    template <class T>
    IDataContainerConstPtr_t TPublisher<T, 0>::getData(const Header* desired, IAsyncStream* stream)
    {
        // TODO stream synchronization
        IDataContainerConstPtr_t out;
        if (desired)
        {
            Mutex_t::Lock_t lock(this->mtx());
            if (m_data)
            {
                if (m_data->getHeader() == *desired)
                {
                    out = m_data;
                }
            }
        }
        else
        {
            Mutex_t::Lock_t lock(this->mtx());
            out = m_data;
        }
        return out;
    }

    template <class T>
    void TPublisher<T, 0>::publish(const TContainerConstPtr_t& data, IAsyncStream* stream)
    {
        if (stream == nullptr)
        {
            stream = this->getStream();
        }
        if (stream == nullptr)
        {
            stream = IAsyncStream::current().get();
        }
        m_data = data;
        // TODO finish
        m_update_signal(data, *this, ct::value(UpdateFlags::kVALUE_UPDATED), stream);
        emitUpdate(data->header, ct::value(UpdateFlags::kVALUE_UPDATED), stream);
    }

    template <class T>
    void TPublisher<T, 0>::publish(const TContainerPtr_t& data, IAsyncStream* stream)
    {
        publish(TContainerConstPtr_t(data), stream);
    }

    template <class T>
    void TPublisher<T, 0>::publish(const T& data, Header hdr, IAsyncStream* stream)
    {
        auto out = std::make_shared<TDataContainer<T>>(m_allocator, data);
        if (!hdr.frame_number.valid())
        {
            hdr.frame_number = m_update_counter;
            ++m_update_counter;
        }
        out->header = std::move(hdr);
        publish(TContainerConstPtr_t(std::move(out)), stream);
    }

    template <class T>
    void TPublisher<T, 0>::publish(T&& data, Header hdr, IAsyncStream* stream)
    {
        auto out = std::make_shared<TDataContainer<T>>(m_allocator, std::move(data));
        if (!hdr.frame_number.valid())
        {
            hdr.frame_number = m_update_counter;
            ++m_update_counter;
        }
        out->header = std::move(hdr);
        publish(TContainerConstPtr_t(std::move(out)), stream);
    }

    template <class T>
    template <class... Ts>
    auto TPublisher<T, 0>::publish(T data, Ts&&... args)
        -> void // ct::EnableIf<ct::IsBase<ct::Base<T>, ct::Derived<ct::decay_t<U>>>::value ||
                // std::is_same<ct::decay_t<U>, T>::value>
    {
        auto stream = getKeywordInputDefault<tags::Stream>(nullptr, std::forward<Ts>(args)...);

        auto header = getKeywordInputDefault<tags::Header>(Header(), std::forward<Ts>(args)...);
        auto timestamp = getKeywordInputOptional<tags::Timestamp>(std::forward<Ts>(args)...);
        auto fn = getKeywordInputOptional<tags::FrameNumber>(std::forward<Ts>(args)...);
        auto src = getKeywordInputOptional<tags::Source>(std::forward<Ts>(args)...);
        auto param = getKeywordInputOptional<tags::Param>(std::forward<Ts>(args)...);
        if (param)
        {
            auto header_ = param->getNewestHeader();
            if (header_)
            {
                header = *header_;
            }
        }
        if (timestamp)
        {
            header.timestamp = *timestamp;
        }
        if (fn)
        {
            header.frame_number = *fn;
        }
        if (src)
        {
            header.source_id = *src;
        }
        auto out = std::make_shared<TDataContainer<T>>(m_allocator, std::move(data));
        if (!header.frame_number.valid())
        {
            header.frame_number = m_update_counter;
            ++m_update_counter;
        }
        out->header = std::move(header);
        publish(TContainerConstPtr_t(std::move(out)), stream);
    }

    template <class T>
    template <class... Args>
    typename TPublisher<T, 0>::TContainerPtr_t TPublisher<T, 0>::create(Args&&... args)
    {
        return std::make_shared<TDataContainer<T>>(m_allocator, std::forward<Args>(args)...);
    }

    template <class T>
    uint32_t TPublisher<T, 0>::getNumSubscribers() const
    {
        Mutex_t::Lock_t lock(this->mtx());
        return m_update_signal.numSlots();
    }

    template <class T>
    void TPublisher<T, 0>::setAllocator(typename Allocator::Ptr_t alloc)
    {
        Mutex_t::Lock_t lock(this->mtx());
        m_allocator = ParamAllocator::create(std::move(alloc));
    }

    template <class T>
    std::ostream& TPublisher<T, 0>::print(std::ostream& os) const
    {
        Mutex_t::Lock_t lock(this->mtx());
        os << this->getTreeName();
        os << " type: " << this->getOutputTypes();
        return os;
    }

    template <class T>
    ConnectionPtr_t TPublisher<T, 0>::registerUpdateNotifier(ISlot& f)
    {
        if (f.getSignature() == m_update_signal.getSignature())
        {
            return m_update_signal.connect(f);
        }
        return TParam<IPublisher>::registerUpdateNotifier(f);
    }

    template <class T>
    ConnectionPtr_t TPublisher<T, 0>::registerUpdateNotifier(const ISignalRelay::Ptr_t& relay_)
    {
        if (relay_->getSignature() == m_update_signal.getSignature())
        {
            auto tmp = relay_;
            return m_update_signal.connect(tmp);
        }
        return TParam<IPublisher>::registerUpdateNotifier(relay_);
    }

    template <class T>
    void TPublisher<T, 0>::load(ILoadVisitor& visitor)
    {
    }

    template <class T>
    void TPublisher<T, 0>::save(ISaveVisitor& visitor) const
    {
        TContainerConstPtr_t data = this->getCurrentData();
        if (data)
        {
            visitor(&data, "data");
        }
    }

    template <class T>
    void TPublisher<T, 0>::visit(StaticVisitor& visitor) const
    {
    }

    template <class T>
    typename TPublisher<T, 0>::TContainerConstPtr_t TPublisher<T, 0>::getCurrentData() const
    {
        Mutex_t::Lock_t lock(this->mtx());
        return m_data;
    }

    template <class T>
    void TPublisher<T, 0>::setCurrentData(const TContainerConstPtr_t data)
    {
        Mutex_t::Lock_t lock(this->mtx());
        m_data = std::move(data);
    }

} // namespace mo
#endif // MO_PARAMS_TPublisher_HPP
