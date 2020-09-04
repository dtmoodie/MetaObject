#ifndef MO_PARAMS_TPublisher_HPP
#define MO_PARAMS_TPublisher_HPP

#include <MetaObject/params/IPublisher.hpp>
#include <MetaObject/params/ParamAllocator.hpp>
#include <MetaObject/params/TParam.hpp>

namespace mo
{

    template <typename T>
    struct MO_EXPORTS TPublisherImpl : TParam<IPublisher>
    {
        using TContainerPtr_t = std::shared_ptr<TDataContainer<T>>;
        using TContainerConstPtr_t = std::shared_ptr<const TDataContainer<T>>;

        TPublisherImpl();

        bool providesOutput(TypeInfo type) const override;
        std::vector<TypeInfo> getOutputTypes() const override;
        std::vector<Header> getAvailableHeaders() const override;
        boost::optional<Header> getNewestHeader() const override;

        IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* = nullptr) override;

        uint32_t getNumSubscribers() const override;

        void setAllocator(Allocator::Ptr_t alloc) override;

        // TODO create a few overloads of publish
        void publish(TContainerConstPtr_t data, IAsyncStream* = nullptr);
        void publish(const T& data, Header, IAsyncStream* = nullptr);
        void publish(T&& data, Header, IAsyncStream* = nullptr);
        template <class U, class... Ts>
        void publish(U&& data, Ts&&... args);

        // create data using m_allocator for publishing
        // This is more optimal than creating the data on your own because it can use
        // an allocator that is aware of how the data is used after publishing
        template <class... Args>
        TContainerPtr_t create(Args&&... args);

        std::ostream& print(std::ostream&) const override;

        ConnectionPtr_t registerUpdateNotifier(ISlot& f);

        ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr_t& relay_);

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

    template <class T>
    struct TPublisher : TPublisherImpl<T>
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

    template <typename T>
    TPublisherImpl<T>::TPublisherImpl()
    {
        this->setFlags(ParamFlags(ParamFlags::kOUTPUT));
        setAllocator(Allocator::getDefault());
    }

    template <typename T>
    bool TPublisherImpl<T>::providesOutput(const TypeInfo type) const
    {
        return type == TypeInfo::create<T>();
    }

    template <typename T>
    std::vector<TypeInfo> TPublisherImpl<T>::getOutputTypes() const
    {
        return {TypeInfo::create<T>()};
    }

    template <typename T>
    std::vector<Header> TPublisherImpl<T>::getAvailableHeaders() const
    {
        std::vector<Header> out;
        Mutex_t::Lock_t lock(this->mtx());
        if (m_data)
        {
            out.push_back(m_data->getHeader());
        }
        return out;
    }

    template <typename T>
    boost::optional<Header> TPublisherImpl<T>::getNewestHeader() const
    {
        Mutex_t::Lock_t lock(this->mtx());
        if (m_data)
        {
            return m_data->getHeader();
        }
        return {};
    }

    template <typename T>
    IDataContainerConstPtr_t TPublisherImpl<T>::getData(const Header* desired, IAsyncStream* stream)
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

    template <typename T>
    void TPublisherImpl<T>::publish(TContainerConstPtr_t data, IAsyncStream* stream)
    {
        if (stream == nullptr)
        {
            stream = this->getStream();
        }
        if (stream == nullptr)
        {
            stream = &IAsyncStream::currentRef();
        }
        m_data = data;
        MO_ASSERT_LOGGER(getLogger(), stream != nullptr);
        // TODO finish
        m_update_signal(data, *this, ct::value(UpdateFlags::kVALUE_UPDATED), *stream);
        emitUpdate(data->header, ct::value(UpdateFlags::kVALUE_UPDATED), *stream);
    }

    template <typename T>
    void TPublisherImpl<T>::publish(const T& data, Header hdr, IAsyncStream* stream)
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

    template <typename T>
    void TPublisherImpl<T>::publish(T&& data, Header hdr, IAsyncStream* stream)
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

    template <typename T>
    template <class U, class... Ts>
    void TPublisherImpl<T>::publish(U&& data, Ts&&... args)
    {
        auto stream = getKeywordInputDefault<tags::Stream>(nullptr, std::forward<Ts>(args)...);

        auto header = getKeywordInputDefault<tags::Header>(Header(), std::forward<Ts>(args)...);
        auto timestamp = getKeywordInputOptional<tags::Timestamp>(std::forward<Ts>(args)...);
        auto fn = getKeywordInputOptional<tags::FrameNumber>(std::forward<Ts>(args)...);
        auto src = getKeywordInputOptional<tags::Source>(std::forward<Ts>(args)...);
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

    template <typename T>
    template <class... Args>
    typename TPublisherImpl<T>::TContainerPtr_t TPublisherImpl<T>::create(Args&&... args)
    {
        return std::make_shared<TDataContainer<T>>(m_allocator, std::forward<Args>(args)...);
    }

    template <typename T>
    uint32_t TPublisherImpl<T>::getNumSubscribers() const
    {
        Mutex_t::Lock_t lock(this->mtx());
        return m_update_signal.numSlots();
    }

    template <typename T>
    void TPublisherImpl<T>::setAllocator(typename Allocator::Ptr_t alloc)
    {
        Mutex_t::Lock_t lock(this->mtx());
        m_allocator = ParamAllocator::create(std::move(alloc));
    }

    template <typename T>
    std::ostream& TPublisherImpl<T>::print(std::ostream& os) const
    {
        Mutex_t::Lock_t lock(this->mtx());
        os << this->getTreeName();
        os << " type: " << this->getOutputTypes();
        return os;
    }

    template <typename T>
    ConnectionPtr_t TPublisherImpl<T>::registerUpdateNotifier(ISlot& f)
    {
        if (f.getSignature() == m_update_signal.getSignature())
        {
            return m_update_signal.connect(f);
        }
        return TParam<IPublisher>::registerUpdateNotifier(f);
    }

    template <typename T>
    ConnectionPtr_t TPublisherImpl<T>::registerUpdateNotifier(const ISignalRelay::Ptr_t& relay_)
    {
        if (relay_->getSignature() == m_update_signal.getSignature())
        {
            auto tmp = relay_;
            return m_update_signal.connect(tmp);
        }
        return TParam<IPublisher>::registerUpdateNotifier(relay_);
    }

    template <typename T>
    typename TPublisherImpl<T>::TContainerConstPtr_t TPublisherImpl<T>::getCurrentData() const
    {
        Mutex_t::Lock_t lock(this->mtx());
        return m_data;
    }

    template <typename T>
    void TPublisherImpl<T>::setCurrentData(const TContainerConstPtr_t data)
    {
        Mutex_t::Lock_t lock(this->mtx());
        m_data = std::move(data);
    }

} // namespace mo
#endif // MO_PARAMS_TPublisher_HPP
