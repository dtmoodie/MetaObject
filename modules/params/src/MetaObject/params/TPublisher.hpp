#ifndef MO_PARAMS_TPublisher_HPP
#define MO_PARAMS_TPublisher_HPP

#include <MetaObject/params/IPublisher.hpp>
#include <MetaObject/params/ParamAllocator.hpp>
#include <MetaObject/params/TParam.hpp>

namespace mo
{
    // this is a spot where specializations can be inserted into the TPublisher
    template <class T>
    struct MO_EXPORTS TPublisherDataPolicy : TParam<IPublisher>
    {

        using TContainerPtr_t = std::shared_ptr<TDataContainer<T>>;
        using TContainerConstPtr_t = std::shared_ptr<const TDataContainer<T>>;

      protected:
        TContainerConstPtr_t getCurrentData() const;
        void setCurrentData(TContainerConstPtr_t);

      private:
        TContainerConstPtr_t m_data;
    };

    template <typename T>
    struct MO_EXPORTS TPublisher : TPublisherDataPolicy<T>
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

      private:
        // TContainerConstPtr_t m_data;
        uint32_t m_num_subscribers = 0;
        ParamAllocator::Ptr_t m_allocator;

        TSignal<DataUpdate_s> m_update_signal;
        FrameNumber m_update_counter = 0;
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
    TPublisher<T>::TPublisher()
    {
        this->setFlags(ParamFlags(ParamFlags::kOUTPUT));
        setAllocator(Allocator::getDefault());
    }

    template <typename T>
    bool TPublisher<T>::providesOutput(const TypeInfo type) const
    {
        return type.template isType<T>();
    }

    template <typename T>
    std::vector<TypeInfo> TPublisher<T>::getOutputTypes() const
    {
        return {TypeInfo::create<T>()};
    }

    template <typename T>
    std::vector<Header> TPublisher<T>::getAvailableHeaders() const
    {
        std::vector<Header> out;
        Mutex_t::Lock_t lock(this->mtx());
        auto data = this->getCurrentData();
        if (data)
        {
            out.push_back(data->getHeader());
        }
        return out;
    }

    template <typename T>
    boost::optional<Header> TPublisher<T>::getNewestHeader() const
    {
        auto data = this->getCurrentData();
        if (data)
        {
            return data->getHeader();
        }
        return {};
    }

    template <typename T>
    IDataContainerConstPtr_t TPublisher<T>::getData(const Header* desired, IAsyncStream* stream)
    {
        // TODO stream synchronization
        IDataContainerConstPtr_t out;
        if (desired)
        {
            auto data = TPublisherDataPolicy<T>::getCurrentData();
            if (data)
            {
                if (data->getHeader() == *desired)
                {
                    out = data;
                }
            }
        }
        else
        {
            auto data = TPublisherDataPolicy<T>::getCurrentData();
            out = data;
        }
        return out;
    }

    template <typename T>
    void TPublisher<T>::publish(TContainerConstPtr_t data, IAsyncStream* stream)
    {
        if (stream == nullptr)
        {
            stream = this->getStream();
        }
        if (stream == nullptr)
        {
            stream = &IAsyncStream::currentRef();
        }
        TPublisherDataPolicy<T>::setCurrentData(data);
        MO_ASSERT_LOGGER(getLogger(), stream != nullptr);
        // TODO finish
        m_update_signal(data, *this, ct::value(UpdateFlags::kVALUE_UPDATED), *stream);
        this->emitUpdate(data->header, ct::value(UpdateFlags::kVALUE_UPDATED), *stream);
    }

    template <typename T>
    void TPublisher<T>::publish(const T& data, Header hdr, IAsyncStream* stream)
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
    void TPublisher<T>::publish(T&& data, Header hdr, IAsyncStream* stream)
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
    void TPublisher<T>::publish(U&& data, Ts&&... args)
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

    template <class T>
    auto TPublisherDataPolicy<T>::getCurrentData() const -> TContainerConstPtr_t
    {
        Mutex_t::Lock_t lock(this->mtx());
        return this->m_data;
    }

    template <class T>
    void TPublisherDataPolicy<T>::setCurrentData(TContainerConstPtr_t data)
    {
        Mutex_t::Lock_t lock(this->mtx());
        this->m_data = std::move(data);
    }

    template <typename T>
    template <class... Args>
    typename TPublisher<T>::TContainerPtr_t TPublisher<T>::create(Args&&... args)
    {
        return std::make_shared<TDataContainer<T>>(m_allocator, std::forward<Args>(args)...);
    }

    template <typename T>
    uint32_t TPublisher<T>::getNumSubscribers() const
    {
        Mutex_t::Lock_t lock(this->mtx());
        return m_update_signal.numSlots();
    }

    template <typename T>
    void TPublisher<T>::setAllocator(typename Allocator::Ptr_t alloc)
    {
        Mutex_t::Lock_t lock(this->mtx());
        m_allocator = ParamAllocator::create(std::move(alloc));
    }

    template <typename T>
    std::ostream& TPublisher<T>::print(std::ostream& os) const
    {
        Mutex_t::Lock_t lock(this->mtx());
        os << this->getTreeName();
        return os;
    }

    template <typename T>
    ConnectionPtr_t TPublisher<T>::registerUpdateNotifier(ISlot& f)
    {
        if (f.getSignature() == m_update_signal.getSignature())
        {
            return m_update_signal.connect(f);
        }
        return TParam<IPublisher>::registerUpdateNotifier(f);
    }

    template <typename T>
    ConnectionPtr_t TPublisher<T>::registerUpdateNotifier(const ISignalRelay::Ptr_t& relay_)
    {
        if (relay_->getSignature() == m_update_signal.getSignature())
        {
            auto tmp = relay_;
            return m_update_signal.connect(tmp);
        }
        return TParam<IPublisher>::registerUpdateNotifier(relay_);
    }

} // namespace mo
#endif // MO_PARAMS_TPublisher_HPP
