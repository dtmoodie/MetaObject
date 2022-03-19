#ifndef MO_PARAMS_TSUBSCRIBER_HPP
#define MO_PARAMS_TSUBSCRIBER_HPP
#include "IPublisher.hpp"
#include "ISubscriber.hpp"
#include "TParam.hpp"
#include <MetaObject/params/buffers/IBuffer.hpp>

#include <ct/reflect/print.hpp>

#include <ostream>
#include <vector>

namespace mo
{
    template <class T>
    struct TSubscriberImpl : TParam<ISubscriber>
    {
        using type = typename ContainerTraits<T>::type;
        using Super_t = TParam<ISubscriber>;
        TSubscriberImpl();
        ~TSubscriberImpl() override = default;

        bool setInput(std::shared_ptr<IPublisher> input) override;
        bool setInput(IPublisher* input = nullptr) override;

        bool acceptsPublisher(const IPublisher& param) const override;
        bool acceptsType(const TypeInfo& type) const override;
        std::vector<TypeInfo> getInputTypes() const override;

        void load(ILoadVisitor&) override;
        void save(ISaveVisitor&) const override;
        void visit(StaticVisitor&) const override;

        // IDataContainerConstPtr_t getInputData(IAsyncStream* = nullptr) const override;
        IDataContainerConstPtr_t getCurrentData(IAsyncStream* = nullptr) const override;
        bool getCurrentData(T& out, IAsyncStream* = nullptr) const;

        IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* stream = nullptr) override;
        bool getData(T& out, const Header* desired = nullptr, IAsyncStream* stream = nullptr);

        bool hasNewData() const override;

        ConnectionPtr_t registerUpdateNotifier(ISlot& f) override;
        ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr_t& relay) override;

        IPublisher* getPublisher() const override;
        bool isInputSet() const override;
        std::vector<Header> getAvailableHeaders() const override;
        boost::optional<Header> getNewestHeader() const override;

        std::ostream& print(std::ostream&) const override;
        TDataContainerConstPtr_t<T> getTypedData(const Header* desired = nullptr, IAsyncStream* stream = nullptr);

      protected:
        void onInputUpdate(const IDataContainerConstPtr_t& data, const IParam& param, UpdateFlags, IAsyncStream*);
        void onInputDelete(const IParam&);

        virtual void onData(TDataContainerConstPtr_t<T>, const IParam&, UpdateFlags, IAsyncStream* stream);

        TSlot<DataUpdate_s> m_update_slot;
        TSlot<Delete_s> m_delete_slot;

        TSignal<DataUpdate_s> m_update_signal;

        std::shared_ptr<IPublisher> m_shared_publisher;
        IPublisher* m_publisher = nullptr;

        ConnectionPtr_t m_input_connection;
        ConnectionPtr_t m_delete_connection;

        TDataContainerConstPtr_t<T> m_data;
        bool m_new_data_available = false;
    };

    template <class T>
    struct TSubscriber : TSubscriberImpl<T>
    {
    };

    //////////////////////////////////////////////////////////
    // Implementation
    //////////////////////////////////////////////////////////

    template <class T>
    TSubscriberImpl<T>::TSubscriberImpl()
    {
        this->setFlags(ParamFlags::kINPUT);
        m_update_slot.bind(&TSubscriberImpl<T>::onInputUpdate, this);
    }

    template <class T>
    bool TSubscriberImpl<T>::setInput(std::shared_ptr<IPublisher> publisher)
    {
        if (setInput(publisher.get()))
        {
            m_shared_publisher = std::move(publisher);
            return true;
        }
        return false;
    }

    template <class T>
    bool TSubscriberImpl<T>::setInput(IPublisher* publisher)
    {
        Lock_t lock(this->mtx());
        const auto my_type = TypeInfo::create<T>();
        if (publisher->providesOutput(my_type))
        {
            if (m_publisher != nullptr)
            {
                m_update_slot.clear();
                m_delete_slot.clear();
                emitUpdate(Header(), UpdateFlags::kINPUT_CLEARED, IAsyncStream::current().get());
            }
            m_publisher = publisher;
            m_input_connection = publisher->registerUpdateNotifier(m_update_slot);
            m_delete_connection = publisher->registerDeleteNotifier(m_delete_slot);
            if (m_input_connection == nullptr || m_delete_connection == nullptr)
            {
                m_publisher = nullptr;
                return false;
            }
            emitUpdate(Header(), UpdateFlags::kINPUT_SET, IAsyncStream::current().get());
            mo::IAsyncStream* stream = this->getStream();
            IDataContainerConstPtr_t data = publisher->getData(nullptr, stream);
            /*if (data && stream)
            {
                this->onInputUpdate(data, *publisher, mo::UpdateFlags::kVALUE_UPDATED, *stream);
            }*/
            return true;
        }
        const auto output_types = publisher->getOutputTypes();
        this->getLogger().info(
            "Unable to connect publisher '{}' of type <{}> to subscriber '{}' of type <{}> due to mismatching types",
            publisher->getTreeName(),
            output_types,
            this->getTreeName(),
            my_type);

        return false;
    }

    template <class T>
    bool TSubscriberImpl<T>::acceptsPublisher(const IPublisher& param) const
    {
        return param.providesOutput(TypeInfo::create<T>());
    }

    template <class T>
    bool TSubscriberImpl<T>::acceptsType(const TypeInfo& type) const
    {
        return type == TypeInfo::create<T>();
    }

    template <class T>
    std::vector<TypeInfo> TSubscriberImpl<T>::getInputTypes() const
    {
        return {TypeInfo::create<T>()};
    }

    template <class T>
    void TSubscriberImpl<T>::load(ILoadVisitor& visitor)
    {
        Super_t::load(visitor);
    }

    template <class T>
    void TSubscriberImpl<T>::save(ISaveVisitor& visitor) const
    {
        Super_t::save(visitor);
    }

    template <class T>
    void TSubscriberImpl<T>::visit(StaticVisitor& ar) const
    {
        Super_t::visit(ar);
    }

    template <class T>
    IDataContainerConstPtr_t TSubscriberImpl<T>::getCurrentData(IAsyncStream* stream) const
    {
        // TODO use stream
        Mutex_t::Lock_t lock(this->mtx());
        return m_data;
    }

    template <class T>
    bool TSubscriberImpl<T>::getCurrentData(T& out, IAsyncStream* stream) const
    {
        auto container = getCurrentData(stream);
        if (container)
        {
            auto typed = static_cast<const TDataContainer<T>*>(container.get());
            if (typed)
            {
                out = typed->data;
                return true;
            }
        }
        return false;
    }

    template <class T>
    IDataContainerConstPtr_t TSubscriberImpl<T>::getData(const Header* desired, IAsyncStream* stream)
    {
        Mutex_t::Lock_t lock(this->mtx());
        if (m_publisher)
        {
            auto data = m_publisher->getData(desired, stream);
            // TODO evaluate if this is correct
            m_new_data_available = false;
            m_data = std::static_pointer_cast<const TDataContainer<T>>(data);
            return data;
        }
        return {};
    }

    template <class T>
    TDataContainerConstPtr_t<T> TSubscriberImpl<T>::getTypedData(const Header* desired, IAsyncStream* stream)
    {
        auto data = getData(desired, stream);
        if (data)
        {
            return std::static_pointer_cast<const TDataContainer<T>>(data);
        }
        return {};
    }

    template <class T>
    bool TSubscriberImpl<T>::getData(T& out, const Header* desired, IAsyncStream* stream)
    {
        auto typed = getTypedData(desired, stream);
        if (typed)
        {
            out = typed->data;
            return true;
        }
        return false;
    }

    template <class T>
    bool TSubscriberImpl<T>::hasNewData() const
    {
        return m_new_data_available;
    }

    template <class T>
    ConnectionPtr_t TSubscriberImpl<T>::registerUpdateNotifier(ISlot& f)
    {
        auto connection = TParam<ISubscriber>::registerUpdateNotifier(f);
        if (!connection && m_update_signal.getSignature() == f.getSignature())
        {
            connection = m_update_signal.connect(f);
        }
        return connection;
    }

    template <class T>
    ConnectionPtr_t TSubscriberImpl<T>::registerUpdateNotifier(const ISignalRelay::Ptr_t& relay)
    {
        auto connection = TParam<ISubscriber>::registerUpdateNotifier(relay);
        if (!connection && m_update_signal.getSignature() == relay->getSignature())
        {
            auto tmp = relay;
            connection = m_update_signal.connect(tmp);
        }
        return connection;
    }

    template <class T>
    void TSubscriberImpl<T>::onInputUpdate(const IDataContainerConstPtr_t& data,
                                           const IParam& param,
                                           UpdateFlags fgs,
                                           IAsyncStream* stream)
    {
        if (data == nullptr)
        {
            if (m_publisher)
            {
                this->getLogger().warn(
                    "{} published nullptr data to {}", m_publisher->getTreeName(), this->getTreeName());
            }
            else
            {
                this->getLogger().error("onInputUpdate called without having a input publisher set... wut?");
            }

            return;
        }
        if (data->getType().isType<T>())
        {
            auto typed = std::static_pointer_cast<const TDataContainer<T>>(data);
            MO_ASSERT(typed);
            onData(std::move(typed), param, fgs, stream);
        }
    }

    template <class T>
    IPublisher* TSubscriberImpl<T>::getPublisher() const
    {
        Mutex_t::Lock_t lock(this->mtx());
        return m_publisher;
    }

    template <class T>
    bool TSubscriberImpl<T>::isInputSet() const
    {
        Mutex_t::Lock_t lock(this->mtx());
        return m_publisher != nullptr;
    }

    template <class T>
    std::vector<Header> TSubscriberImpl<T>::getAvailableHeaders() const
    {
        std::vector<Header> output;
        Mutex_t::Lock_t lock(this->mtx());
        if (m_data)
        {
            output.push_back(m_data->getHeader());
        }
        return output;
    }

    template <class T>
    boost::optional<Header> TSubscriberImpl<T>::getNewestHeader() const
    {
        Mutex_t::Lock_t lock(this->mtx());
        if (m_data)
        {
            return m_data->getHeader();
        }
        if (m_publisher)
        {
            return m_publisher->getNewestHeader();
        }
        return {};
    }

    template <class T>
    void TSubscriberImpl<T>::onInputDelete(const IParam& param)
    {
        if (m_publisher == &param)
        {
            m_publisher = nullptr;
            m_shared_publisher.reset();
        }
    }

    template <class T>
    void TSubscriberImpl<T>::onData(TDataContainerConstPtr_t<T> data,
                                    const IParam& update_source,
                                    UpdateFlags fg,
                                    IAsyncStream* stream)
    {
        if(data == nullptr)
        {
            return;
        }
        Lock_t lock(this->mtx());
        const auto header = data->getHeader();
        if (fg & ct::value(UpdateFlags::kBUFFER_UPDATED) && update_source.checkFlags(mo::ParamFlags::kBUFFER))
        {
            TParam<ISubscriber>::emitUpdate(header, fg, stream);
            return;
        }

        auto dst_stream = this->getStream();
        if (dst_stream && dst_stream != stream)
        {
            m_data = data;
            if(stream)
            {
                m_data->record(*stream);
            }
            m_data->sync(*dst_stream);
            m_new_data_available = true;
            emitUpdate(header, ct::value(UpdateFlags::kINPUT_UPDATED), dst_stream);
            m_update_signal(data, *this, ct::value(UpdateFlags::kINPUT_UPDATED), dst_stream);
        }
        else
        {
            m_data = data;
            m_new_data_available = true;
            emitUpdate(header, ct::value(UpdateFlags::kINPUT_UPDATED), dst_stream);
            m_update_signal(data, *this, ct::value(UpdateFlags::kINPUT_UPDATED), stream);
        }
    }
    template <class T>
    std::ostream& TSubscriberImpl<T>::print(std::ostream& os) const
    {
        os << this->getTreeName();
        os << ' ';
        os << TypeInfo::create<T>();
        Mutex_t::Lock_t lock(this->mtx());
        if (m_data)
        {
            os << *m_data;
        }
        return os;
    }

} // namespace mo

#endif // MO_PARAMS_TSUBSCRIBER_HPP
