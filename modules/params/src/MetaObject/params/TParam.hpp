#pragma once
/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/MetaObject
*/
#include "AccessToken.hpp"
#include "IParam.hpp"
#include "TDataContainer.hpp"

#include <MetaObject/thread/fiber_include.hpp>

#include <boost/thread/locks.hpp>
namespace mo
{
    template <typename T>
    class AccessToken;
    template <typename T>
    struct ConstAccessToken;

    template <typename T>
    class MO_EXPORTS TParam<T> : virtual public IParam
    {
      public:
        using Container_t = TDataContainer<T>;
        using TContainerPtr_t = std::shared_ptr<Container_t>;
        using ContainerConstPtr_t = std::shared_ptr<const Container_t>;
        using TUpdate_s = void(TContainerPtr_t, IParam*, UpdateFlags);
        using TUpdateSignal_t = TSignal<TUpdate_s>;
        using TUpdateSlot_t = TSlot<TUpdate_s>;

        // brief TParam default constructor, passes args to IParam
        TParam(const std::string& name = "", ParamFlags flags = ParamFlags::Control_e);

        template <class... Args>
        void updateData(const T& data, const Args&... args);
        template <class... Args>
        void updateData(T&& data, const Args&... args);

        void updateData(const T& data, const Header& header = Header());
        void updateData(T&& data, Header&& header = Header());
        void updateData(const TContainerPtr_t& data);

        TypeInfo getTypeInfo() const override;

        ConnectionPtr_t registerUpdateNotifier(ISlot* f) override;
        ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr& relay) override;

        void load(ILoadVisitor&) override;
        void save(ISaveVisitor&) const override;
        void load(BinaryInputVisitor& ar) override;
        void save(BinaryOutputVisitor& ar) const override;
        void visit(StaticVisitor&) const override;

        virtual bool isValid() const;
        virtual ConstAccessToken<T> read() const;
        virtual AccessToken<T> access();

        IContainerPtr_t getData(const Header& desired = Header()) override;
        IContainerConstPtr_t getData(const Header& desired = Header()) const override;

        template <class U = T>
        typename TDataContainer<T>::Ptr getTypedData(const Header& desired = Header())
        {
            return getDataImpl(desired);
        }

        template <class U = T>
        typename TDataContainer<T>::ConstPtr getTypedData(const Header& desired = Header()) const
        {
            return getDataImpl(desired);
        }

        bool getTypedData(T* data_, const Header& desired = Header()) const
        {
            auto data = getDataImpl(desired);
            if (data)
            {
                *data_ = data->data;
                return true;
            }
            return false;
        }

        T value() const
        {
            MO_ASSERT(m_data);
            return m_data->data;
        }

      protected:
        typename TDataContainer<T>::Ptr getDataImpl(const Header& desired = Header());
        typename TDataContainer<T>::ConstPtr getDataImpl(const Header& desired = Header()) const;

        void updateDataImpl(const TContainerPtr_t& data);

        void emitTypedUpdate(TContainerPtr_t data, UpdateFlags flags)
        {
            m_typed_update_signal(data, this, flags);
        }

      private:
        typename TDataContainer<T>::Ptr m_data;
        TSignal<TUpdate_s> m_typed_update_signal;
        static const TypeInfo m_type_info;
    };

    ///////////////////////////////////////////////////////////////////////////////////////
    /// implementation
    ///////////////////////////////////////////////////////////////////////////////////////

    template <class T>
    TParam<T>::TParam(const std::string& name, ParamFlags flags)
        : IParam(name, flags)
    {
    }

    template <class T>
    template <class... Args>
    void TParam<T>::updateData(const T& data, const Args&... args)
    {
        T tmp = data;
        updateData(std::move(tmp), args...);
    }

    template <class T>
    template <class... Args>
    void TParam<T>::updateData(T&& data, const Args&... args)
    {
        Header header;
        const IParam* param = GetKeywordInputOptional<tag::param>(args...);
        if (param)
        {
            header.frame_number = param->getFrameNumber();
            header.stream = param->getStream();
            header.coordinate_system = param->getCoordinateSystem();
            header.timestamp = param->getTimestamp();
        }
        const uint64_t* fnptr = GetKeywordInputOptional<tag::frame_number>(args...);
        if (fnptr)
        {
            header.frame_number = *fnptr;
        }

        if (auto tsptr = GetKeywordInputOptional<tag::timestamp>(args...))
        {
            header.timestamp = *tsptr;
        }

        if (auto stream_ = GetKeywordInputDefault<tag::stream>(nullptr, args...))
        {
            header.stream = stream_;
        }
        {
            mo::Lock_t lock(this->mtx());
            if (getStream() == nullptr)
            {
                setStream(header.stream);
            }
        }

        header.coordinate_system =
            GetKeywordInputDefault<tag::coordinate_system>(std::shared_ptr<ICoordinateSystem>(), args...);

        updateData(std::move(data), std::move(header));
    }

    template <class T>
    void TParam<T>::updateData(const T& data, const Header& header)
    {
        auto container = std::make_shared<TDataContainer<T>>();
        container->data = data;
        container->header = header;
        updateData(container);
    }

    template <class T>
    void TParam<T>::updateData(T&& data, Header&& header)
    {
        auto container = std::make_shared<TDataContainer<T>>();
        container->data = std::move(data);
        container->header = std::move(header);
        updateData(container);
    }

    template <class T>
    void TParam<T>::updateData(const TContainerPtr_t& data)
    {
        if (!data)
        {
            return;
        }

        updateDataImpl(data);
    }

    template <class T>
    void TParam<T>::updateDataImpl(const TContainerPtr_t& data)
    {
        if (!data->header.frame_number.valid())
        {
            ++m_header.frame_number.val;
            data->header.frame_number = m_header.frame_number.val;
        }
        else
        {
            m_header.frame_number.val = data->header.frame_number;
        }

        {
            mo::Lock_t lock(this->mtx());
            m_data = data;
        }
        emitUpdate(IDataContainer::Ptr(data), ValueUpdated_e);
        m_typed_update_signal(data, this, ValueUpdated_e);
    }

    template <class T>
    TypeInfo TParam<T>::getTypeInfo() const
    {
        return m_type_info;
    }

    template <class T>
    ConnectionPtr_t TParam<T>::registerUpdateNotifier(ISlot* f)
    {
        if (f->getSignature() == m_typed_update_signal.getSignature())
        {
            auto typed = dynamic_cast<TUpdateSlot_t*>(f);
            if (typed)
            {
                return m_typed_update_signal.connect(typed);
            }
        }
        return IParam::registerUpdateNotifier(f);
    }

    template <class T>
    ConnectionPtr_t TParam<T>::registerUpdateNotifier(const ISignalRelay::Ptr& relay)
    {
        if (relay->getSignature() == m_typed_update_signal.getSignature())
        {
            auto tmp = relay;
            return m_typed_update_signal.connect(tmp);
        }
        return IParam::registerUpdateNotifier(relay);
    }

    template <class T>
    void TParam<T>::load(ILoadVisitor& visitor)
    {
        mo::Lock_t lock(this->mtx());
        if (m_data)
        {
            m_data->load(visitor);
        }
    }

    template <class T>
    void TParam<T>::save(ISaveVisitor& visitor) const
    {
        mo::Lock_t lock(this->mtx());
        if (m_data)
        {
            m_data->save(visitor);
        }
    }

    template <class T>
    void TParam<T>::load(BinaryInputVisitor& ar)
    {
        mo::Lock_t lock(this->mtx());
        if (m_data)
        {
            m_data->load(ar);
        }
    }

    template <class T>
    void TParam<T>::save(BinaryOutputVisitor& ar) const
    {
        mo::Lock_t lock(this->mtx());
        if (m_data)
        {
            m_data->save(ar);
        }
    }

    template <class T>
    void TParam<T>::visit(StaticVisitor& visitor) const
    {
        TDataContainer<T>::visitStatic(visitor);
    }

    template <class T>
    typename TParam<T>::IContainerPtr_t TParam<T>::getData(const Header& desired)
    {
        return getDataImpl(desired);
    }

    template <class T>
    typename TParam<T>::IContainerConstPtr_t TParam<T>::getData(const Header& desired) const
    {
        return getDataImpl(desired);
    }

    template <class T>
    typename TDataContainer<T>::Ptr TParam<T>::getDataImpl(const Header& desired)
    {
        mo::Lock_t lock(this->mtx());
        if (m_data)
        {
            if (!desired.timestamp && !desired.frame_number.valid())
            {
                return m_data;
            }

            if (desired.timestamp && m_data->header.timestamp == desired.timestamp)
            {
                return m_data;
            }

            if (desired.frame_number != std::numeric_limits<uint64_t>::max() &&
                desired.frame_number == m_data->header.frame_number)
            {
                return m_data;
            }
        }
        return {};
    }

    template <class T>
    typename TDataContainer<T>::ConstPtr TParam<T>::getDataImpl(const Header& desired) const
    {
        mo::Lock_t lock(this->mtx());
        if (m_data)
        {
            if (!desired.timestamp && !desired.frame_number.valid())
            {
                return m_data;
            }

            if (desired.timestamp && m_data->header.timestamp == desired.timestamp)
            {
                return m_data;
            }

            if (desired.frame_number != std::numeric_limits<uint64_t>::max() &&
                desired.frame_number == m_data->header.frame_number)
            {
                return m_data;
            }
        }
        return {};
    }

    template <class T>
    bool TParam<T>::isValid() const
    {
        return m_data != nullptr;
    }

    template <class T>
    ConstAccessToken<T> TParam<T>::read() const
    {
        mo::Lock_t lock(this->mtx());
        MO_ASSERT(m_data != nullptr);
        return {std::move(lock), *this, m_data->data};
    }

    template <class T>
    AccessToken<T> TParam<T>::access()
    {
        mo::Lock_t lock(this->mtx());
        MO_ASSERT(m_data != nullptr);
        return {std::move(lock), *this, m_data->data};
    }

    template <class T>
    const TypeInfo TParam<T>::m_type_info = TypeInfo(typeid(T));
}
