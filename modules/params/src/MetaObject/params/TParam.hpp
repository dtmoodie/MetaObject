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
#include "ParamTags.hpp"
#include "TDataContainer.hpp"

#include <MetaObject/thread/fiber_include.hpp>

#include <boost/thread/locks.hpp>
namespace mo
{
    template <typename T>
    struct AccessToken;
    template <typename T>
    struct ConstAccessToken;

    template <typename T>
    class MO_EXPORTS TParam : virtual public IParam
    {
      public:
        using Container_t = TDataContainer<T>;
        using TContainerPtr_t = typename TDataContainer<T>::Ptr_t;
        using type = typename TDataContainer<T>::type;
        using ContainerConstPtr_t = std::shared_ptr<const Container_t>;
        using TUpdate_s = void(TContainerPtr_t, IParam*, UpdateFlags, IAsyncStream&);
        using TUpdateSignal_t = TSignal<TUpdate_s>;
        using TUpdateSlot_t = TSlot<TUpdate_s>;

        // brief TParam default constructor, passes args to IParam
        TParam(const std::string& name = "", ParamFlags flags = ParamFlags::kCONTROL);

        template <class... Args>
        auto updateData(const type& data, const Args&... args) -> ct::EnableIf<hasNamedParam<Args...>()>;

        template <class... Args>
        auto updateData(type&& data, const Args&... args) -> ct::EnableIf<hasNamedParam<Args...>()>;

        void updateData(const type& data, const Header& header = Header(), IAsyncStream* stream = nullptr);
        void updateData(type&& data, Header&& header = Header(), IAsyncStream* stream = nullptr);
        void updateData(const TContainerPtr_t& data, IAsyncStream* stream = nullptr);

        TypeInfo getTypeInfo() const override;

        ConnectionPtr_t registerUpdateNotifier(ISlot& f) override;
        ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr_t& relay) override;

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

        template <class... ARGS>
        typename TDataContainer<T>::Ptr_t create(ARGS&&... args) const;

        const type& value() const
        {
            MO_ASSERT(m_data);
            return m_data->data;
        }

        OptionalTime getTimestamp() const override;
        FrameNumber getFrameNumber() const override;

        void setAllocator(typename Allocator::Ptr_t alloc)
        {
            m_allocator = ParamAllocator::create(alloc);
        }

      protected:
        typename TDataContainer<T>::Ptr_t getDataImpl(const Header& desired = Header());
        typename TDataContainer<T>::ConstPtr_t getDataImpl(const Header& desired = Header()) const;

        virtual void updateDataImpl(const TContainerPtr_t& data, mo::UpdateFlags fg, IAsyncStream& stream);

        void emitTypedUpdate(TContainerPtr_t data, UpdateFlags flags, IAsyncStream& stream)
        {
            m_typed_update_signal(data, this, flags, stream);
        }

        ParamAllocator::Ptr_t allocator() const
        {
            return m_allocator;
        }

      private:
        FrameNumber m_update_count;
        typename TDataContainer<T>::Ptr_t m_data;
        TSignal<TUpdate_s> m_typed_update_signal;
        static const TypeInfo m_type_info;
        ParamAllocator::Ptr_t m_allocator;
    };

    ///////////////////////////////////////////////////////////////////////////////////////
    /// implementation
    ///////////////////////////////////////////////////////////////////////////////////////

    template <class T>
    TParam<T>::TParam(const std::string& name, ParamFlags flags)
        : IParam(name, flags)
        , m_allocator(ParamAllocator::create())
    {
    }

    template <class T>
    template <class... Args>
    auto TParam<T>::updateData(const type& data, const Args&... args) -> ct::EnableIf<hasNamedParam<Args...>()>
    {
        auto tmp = data;
        updateData(std::move(tmp), args...);
    }

    template <class T>
    template <class... Args>
    auto TParam<T>::updateData(type&& data, const Args&... args) -> ct::EnableIf<hasNamedParam<Args...>()>
    {
        Header header;
        const IParam* param = getKeywordInputDefault<params::Param>(static_cast<const IParam*>(nullptr), args...);
        if (param)
        {
            header.frame_number = param->getFrameNumber();
            header.timestamp = param->getTimestamp();
        }
        const uint64_t* fnptr = getKeywordInputOptional<params::FrameNumber>(args...);
        if (fnptr)
        {
            header.frame_number = *fnptr;
        }

        auto tsptr = getKeywordInputOptional<params::Timestamp>(args...);
        if (tsptr)
        {
            header.timestamp = *tsptr;
        }

        auto stream = getKeywordInputDefault<params::Stream>(nullptr, args...);
        if (!stream)
        {
            stream = getStream();
        }
        if (!stream)
        {
            stream = IAsyncStream::current().get();
        }

        updateData(std::move(data), std::move(header));
    }

    template <class T>
    void TParam<T>::updateData(const type& data, const Header& header, IAsyncStream* stream)
    {
        auto container = create(data);
        container->header = header;
        updateData(container, stream);
    }

    template <class T>
    void TParam<T>::updateData(type&& data, Header&& header, IAsyncStream* stream)
    {
        auto container = create(std::move(data));
        container->header = std::move(header);
        updateData(container, stream);
    }

    template <class T>
    void TParam<T>::updateData(const TContainerPtr_t& data, IAsyncStream* stream)
    {
        if (!data)
        {
            return;
        }

        if (!stream)
        {
            stream = this->getStream();
        }
        if (!stream)
        {
            stream = IAsyncStream::current().get();
        }
        MO_ASSERT(stream != nullptr);

        updateDataImpl(data, UpdateFlags::kVALUE_UPDATED, *stream);
    }

    template <class T>
    template <class... ARGS>
    typename TDataContainer<T>::Ptr_t TParam<T>::create(ARGS&&... args) const
    {
        return std::make_shared<TDataContainer<T>>(m_allocator, std::forward<ARGS>(args)...);
    }

    template <class T>
    void TParam<T>::updateDataImpl(const TContainerPtr_t& data, mo::UpdateFlags fg, IAsyncStream& stream)
    {
        if (!data->header.frame_number.valid())
        {
            ++m_update_count.val;
            data->header.frame_number = m_update_count;
        }
        else
        {
            m_update_count = data->header.frame_number;
        }

        {
            mo::Lock_t lock(this->mtx());
            m_data = data;
        }
        emitUpdate(IDataContainer::Ptr_t(data), fg, stream);
        m_typed_update_signal(data, this, fg, stream);
    }

    template <class T>
    TypeInfo TParam<T>::getTypeInfo() const
    {
        return m_type_info;
    }

    template <class T>
    ConnectionPtr_t TParam<T>::registerUpdateNotifier(ISlot& f)
    {
        if (f.getSignature() == m_typed_update_signal.getSignature())
        {
            // Is the additional check necessary since we do the above check?
            auto typed = dynamic_cast<TUpdateSlot_t*>(&f);
            if (typed)
            {
                return m_typed_update_signal.connect(*typed);
            }
        }
        return IParam::registerUpdateNotifier(f);
    }

    template <class T>
    ConnectionPtr_t TParam<T>::registerUpdateNotifier(const ISignalRelay::Ptr_t& relay)
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
    typename TDataContainer<T>::Ptr_t TParam<T>::getDataImpl(const Header& desired)
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

            if (desired.frame_number.valid() && desired.frame_number == m_data->header.frame_number)
            {
                return m_data;
            }
        }
        return {};
    }

    template <class T>
    typename TDataContainer<T>::ConstPtr_t TParam<T>::getDataImpl(const Header& desired) const
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
    OptionalTime TParam<T>::getTimestamp() const
    {
        if (m_data)
        {
            return m_data->getHeader().timestamp;
        }
        return {};
    }

    template <class T>
    FrameNumber TParam<T>::getFrameNumber() const
    {
        if (m_data)
        {
            return m_data->getHeader().frame_number;
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
        return ConstAccessToken<T>(std::move(lock), *this, m_data->data);
    }

    template <class T>
    AccessToken<T> TParam<T>::access()
    {
        mo::Lock_t lock(this->mtx());
        MO_ASSERT(m_data != nullptr);
        return AccessToken<T>(std::move(lock), *this, m_data->data);
    }

    template <class T>
    const TypeInfo TParam<T>::m_type_info = TypeInfo(typeid(TParam<T>::type));
} // namespace mo
