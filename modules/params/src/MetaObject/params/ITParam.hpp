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

#include <boost/thread/locks.hpp>
#include <boost/thread/recursive_mutex.hpp>
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

        virtual ConnectionPtr_t registerUpdateNotifier(ISlot* f) override;
        virtual ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr& relay) override;

        virtual void visit(IReadVisitor*) override;
        virtual void visit(IWriteVisitor*) const override;

        bool isValid() const;
        ConstAccessToken<T> read() const;
        AccessToken<T> access();

        IContainerPtr_t getData(const Header& desired = Header());
        IContainerConstPtr_t getData(const Header& desired = Header()) const;

      protected:
        virtual void updateDataImpl(const TContainerPtr_t& data);
        typename TDataContainer<T>::Ptr _data;
        void emitTypedUpdate(TContainerPtr_t data, UpdateFlags flags)
        {
            _typed_update_signal(data, this, flags);
        }

      private:
        TSignal<TUpdate_s> _typed_update_signal;
        static const TypeInfo _type_info;
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
            header.ctx = param->getContext();
            header.coordinate_system = param->getCoordinateSystem();
            header.timestamp = param->getTimestamp();
        }
        const size_t* fnptr = GetKeywordInputOptional<tag::frame_number>(args...);
        if (fnptr)
        {
            header.frame_number = *fnptr;
        }
        else
        {
            if (param == nullptr)
            {
                header.frame_number = header.frame_number + 1;
            }
        }

        if (auto tsptr = GetKeywordInputOptional<tag::timestamp>(args...))
        {
            header.timestamp = *tsptr;
        }
        if (auto ctx_ = GetKeywordInputDefault<tag::context>(nullptr, args...))
        {
            header.ctx = ctx_;
        }
        {
            mo::Lock lock(this->mtx());
            if (getContext() == nullptr)
            {
                setContext(header.ctx);
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
        updateDataImpl(data);
    }

    template <class T>
    void TParam<T>::updateDataImpl(const TContainerPtr_t& data)
    {
        {
            mo::Lock lock(this->mtx());
            _data = data;
        }
        emitUpdate(IDataContainer::Ptr(_data), ValueUpdated_e);
        _typed_update_signal(_data, this, ValueUpdated_e);
    }

    template <class T>
    TypeInfo TParam<T>::getTypeInfo() const
    {
        return _type_info;
    }

    template <class T>
    ConnectionPtr_t TParam<T>::registerUpdateNotifier(ISlot* f)
    {
        if (f->getSignature() == _typed_update_signal.getSignature())
        {
            auto typed = dynamic_cast<TUpdateSlot_t*>(f);
            if (typed)
            {
                return _typed_update_signal.connect(typed);
            }
        }
        return IParam::registerUpdateNotifier(f);
    }

    template <class T>
    ConnectionPtr_t TParam<T>::registerUpdateNotifier(const ISignalRelay::Ptr& relay)
    {
        if (relay->getSignature() == _typed_update_signal.getSignature())
        {
            auto tmp = relay;
            return _typed_update_signal.connect(tmp);
        }
        return IParam::registerUpdateNotifier(relay);
    }

    template <class T>
    void TParam<T>::visit(IReadVisitor* visitor)
    {
        mo::Lock lock(this->mtx());
        if (_data)
        {
            _data->visit(visitor);
        }
    }

    template <class T>
    void TParam<T>::visit(IWriteVisitor* visitor) const
    {
        mo::Lock lock(this->mtx());
        if (_data)
        {
            _data->visit(visitor);
        }
    }

    template <class T>
    typename TParam<T>::IContainerPtr_t TParam<T>::getData(const Header& desired)
    {
        mo::Lock lock(this->mtx());
        if (_data)
        {
            if (!desired.timestamp && !desired.frame_number.valid())
            {
                return _data;
            }
            else
            {
                if (desired.timestamp)
                {
                    if (_data->header.timestamp == desired.timestamp)
                    {
                        return _data;
                    }
                }
                else
                {
                    if (desired.frame_number != std::numeric_limits<uint64_t>::max() &&
                        desired.frame_number == _data->header.frame_number)
                    {
                        return _data;
                    }
                }
            }
        }
        return {};
    }

    template <class T>
    typename TParam<T>::IContainerConstPtr_t TParam<T>::getData(const Header& desired) const
    {
        mo::Lock lock(this->mtx());
        if (_data)
        {
            if (!desired.timestamp && !desired.frame_number.valid())
            {
                return _data;
            }
            else
            {
                if (desired.timestamp && _data->header.timestamp == desired.timestamp)
                {
                    return _data;
                }
                else
                {
                    if (desired.frame_number != std::numeric_limits<uint64_t>::max() &&
                        desired.frame_number == _data->header.frame_number)
                    {
                        return _data;
                    }
                }
            }
        }
        return {};
    }

    template <class T>
    bool TParam<T>::isValid() const
    {
        return _data != nullptr;
    }

    template <class T>
    ConstAccessToken<T> TParam<T>::read() const
    {
        mo::Lock lock(this->mtx());
        MO_ASSERT(_data != nullptr);
        return {std::move(lock), *this, _data->data};
    }

    template <class T>
    AccessToken<T> TParam<T>::access()
    {
        mo::Lock lock(this->mtx());
        MO_ASSERT(_data != nullptr);
        return {std::move(lock), *this, _data->data};
    }

    template <class T>
    const TypeInfo TParam<T>::_type_info = TypeInfo(typeid(T));
}
