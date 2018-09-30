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
#include "IParam.hpp"
#include "TDataContainer.hpp"
#include "traits/TypeTraits.hpp"
#include <boost/thread/locks.hpp>

namespace mo
{
    template <typename T>
    class AccessToken;
    template <typename T>
    struct ConstAccessToken;

    template <typename T>
    class MO_EXPORTS ITParam<T> : virtual public IParam
    {
      public:
        using Container_t = TDataContainer<T>;
        using ContainerPtr_t = std::shared_ptr<Container_t>;
        using TUpdate_s = void(ContainerPtr_t, IParam*, UpdateFlags);
        using TUpdateSignal_t = TSignal<TUpdate_s>;
        using TUpdateSlot_t = TSlot<TUpdate_s>;

        // brief ITParam default constructor, passes args to IParam
        ITParam(const std::string& name = "", ParamFlags flags = ParamFlags::Control_e);

        template <class... Args>
        void updateData(const T& data, const Args&... args);
        template <class... Args>
        void updateData(T&& data, const Args&... args);

        virtual void updateData(const T& data, const Header& header);
        virtual void updateData(T&& data, Header&& header);

        virtual bool getData(ContainerPtr_t& data, const Header& desired = Header(), Header* received = nullptr);

        TypeInfo getTypeInfo() const override;

        virtual ConnectionPtr_t registerUpdateNotifier(UpdateSlot_t* f) override;
        virtual ConnectionPtr_t registerUpdateNotifier(TSignalRelay<Update_s>::Ptr& relay) override;
        virtual ConnectionPtr_t registerUpdateNotifier(ISlot* f) override;
        virtual ConnectionPtr_t registerUpdateNotifier(ISignalRelay::Ptr relay) override;

        ConnectionPtr_t registerUpdateNotifier(TUpdateSlot_t* f);
        ConnectionPtr_t registerUpdateNotifier(typename TSignalRelay<TUpdate_s>::Ptr& relay);

        virtual void visit(IReadVisitor*) override;
        virtual void visit(IWriteVisitor*) const override;

        bool canAccess() const;
        ConstAccessToken<T> read() const;
        AccessToken<T> access();

      private:
        TSignal<TUpdate_s> _typed_update_signal;
        static const TypeInfo _type_info;
        std::shared_ptr<TDataContainer<T>> _data;
    };

    /// implementations
    template <class T>
    ITParam<T>::ITParam(const std::string& name, ParamFlags flags) : IParam(name, flags)
    {
    }

    template <class T>
    template <class... Args>
    void ITParam<T>::updateData(const T& data, const Args&... args)
    {
        T tmp = data;
        updateData(std::move(tmp), std::forward<Args>(args)...);
    }

    template <class T>
    template <class... Args>
    void ITParam<T>::updateData(T&& data, const Args&... args)
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
            if (_header.ctx == nullptr)
            {
                _header.ctx = header.ctx;
            }
        }

        header.coordinate_system =
            GetKeywordInputDefault<tag::coordinate_system>(std::shared_ptr<ICoordinateSystem>(), args...);

        updateData(std::move(data), std::move(header));
        return this;
    }

    template <class T>
    void ITParam<T>::updateData(const T& data, const Header& header)
    {
        {
            mo::Lock lock(this->mtx());
            _data = std::make_shared<TDataContainer<T>>();
            _data->data = data;
            _data->header = header;
        }
        emitUpdate(_data->header);
        _typed_update_signal(_data, this, ValueUpdated_e);
    }

    template <class T>
    void ITParam<T>::updateData(T&& data, Header&& header)
    {
        {
            mo::Lock lock(this->mtx());
            _data = std::make_shared<TDataContainer<T>>();
            _data->data = std::move(data);
            _data->header = std::move(header);
        }
        emitUpdate(_data->header);
        _typed_update_signal(_data, this, ValueUpdated_e);
    }

    template <class T>
    TypeInfo ITParam<T>::getTypeInfo() const
    {
        return _type_info;
    }

    template <class T>
    ConnectionPtr_t ITParam<T>::registerUpdateNotifier(UpdateSlot_t* f)
    {
        if (f->getSignature() == _typed_update_signal.getSignature())
        {
            return _typed_update_signal.connect(f);
        }
        return IParam::registerUpdateNotifier(f);
    }

    template <class T>
    ConnectionPtr_t ITParam<T>::registerUpdateNotifier(TSignalRelay<Update_s>::Ptr& relay)
    {
        if (relay->getSignature() == _typed_update_signal.getSignature())
        {
            return _typed_update_signal.connect(relay);
        }
        return IParam::registerUpdateNotifier(relay);
    }

    template <class T>
    ConnectionPtr_t ITParam<T>::registerUpdateNotifier(ISlot* f)
    {
        if (f->getSignature() == _typed_update_signal.getSignature())
        {
            auto typed = dynamic_cast<TUpdateSlot_t*>(f);
            if (typed)
            {
                return registerUpdateNotifier(typed);
            }
        }
        return IParam::registerUpdateNotifier(f);
    }

    template <class T>
    ConnectionPtr_t ITParam<T>::registerUpdateNotifier(ISignalRelay::Ptr relay)
    {
        if (relay->getSignature() == _typed_update_signal.getSignature())
        {
            auto typed = std::dynamic_pointer_cast<TSignalRelay<TUpdate_s>>(relay);
            return registerUpdateNotifier(typed);
        }
        return IParam::registerUpdateNotifier(relay);
    }

    template <class T>
    ConnectionPtr_t ITParam<T>::registerUpdateNotifier(TUpdateSlot_t* f)
    {
        return _typed_update_signal.connect(f);
    }

    template <class T>
    ConnectionPtr_t ITParam<T>::registerUpdateNotifier(typename TSignalRelay<TUpdate_s>::Ptr& relay)
    {
        return _typed_update_signal.connect(relay);
    }

    template <class T>
    void ITParam<T>::visit(IReadVisitor* visitor)
    {
        mo::Lock lock(this->mtx());
        if (_data)
        {
            _data->visit(visitor);
        }
    }

    template <class T>
    void ITParam<T>::visit(IWriteVisitor* visitor) const
    {
        mo::Lock lock(this->mtx());
        if (_data)
        {
            _data->visit(visitor);
        }
    }

    template <class T>
    bool ITParam<T>::getData(ContainerPtr_t& data, const Header& desired, Header* received)
    {

        mo::Lock lock(this->mtx());
        if (_data)
        {
            if (!desired.timestamp && desired.frame_number == std::numeric_limits<uint64_t>::max())
            {
                data = _data;
                if (received)
                {
                    *received = _header;
                }
                return true;
            }
            else
            {
                if (desired.timestamp && _data->header.timestamp == desired.timestamp)
                {
                    data = _data;
                    if (received)
                    {
                        *received = _header;
                    }
                    return true;
                }
                else
                {
                    if (desired.frame_number != std::numeric_limits<uint64_t>::max() &&
                        desired.frame_number == _data->header.frame_number)
                    {
                        data = _data;
                        if (received)
                        {
                            *received = _header;
                        }
                        return true;
                    }
                }
            }
        }
        return false;
    }

    template <class T>
    bool ITParam<T>::canAccess() const
    {
        return _data != nullptr;
    }

    template <class T>
    ConstAccessToken<T> ITParam<T>::read() const
    {
        mo::Lock lock(this->mtx());
        MO_ASSERT(_data != nullptr);
        return {std::move(lock), *this, _data->data};
    }

    template <class T>
    AccessToken<T> ITParam<T>::access()
    {
        mo::Lock lock(this->mtx());
        MO_ASSERT(_data != nullptr);
        return {std::move(lock), *this, _data->data};
    }
}
