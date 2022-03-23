#pragma once
#include "../ITParam.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/AccessToken.hpp"
namespace mo
{
    template <class T>
    class TParam;

    template <typename T>
    TParam<T>::TParam()
        : TParam<T>()
        , _data()
        , IParam()
    {
    }

    template <typename T>
    bool TParam<T>::getData(InputStorage_t& value, const OptionalTime& ts, Context* /*ctx*/, size_t* fn)
    {
        Lock lock(IParam::mtx());
        if (!ts)
        {
            value = _data;
            return true;
        }
        else
        {
            if (ts == IParam::header.timestamp)
            {
                value = _data;
                if (fn)
                {
                    *fn = this->_fn;
                }
                return true;
            }
        }
        return false;
    }

    template <typename T>
    bool TParam<T>::getData(InputStorage_t& value, size_t fn, Context* /*ctx*/, OptionalTime* ts)
    {
        Lock lock(IParam::mtx());
        if (this->_fn == fn)
        {
            if (ts)
                *ts = this->_ts;
            value = _data;
            return true;
        }
        return false;
    }

    template <typename T>
    AccessToken<T> TParam<T>::access()
    {
        return AccessToken<T>(*this, _data);
    }

    template <typename T>
    ConstAccessToken<T> TParam<T>::read() const
    {
        return ConstAccessToken<T>(*this, ParamTraits<T>::get(_data));
    }

    template <typename T>
    bool TParam<T>::updateDataImpl(const Storage_t& data,
                                   const OptionalTime& ts,
                                   Context* ctx,
                                   size_t fn,
                                   const std::shared_ptr<ICoordinateSystem>& cs)
    {
        _data = data;
        this->_fn = fn;
        this->_ts = ts;
        TParamImpl<T>::emitTypedUpdate(_data, this, ctx, ts, this->_fn, cs, UpdateFlags::kVALUE_UPDATED);
        return true;
    }

    template <typename T>
    bool TParam<T>::updateDataImpl(
        Storage_t&& data, const OptionalTime& ts, Context* ctx, size_t fn, const std::shared_ptr<ICoordinateSystem>& cs)
    {
        _data = std::move(data);
        this->_fn = fn;
        this->_ts = ts;
        TParamImpl<T>::emitTypedUpdate(_data, this, ctx, ts, this->_fn, cs, UpdateFlags::kVALUE_UPDATED);
        return true;
    }
} // namespace mo
