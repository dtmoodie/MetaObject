#pragma once
#include "../TParam.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/AccessToken.hpp"
namespace mo
{
    template <class T>
    class TParam;

    template <typename T>
    TParam<T>::TParam() : ITParam<T>(), _data(), IParam()
    {
    }

    template <typename T>
    bool TParam<T>::getData(InputStorage_t& value, const OptionalTime_t& ts, Context* /*ctx*/, size_t* fn)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
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
    bool TParam<T>::getData(InputStorage_t& value, size_t fn, Context* /*ctx*/, OptionalTime_t* ts)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
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
                                   const OptionalTime_t& ts,
                                   Context* ctx,
                                   size_t fn,
                                   const std::shared_ptr<ICoordinateSystem>& cs)
    {
        _data = data;
        this->_fn = fn;
        this->_ts = ts;
        ITParamImpl<T>::emitTypedUpdate(_data, this, ctx, ts, this->_fn, cs, ValueUpdated_e);
        return true;
    }

    template <typename T>
    bool TParam<T>::updateDataImpl(Storage_t&& data,
                                   const OptionalTime_t& ts,
                                   Context* ctx,
                                   size_t fn,
                                   const std::shared_ptr<ICoordinateSystem>& cs)
    {
        _data = std::move(data);
        this->_fn = fn;
        this->_ts = ts;
        ITParamImpl<T>::emitTypedUpdate(_data, this, ctx, ts, this->_fn, cs, ValueUpdated_e);
        return true;
    }
}
