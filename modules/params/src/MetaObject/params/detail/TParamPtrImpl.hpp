#pragma once
#ifndef __CUDACC__
#include "MetaObject/logging/logging.hpp"
#include <MetaObject/params/AccessToken.hpp>
#include <boost/thread/recursive_mutex.hpp>
namespace mo {
template <typename T>
class TParamPtr;
template <typename T, int N, typename Enable>
struct MetaParam;

template <typename T>
TParamPtr<T>::TParamPtr(const std::string& name,
    T*                                     ptr_,
    ParamFlags                             type,
    bool                                   ownsData_)
    : ptr(ptr_)
    , ownsData(ownsData_)
    , ITParam<T>(name, type) 
{
    (void)&_meta_Param;
}

template <typename T>
TParamPtr<T>::~TParamPtr() 
{
    if (ownsData && ptr)
        delete ptr;
}

template <typename T>
bool TParamPtr<T>::getData(InputStorage_t& value, const OptionalTime_t& ts, Context* ctx, size_t* fn_) 
{
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    if (!ts) 
    {
        if (ptr) 
        {
            ParamTraits<T>::reset(value, *ptr);
            //value = *ptr;
            return true;
        }
    } else 
    {
        if (this->_ts && *(this->_ts) == *ts && ptr) 
        {
            ParamTraits<T>::reset(value, *ptr);
            return true;
        }
    }
    return false;
}

template <typename T>
bool TParamPtr<T>::getData(InputStorage_t& value, size_t fn, Context* ctx, OptionalTime_t* ts) 
{
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    if (fn == this->_fn && ptr) 
    {
        ParamTraits<T>::reset(value, *ptr);
        if (ts)
        {
            *ts = this->_ts;
        }
        return true;
    }
    return false;
}

template <typename T>
IParam* TParamPtr<T>::emitUpdate(const OptionalTime_t& ts_,
    Context*                                           ctx_,
    const boost::optional<size_t>&                     fn_,
    const std::shared_ptr<ICoordinateSystem>&          cs_,
    UpdateFlags                                        flags_) 
{
    IParam::emitUpdate(ts_, ctx_, fn_, cs_, flags_);
    if (ptr)
    {
        ITParam<T>::_typed_update_signal(ParamTraits<T>::copy(*ptr), this, ctx_, ts_, this->_fn, cs_, flags_);
    }
    return this;
}

template <typename T>
AccessToken<T> TParamPtr<T>::access() 
{
    MO_ASSERT(ptr);
    return AccessToken<T>(*this, *ptr);
}

template <typename T>
ConstAccessToken<T> TParamPtr<T>::access() const
{
    MO_ASSERT(ptr);
    return ConstAccessToken<T>(*this, ParamTraits<T>::get(*ptr));
}

template <typename T>
bool TParamPtr<T>::updateDataImpl(const Storage_t& data, const OptionalTime_t& ts, Context* ctx, size_t fn, const std::shared_ptr<ICoordinateSystem>& cs) 
{
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    if (ptr) 
    {
        *ptr = ParamTraits<T>::get(data);
        lock.unlock();
        this->emitUpdate(ts, ctx, fn, cs);
        ITParam<T>::_typed_update_signal(data, this, ctx, ts, fn, cs, mo::ValueUpdated_e);
        return true;
    }
    return false;
}

template <typename T>
ITParam<T>* TParamPtr<T>::updatePtr(Raw_t* ptr, bool ownsData_) 
{
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    this->ptr      = ptr;
    this->ownsData = ownsData_;
    return this;
}

template <typename T>
MetaParam<T, 100, void> TParamPtr<T>::_meta_Param;

template <typename T>
bool TParamOutput<T>::getData(InputStorage_t& data, const OptionalTime_t& ts, Context* ctx, size_t* fn_) 
{
    if (!ts || ts == this->_ts) 
    {
        data = this->data;
        return true;
    }
    return false;
}

template <typename T>
bool TParamOutput<T>::getData(InputStorage_t& data, size_t fn, Context* ctx, OptionalTime_t* ts_) 
{
    if (fn == this->_fn) 
    {
        data = this->data;
        return true;
    }
    return false;
}

template <typename T>
AccessToken<T> TParamOutput<T>::access() 
{
    return AccessToken<T>(*this, data);
}

template <typename T>
ConstAccessToken<T> TParamOutput<T>::access() const
{
    return ConstAccessToken<T>(*this, ParamTraits<T>::get(data));
}

template <typename T>
bool TParamOutput<T>::updateDataImpl(const Storage_t& data, const OptionalTime_t& ts, Context* ctx, size_t fn, const std::shared_ptr<ICoordinateSystem>& cs) 
{
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    //this->data = data;
    if (this->ptr) {
        *(this->ptr) = ParamTraits<T>::get(data);
    }
    lock.unlock();
    this->emitUpdate(ts, ctx, fn, cs);
    return true;
}

template <typename T>
IParam* TParamOutput<T>::emitUpdate(const OptionalTime_t& ts_,
    Context*                                              ctx_,
    const boost::optional<size_t>&                        fn_,
    const std::shared_ptr<ICoordinateSystem>&             cs_,
    UpdateFlags                                           flags_) 
{
    if (this->ptr) 
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if (this->checkFlags(mo::ParamFlags::Unstamped_e)) 
        {
            data = ParamTraits<T>::copy(*(this->ptr));
        } else 
        {
            ParamTraits<T>::move(data, std::move(*(this->ptr)));
        }
        lock.unlock();
        IParam::emitUpdate(ts_, ctx_, fn_, cs_, flags_);
        ITParam<T>::_typed_update_signal(data, this, ctx_, ts_, this->_fn, cs_, flags_);
    }

    return this;
}
}
#endif
