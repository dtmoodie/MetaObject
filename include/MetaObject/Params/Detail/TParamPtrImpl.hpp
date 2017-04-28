#pragma once
#ifndef __CUDACC__
#include "MetaObject/Logging/Log.hpp"
#include <boost/thread/recursive_mutex.hpp>
namespace mo
{
    template<typename T> class TParamPtr;
    template<typename T, int N, typename Enable> struct MetaParam;


    template<typename T>
    TParamPtr<T>::TParamPtr(const std::string& name,
                                            T* ptr_,
                                            ParamFlags type,
                                            bool ownsData_) :
        ptr(ptr_), ownsData(ownsData_), ITParam<T>(name, type)
    {
        (void)&_meta_Param;
    }

    template<typename T>
    TParamPtr<T>::~TParamPtr()
    {
        if (ownsData && ptr)
            delete ptr;
    }

    template<typename T>
    T* TParamPtr<T>::GetDataPtr(OptionalTime_t ts, Context* ctx, size_t* fn_)
    {
        if(ts)
        {
            if (ts == this->_ts)
            {
                if (fn_)
                {
                    *fn_ = IParam::_fn;
                }
                return ptr;
            }
            else
            {
                if (this->_ts)
                {
                    LOG(trace) << "Requested timestamp != current [" << ts
                               << " != " << this->_fn
                               << "] for Param " << this->getTreeName();
                    return nullptr;
                }
                else
                {
                    if (this->checkFlags(mo::Unstamped_e))
                    {
                        if (fn_)
                        {
                            *fn_ = IParam::_fn;
                        }
                        return ptr;
                    }
                    else
                    {
                        return nullptr;
                    }
                }
            }
        }
        if(fn_)
        {
            *fn_ = IParam::_fn;
        }
        return ptr;
    }

    template<typename T>
    T* TParamPtr<T>::GetDataPtr(size_t fn, Context* ctx, OptionalTime_t* ts_)
    {
        if(fn != this->_fn)
        {
            LOG(trace) << "Requested frame != current [" << fn
                       << " != " << this->_fn
                       << "] for Param " << this->getTreeName();
            return nullptr;
        }
        if(ts_)
        {
            *ts_ = IParam::_ts;
        }
        return ptr;
    }

    template<typename T>
    T TParamPtr<T>::GetData(OptionalTime_t ts, Context* ctx, size_t* fn)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ts)
        {
            if(ts != IParam::_ts && this->_ts)
            {
                THROW(debug) << "Requested timestamp != current ["
                             << ts << " != " << IParam::_ts
                             << "] for Param " << this->getTreeName();
            }
        }
        if(ptr == nullptr)
            THROW(debug) << "Data pointer not set";
        if(fn)
            *fn = this->_fn;
        return *ptr;
    }

    template<typename T>
    T TParamPtr<T>::GetData(size_t fn, Context* ctx, OptionalTime_t* ts)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(fn != std::numeric_limits<size_t>::max())
        {
            if(fn != IParam::_fn && IParam::_fn != std::numeric_limits<size_t>::max())
            {
                THROW(debug) << "Requested frame != current ["
                             << fn << " != " << IParam::_fn
                             << "] for Param " << this->getTreeName();
            }
        }
        if(ptr == nullptr)
            THROW(debug) << "Data pointer not set";
        if(ts)
            *ts = IParam::_ts;
        return *ptr;
    }

    template<typename T>
    bool TParamPtr<T>::GetData(T& value, OptionalTime_t ts, Context* ctx, size_t* fn)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ts)
        {
            if(ts != this->_ts)
            {
                LOG(trace) << "Requested timestamp != current ["
                           << ts << " != " << IParam::_ts
                           << "] for Param " << this->getTreeName();
                return false;
            }
        }

        if (ptr)
        {
            value = *ptr;
            if(fn)
                *fn = this->_fn;
            return true;
        }
        return false;
    }

    template<typename T>
    bool TParamPtr<T>::GetData(T& value, size_t fn, Context* ctx, OptionalTime_t* ts)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(fn != std::numeric_limits<size_t>::max())
        {
            if(fn != this->_fn)
            {
                LOG(trace) << "Requested frame != current ["
                           << fn << " != " << this->_fn
                           << "] for Param " << this->getTreeName();
                return false;
            }
        }

        if (ptr)
        {
            value = *ptr;
            if(ts)
                *ts = this->_ts;
            return true;
        }
        return false;
    }

    /*template<typename T>
    ITParam<T>* TParamPtr<T>::UpdateData(const T& data_, OptionalTime_t ts,
                                                         Context* ctx, size_t fn, ICoordinateSystem* cs)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ptr)
        {
            *ptr = data_;
            this->commit(ts, ctx, fn, cs);
        }
        return this;
    }*/
    template<typename T>
    bool TParamPtr<T>::UpdateDataImpl(const T& data, OptionalTime_t ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ptr)
        {
            *ptr = data;
            lock.unlock();
            this->commit(ts, ctx, fn, cs);
            return true;
        }
        return false;
    }

    template<typename T>
    bool TParamPtr<T>::Update(IParam* other)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        auto typed = dynamic_cast<ITParam<T>*>(other);
        if (typed)
        {
            *ptr = typed->GetData();
            this->commit(other->getTimestamp(),
                         other->getContext(),
                         other->getFrameNumber(),
                         other->GetCoordinateSystem());
            return true;
        }
        return false;
    }

    template<typename T>
    std::shared_ptr<IParam> TParamPtr<T>::DeepCopy() const
    {
        return std::shared_ptr<IParam>(new TParamPtr<T>(IParam::getName(), ptr));
    }

    template<typename T>
    ITParam<T>* TParamPtr<T>::UpdatePtr(T* ptr, bool ownsData_)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        this->ptr = ptr;
        this->ownsData = ownsData_;
        return this;
    }

    template<typename T>
    MetaParam<T, 100, void> TParamPtr<T>::_meta_Param;
}
#endif
