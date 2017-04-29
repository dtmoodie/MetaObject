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
    bool TParamPtr<T>::getData(Storage_t& value, const OptionalTime_t& ts, Context* ctx, size_t* fn_) {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(!ts){
            if(ptr){
                value = *ptr;
                return true;
            }
        }else{
            if(this->_ts && *(this->_ts) == *ts && ptr){
                value = *ptr;
                return true;
            }
        }
        return false;
    }

    template<typename T>
    bool TParamPtr<T>::getData(Storage_t& value, size_t fn, Context* ctx, OptionalTime_t* ts){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(fn == this->_fn && ptr){
            value = *ptr;
            if(ts)
                *ts = this->_ts;
            return true;
        }
        return false;
    }

    template<typename T>
    bool TParamPtr<T>::updateDataImpl(ConstStorageRef_t data, OptionalTime_t ts, Context* ctx, size_t fn, ICoordinateSystem* cs){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ptr){
            *ptr = data;
            lock.unlock();
            this->emitUpdate(ts, ctx, fn, cs);
            return true;
        }
        return false;
    }

    
    template<typename T>
    ITParam<T>* TParamPtr<T>::updatePtr(T* ptr, bool ownsData_){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        this->ptr = ptr;
        this->ownsData = ownsData_;
        return this;
    }

    template<typename T>
    MetaParam<T, 100, void> TParamPtr<T>::_meta_Param;
}
#endif
