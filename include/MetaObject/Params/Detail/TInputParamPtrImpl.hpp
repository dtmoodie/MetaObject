#pragma once
#ifndef __CUDACC__
#include <boost/thread/recursive_mutex.hpp>
#include "MetaObject/Params/MetaParam.hpp"
namespace mo
{
    template<typename T> class TInputParamPtr;
    class Context;

    template<typename T> TInputParamPtr<T>::TInputParamPtr(const std::string& name, Input_t* user_var_, Context* ctx) :
        _user_var(user_var_),
            ITInputParam<T>(name, ctx),
            IParam(name, mo::Input_e)
    {
    }

    template<typename T>
    bool TInputParamPtr<T>::setInput(std::shared_ptr<IParam> param){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ITInputParam<T>::SetInput(param)){
            if(_user_var){
                Storage_t data;
                if(this->_input)
                     if(this->_input->getData(data)){
                         _current_data = data;
                         *_user_var = &(*_current_data);
                         return true;
                     }
                if(this->_shared_input)
                    if(this->_shared_input->getData(data)){
                        _current_data = data;
                        *_user_var = &(*_current_data);
                        return true;
                    }
            }
            return true;
        }
        return false;
    }

    template<typename T>
    bool TInputParamPtr<T>::setInput(IParam* param){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ITInputParam<T>::SetInput(param)){
            if(_user_var){
                Storage_t data;
                if(this->_input)
                    if(_input->getData(data)){
                        _current_data = data;
                        *_user_var = &(*_current_data);
                    }
                if(this->_shared_input)
                    if(_shared_input->getData(data)){
                        _current_data = data;
                        *_user_var = &(*_current_data);
                    }
            }
            return true;
        }
        return false;
    }

    template<typename T>
    void TInputParamPtr<T>::setUserDataPtr(Input_t* user_var_){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        _user_var = user_var_;
    }

    template<typename T>
    void TInputParamPtr<T>::onInputUpdate(ConstStorageRef_t data, IParam* param, Context* ctx, OptionalTime_t ts, size_t fn, ICoordinateSystem* cs, UpdateFlags fg){
        if(ctx == this->_ctx){
            _current_data = data;
            this->_ts = ts;
            this->_fn = fn;
            if(_user_var){
                *_user_var = &(*_current_data);
            }
        }
    }

    template<typename T>
    bool TInputParamPtr<T>::getInput(OptionalTime_t ts, size_t* fn_){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(_user_var){
            size_t fn;
            Storage_t data;
            if(this->_shared_input){
                if(!this->_shared_input->getData(data, ts, this->_ctx, &fn)){
                    return false;
                }
            }
            if(this->_input)
            {
                if (!this->_input->getData(data, ts, this->_ctx, &fn)) {
                    return false;
                }
            }
            _current_data = data;
            *_user_var = &(*_current_data);
            if (fn_)
                *fn_ = fn;
            return true;
        }
        return false;
    }

    template<typename T>
    bool TInputParamPtr<T>::getInput(size_t fn, OptionalTime_t* ts_){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        OptionalTime_t ts;
        if(_user_var){
            if(this->_shared_input){
                Storage_t data;
                if(this->_shared_input->getData(data, fn, this->_ctx, &ts)){
                    _current_data = data;
                    
                    *_user_var = &(*_current_data);
                    if (ts_)
                        *ts_ = ts;
                    this->_ts = ts;
                    this->_fn = fn;
                    return true;
                }
            }
            if(this->_input){
                Storage_t data;
                if(this->_input->getData(data, fn, this->_ctx, &ts)){
                    _current_data = data;
                    *_user_var = &(*_current_data);
                    if (ts_)
                        *ts_ = ts;
                    this->_ts = ts;
                    this->_fn = fn;
                    return true;
                }
            }
        }
        return false;
    }

}
#endif
