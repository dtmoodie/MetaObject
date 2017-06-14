#pragma once
#ifndef __CUDACC__
#include "MetaObject/params/MetaParam.hpp"
#include <boost/thread/recursive_mutex.hpp>
namespace mo {
template <typename T>
class TInputParamPtr;
class Context;

template <typename T>
TInputParamPtr<T>::TInputParamPtr(const std::string& name, Input_t* user_var_, Context* ctx)
    : _user_var(user_var_)
    , ITInputParam<T>(name, ctx)
    , IParam(name, mo::Input_e) {
}

template <typename T>
bool TInputParamPtr<T>::setInput(std::shared_ptr<IParam> param) {
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    if (ITInputParam<T>::setInput(param)) {
        if (_user_var) {
            InputStorage_t data;
            if (this->_input)
                if (this->_input->getData(data)) {
                    _current_data = data;
                    *_user_var    = ParamTraits<T>::ptr(_current_data);
                    return true;
                }
            if (this->_shared_input)
                if (this->_shared_input->getData(data)) {
                    _current_data = data;
                    *_user_var    = ParamTraits<T>::ptr(_current_data);
                    return true;
                }
        }
        return true;
    }
    return false;
}

template <typename T>
bool TInputParamPtr<T>::setInput(IParam* param) {
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    if (ITInputParam<T>::setInput(param)) {
        if (_user_var) {
            InputStorage_t data;
            if (ITInputParam<T>::_input)
                if (ITInputParam<T>::_input->getData(data)) {
                    _current_data = data;
                    *_user_var    = ParamTraits<T>::ptr(_current_data);
                }
            if (ITInputParam<T>::_shared_input)
                if (ITInputParam<T>::_shared_input->getData(data)) {
                    _current_data = data;
                    *_user_var    = ParamTraits<T>::ptr(_current_data);
                }
        }
        return true;
    }
    return false;
}

template <typename T>
void TInputParamPtr<T>::setUserDataPtr(Input_t* user_var_) {
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    _user_var = user_var_;
}

template <typename T>
void TInputParamPtr<T>::onInputUpdate(ConstStorageRef_t data, IParam* param, Context* ctx, OptionalTime_t ts, size_t fn, ICoordinateSystem* cs, UpdateFlags fg) {
    if (fg == mo::BufferUpdated_e && param->checkFlags(mo::Buffer_e)) {
        ITParam<T>::_typed_update_signal(data, this, ctx, ts, fn, cs, mo::InputUpdated_e);
        IParam::emitUpdate(ts, ctx, fn, cs, fg);
        return;
    }
    if (ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) {
        _current_data = data;
        this->_ts     = ts;
        this->_fn     = fn;
        if (_user_var) {
            *_user_var = ParamTraits<T>::ptr(_current_data);
        }
    }
}

template <typename T>
bool TInputParamPtr<T>::getInput(const OptionalTime_t& ts, size_t* fn_) {
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    if (_user_var && (ITInputParam<T>::_shared_input || ITInputParam<T>::_input)) {
        size_t             fn;
        ICoordinateSystem* cs = nullptr;
        if (ITInputParam<T>::_shared_input) {
            if (!ITInputParam<T>::_shared_input->getData(_current_data, ts, this->_ctx, &fn)) {
                return false;
            }
            cs = ITInputParam<T>::_shared_input->getCoordinateSystem();
        }
        if (ITInputParam<T>::_input) {
            if (!ITInputParam<T>::_input->getData(_current_data, ts, this->_ctx, &fn)) {
                return false;
            }
            cs = ITInputParam<T>::_input->getCoordinateSystem();
        }
        *_user_var = ParamTraits<T>::ptr(_current_data);
        if (fn_)
            *fn_  = fn;
        this->_ts = ts;
        this->_fn = fn;
        this->setCoordinateSystem(cs);
        return true;
    }
    return false;
}

template <typename T>
bool TInputParamPtr<T>::getInput(size_t fn, OptionalTime_t* ts_) {
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    if (_user_var && (ITInputParam<T>::_shared_input || ITInputParam<T>::_input)) {
        OptionalTime_t     ts;
        ICoordinateSystem* cs = nullptr;
        if (ITInputParam<T>::_shared_input) {
            if (!ITInputParam<T>::_shared_input->getData(_current_data, fn, this->_ctx, &ts)) {
                return false;
            }
            cs = ITInputParam<T>::_shared_input->getCoordinateSystem();
        }
        if (ITInputParam<T>::_input) {
            if (!this->_input->getData(_current_data, fn, this->_ctx, &ts)) {
                return false;
            }
            cs = ITInputParam<T>::_input->getCoordinateSystem();
        }
        *_user_var = ParamTraits<T>::ptr(_current_data);
        if (ts_)
            *ts_  = ts;
        this->_ts = ts;
        this->_fn = fn;
        this->setCoordinateSystem(cs);
        return true;
    }
    return false;
}
}
#endif
