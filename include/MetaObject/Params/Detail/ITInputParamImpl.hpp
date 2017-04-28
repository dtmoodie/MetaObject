#pragma once
#ifndef __CUDACC__
#include "MetaObject/Logging/Log.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include <functional>

namespace mo
{
    template<class T>
    class ITInputParam;

    template<class T>
    ITInputParam<T>::ITInputParam(const std::string& name, Context* ctx):
            _input(nullptr){
        _update_slot = std::bind(&ITInputParam<T>::onInputUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
        _delete_slot = std::bind(&ITInputParam<T>::onInputDelete, this, std::placeholders::_1);
    }

    template<class T>
    ITInputParam<T>::~ITInputParam(){
        if(_input)
            _input->Unsubscribe();
        if(_shared_input)
            _shared_input->Unsubscribe();
    }

    template<class T>
    bool ITInputParam<T>::setInput(std::shared_ptr<IParam> param){
        mo::Mutex_t::scoped_lock lock(this->mtx());
        if (setInputImpl(param.get())) {
            if (param) {
                this->_shared_input = dynamic_ptr_cast<ITParam<T>>(param);
                if(this->_shared_input){
                    lock.unlock();
                    this->sigUpdate(param->getTimestamp(), param->getContext(), param->getFrameNumber(), param->getCoordinateSystem(), InputSet_e);
                    return true;
                }
            }
            else {
                lock.unlock();
                this->sigUpdate(this->getTimestamp(), this->getContext(), this->getFrameNumber(), this->getCoordinateSystem(), Inputcleared_e);
                return true;
            }
        }
        return false;
    }

    template<class T>
    bool ITInputParam<T>::setInput(IParam* param){
        mo::Mutex_t::scoped_lock lock(this->mtx());
        if(setInputImpl(param)){
            if(param){
                this->_input = dynamic_cast<ITParam<T>*>(param);
                if(this->_input){
                    lock.unlock();
                    this->sigUpdate(param->getTimestamp(), param->getContext(), param->getFrameNumber(), param->getCoordinateSystem(), InputSet_e);
                    return true;
                }
            }else{
                lock.unlock();
                this->sigUpdate(this->getTimestamp(), this->getContext(), this->getFrameNumber(), this->getCoordinateSystem(), Inputcleared_e);
                return true;
            }
        }
        return false;
    }

    template<class T>
    bool ITInputParam<T>::setInputImpl(IParam* param){
        if (param == nullptr){
            if (_shared_input){ _shared_input->Unsubscribe(); }
            else if (_input){ _input->Unsubscribe(); }
            _update_slot.clear();
            _delete_slot.clear();
            _input = nullptr;
            _shared_input.reset();
            return true;
        }
        
        if (param->getTypeInfo() == this->getTypeInfo()){
            if (_input)        _input->Unsubscribe();
            if (_shared_input) _shared_input->Unsubscribe();
            param->Subscribe();
            param->registerUpdateNotifier(&_update_slot);
            param->registerDeleteNotifier(&_delete_slot);
            return true;
        }
        return false;
    }

    template<class T>
    bool ITInputParam<T>::acceptsInput(IParam* param) const{
        return param->getTypeInfo() == getTypeInfo();
    }

    template<class T>
    bool ITInputParam<T>::acceptsType(const TypeInfo& type) const{
        return type == getTypeInfo();
    }

    template<class T>
    IParam* ITInputParam<T>::getInputParam(){
        if(_shared_input)
            return _shared_input.get();
        return input;
    }

    template<class T>
    bool ITInputParam<T>::getData(Storage_t& data, const OptionalTime_t& ts,
        Context* ctx, size_t* fn_){
        if(this->_shared_input){
            return this->_shared_input->getData(data, ts, ctx, fn_);
        }else if(this->_input){
            return this->_input->getData(data, ts, ctx, fn_);
        }
        return false;
    }

    template<class T>
    bool ITInputParam<T>::getData(Storage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr){
        if (this->_shared_input) {
            return this->_shared_input->getData(data, fn, ctx, ts_);
        }
        else if (this->_input) {
            return this->_input->getData(data, fn, ctx, ts_);
        }
        return false;
    }

    template<class T>
    OptionalTime_t ITInputParam<T>::getInputTimestamp(){
        if (input)        return input->getTimestamp();
        if (shared_input) return shared_input->getTimestamp();
        THROW(debug) << "Input not set for " << getTreeName();
        return OptionalTime_t();
    }

    template<class T>
    size_t ITInputParam<T>::getInputFrameNumber(){
        if (input)        return input->getFrameNumber();
        if (shared_input) return shared_input->getFrameNumber();
        THROW(debug) << "Input not set for " << getTreeName();
        return size_t(0);
    }

    template<class T>
    bool ITInputParam<T>::isInputSet() const{
        return input || shared_input;
    }

    // ---- protected functions
    template<class T>
    void ITInputParam<T>::onInputDelete(IParam const* param){
        mo::Mutex_t::scoped_lock lock(this->mtx());
        this->shared_input.reset();
        this->input = nullptr;
        this->emitUpdate( {}, nullptr, {}, nullptr, Inputcleared_e);
    }

    template<class T>
    void ITInputParam<T>::onInputUpdate(ConstStorageRef_t, IParam* param, Context* ctx, OptionalTime_t ts, size_t fn, ICoordinateSystem* cs, UpdateFlags fg){
        this->emitUpdate(ts, ctx, fn, cs, InputUpdated_e);
    }
}
#endif
