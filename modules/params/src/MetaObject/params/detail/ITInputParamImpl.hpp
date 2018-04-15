#pragma once
#ifndef __CUDACC__
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/ITInputParam.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include <functional>
#include <memory>

namespace mo
{
    template <class T>
    ITInputParam<T>::ITInputParam(const std::string& name, Context* ctx) : _input(nullptr)
    {
        _update_slot = std::bind(&ITInputParam<T>::onInputUpdate,
                                 this,
                                 std::placeholders::_1,
                                 std::placeholders::_2,
                                 std::placeholders::_3,
                                 std::placeholders::_4,
                                 std::placeholders::_5,
                                 std::placeholders::_6,
                                 std::placeholders::_7);
        _delete_slot = std::bind(&ITInputParam<T>::onInputDelete, this, std::placeholders::_1);
    }

    template <class T>
    ITInputParam<T>::~ITInputParam()
    {
        if (_input)
            _input->unsubscribe();
        if (_shared_input)
            _shared_input->unsubscribe();
    }

    template <class T>
    bool ITInputParam<T>::setInput(std::shared_ptr<IParam> param)
    {
        if (setInput(param.get()))
        {
            _shared_input = param;
            return true;
        }
        return false;
    }

    template <class T>
    bool ITInputParam<T>::setInput(IParam* param)
    {
        mo::Mutex_t::scoped_lock lock(this->mtx());
        if (param == nullptr)
        {
            if (_input)
            {
                _input->unsubscribe();
            }
            _update_slot.clear();
            _delete_slot.clear();
            _input = nullptr;
            _shared_input.reset();
            _shared_input.reset();
            lock.unlock();
            this->emitUpdate(this->getTimestamp(),
                             this->getContext(),
                             this->getFrameNumber(),
                             this->getCoordinateSystem(),
                             InputCleared_e);
            return true;
        }
        if (param->getTypeInfo() == this->getTypeInfo())
        {
            if (_input)
                _input->unsubscribe();
            this->_input = dynamic_cast<ITParam<T>*>(param);
            if (this->_input)
            {
                param->subscribe();
                param->registerUpdateNotifier(&_update_slot);
                param->registerDeleteNotifier(&_delete_slot);
                lock.unlock();
                this->emitUpdate(param->getTimestamp(),
                                 param->getContext(),
                                 param->getFrameNumber(),
                                 param->getCoordinateSystem(),
                                 InputSet_e);
                return true;
            }
        }
        return false;
    }

    template <class T>
    bool ITInputParam<T>::acceptsInput(IParam* param) const
    {
        return param->getTypeInfo() == getTypeInfo();
    }

    template <class T>
    bool ITInputParam<T>::acceptsType(const TypeInfo& type) const
    {
        return type == getTypeInfo();
    }

    template <class T>
    IParam* ITInputParam<T>::getInputParam() const
    {
        if (_shared_input)
            return _shared_input.get();
        return _input;
    }

    template <class T>
    bool ITInputParam<T>::getData(InputStorage_t& data, const OptionalTime_t& ts, Context* ctx, size_t* fn_)
    {
        if (this->_input)
        {
            return this->_input->getData(data, ts, ctx, fn_);
        }
        return false;
    }

    template <class T>
    bool ITInputParam<T>::getData(InputStorage_t& data, size_t fn, Context* ctx, OptionalTime_t* ts_)
    {
        if (this->_input)
        {
            return this->_input->getData(data, fn, ctx, ts_);
        }
        return false;
    }

    template <class T>
    OptionalTime_t ITInputParam<T>::getInputTimestamp()
    {
        if (_input)
            return _input->getTimestamp();
        if (_shared_input)
            return _shared_input->getTimestamp();
        THROW(debug) << "Input not set for " << getTreeName();
        return OptionalTime_t();
    }

    template <class T>
    size_t ITInputParam<T>::getInputFrameNumber()
    {
        if (_input)
            return _input->getFrameNumber();
        if (_shared_input)
            return _shared_input->getFrameNumber();
        THROW(debug) << "Input not set for " << getTreeName();
        return size_t(0);
    }

    template <class T>
    bool ITInputParam<T>::isInputSet() const
    {
        return _input || _shared_input;
    }

    // ---- protected functions
    template <class T>
    void ITInputParam<T>::onInputDelete(IParam const* param)
    {
        if ((this->_input && this->_input != param) || (this->_shared_input && this->_shared_input.get() != param))
            return;
        mo::Mutex_t::scoped_lock lock(this->mtx());
        this->_shared_input.reset();
        this->_input = nullptr;
        this->emitUpdate({}, nullptr, {}, nullptr, InputCleared_e);
    }

    template <class T>
    void ITInputParam<T>::onInputUpdate(ConstStorageRef_t,
                                        IParam* param,
                                        Context* ctx,
                                        OptionalTime_t ts,
                                        size_t fn,
                                        const std::shared_ptr<ICoordinateSystem>& cs,
                                        UpdateFlags fg)
    {
        this->emitUpdate(ts, ctx, fn, cs, InputUpdated_e);
    }
}
#endif
