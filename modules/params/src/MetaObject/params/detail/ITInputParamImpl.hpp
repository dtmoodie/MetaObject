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
    bool ITInputParam<T>::acceptsInput(IParam* param) const
    {
        if (param->checkFlags(mo::ParamFlags::Output_e))
        {
            auto out_param = dynamic_cast<OutputParam*>(param);
            return out_param->providesOutput(getTypeInfo());
        }
        else
        {
            return param->getTypeInfo() == getTypeInfo();
        }
    }

    template <class T>
    bool ITInputParam<T>::acceptsType(const TypeInfo& type) const
    {
        return type == getTypeInfo();
    }

    template <class T>
    IParam* ITInputParam<T>::getInputParam() const
    {
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
        else
        {
            return false;
        }
    }

    template <class T>
    OptionalTime_t ITInputParam<T>::getInputTimestamp()
    {
        if (_input)
        {
            return _input->getTimestamp();
        }
        else
        {
            THROW(debug) << "Input not set for " << getTreeName();
            return OptionalTime_t();
        }
    }

    template <class T>
    size_t ITInputParam<T>::getInputFrameNumber()
    {
        if (_input)
        {
            return _input->getFrameNumber();
        }
        else
        {
            THROW(debug) << "Input not set for " << getTreeName();
            return size_t(0);
        }
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
        {
            return;
        }
        else
        {
            mo::Mutex_t::scoped_lock lock(this->mtx());
            this->_shared_input.reset();
            this->_input = nullptr;
            IParam::emitUpdate({}, nullptr, {}, nullptr, InputCleared_e);
        }
    }

    template <class T>
    void ITInputParam<T>::onInputUpdate(ConstStorageRef_t,
                                        IParam* /*param*/,
                                        Context* ctx,
                                        OptionalTime_t ts,
                                        size_t fn,
                                        const std::shared_ptr<ICoordinateSystem>& cs,
                                        UpdateFlags /*fg*/)
    {
        IParam::emitUpdate(ts, ctx, fn, cs, InputUpdated_e);
    }
}
#endif
