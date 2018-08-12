#pragma once
#ifndef __CUDACC__
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/TInputParam.hpp"

#include <boost/thread/recursive_mutex.hpp>
namespace mo
{
    template <typename T>
    class TInputParamPtr;
    class Context;

    template <typename T>
    TInputParamPtr<T>::TInputParamPtr(const std::string& name, Input_t* user_var_, Context* ctx)
        : _user_var(user_var_), ITInputParam<T>(name, ctx), IParam(name, mo::ParamFlags::Input_e)
    {
    }

    template <typename T>
    bool TInputParamPtr<T>::setInput(std::shared_ptr<IParam> param)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if (ITInputParam<T>::setInput(param))
        {
            if (_user_var)
            {
                InputStorage_t data;
                if (this->_input)
                {
                    if (this->_input->getData(data))
                    {
                        _current_data = data;
                        *_user_var = ParamTraits<T>::ptr(_current_data);
                    }
                }
            }
            return true;
        }
        return false;
    }

    template <typename T>
    bool TInputParamPtr<T>::setInput(IParam* param)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if (ITInputParam<T>::setInput(param))
        {
            if (_user_var)
            {
                InputStorage_t data;
                if (ITInputParam<T>::_input && ITInputParam<T>::_input->getData(data))
                {
                    _current_data = data;
                    *_user_var = ParamTraits<T>::ptr(_current_data);
                }
            }
            return true;
        }
        return false;
    }

    template <typename T>
    void TInputParamPtr<T>::setUserDataPtr(Input_t* user_var_)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        _user_var = user_var_;
    }

    template <typename T>
    void TInputParamPtr<T>::onInputUpdate(ConstStorageRef_t data,
                                          IParam* param,
                                          Context* ctx,
                                          OptionalTime_t ts,
                                          size_t fn,
                                          const std::shared_ptr<ICoordinateSystem>& cs,
                                          UpdateFlags fg)
    {
        if (fg == mo::BufferUpdated_e && param->checkFlags(mo::ParamFlags::Buffer_e))
        {
            ITParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::InputUpdated_e);
            IParam::emitUpdate(ts, ctx, fn, cs, fg);
            return;
        }
        else
        {
        }
        if (ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id)
        {
            _current_data = data;
            this->_ts = ts;
            this->_fn = fn;
            if (_user_var)
            {
                *_user_var = ParamTraits<T>::ptr(_current_data);
                ITParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::InputUpdated_e);
                IParam::emitUpdate(ts, ctx, fn, cs, fg);
            }
            else
            {
            }
        }
        else
        {
        }
    }

    template <typename T>
    bool TInputParamPtr<T>::getInput(const OptionalTime_t& ts_, size_t* fn_)
    {
        OptionalTime_t ts = ts_;
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if (_user_var && (ITInputParam<T>::_shared_input || ITInputParam<T>::_input))
        {
            size_t fn;
            std::shared_ptr<ICoordinateSystem> cs;
            if (ITInputParam<T>::_input)
            {
                mo::Mutex_t::scoped_lock input_lock(ITInputParam<T>::_input->mtx());
                if (!ITInputParam<T>::_input->getData(_current_data, ts, this->_ctx, &fn))
                {
                    return false;
                }
                if(!ts)
                {
                    ts = ITInputParam<T>::_input->getTimestamp();
                }
                cs = ITInputParam<T>::_input->getCoordinateSystem();
            }
            *_user_var = ParamTraits<T>::ptr(_current_data);
            if (fn_)
            {
                *fn_ = fn;
            }
            this->_ts = ts;
            this->_fn = fn;
            this->setCoordinateSystem(cs);
            return true;
        }
        return false;
    }

    template <typename T>
    bool TInputParamPtr<T>::getInput(size_t fn, OptionalTime_t* ts_)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if (_user_var && ITInputParam<T>::_input)
        {
            OptionalTime_t ts;
            if (ITInputParam<T>::_input)
            {
                if (!this->_input->getData(_current_data, fn, this->_ctx, &ts))
                {
                    return false;
                }
            }
            auto cs = ITInputParam<T>::_input->getCoordinateSystem();
            *_user_var = ParamTraits<T>::ptr(_current_data);
            if (ts_)
            {
                *ts_ = ts;
            }
            this->_ts = ts;
            this->_fn = fn;
            this->setCoordinateSystem(cs);
            return true;
        }
        return false;
    }

    template <typename T>
    ConstAccessToken<T> TInputParamPtr<T>::read() const
    {
        return ConstAccessToken<T>(*this, ParamTraits<T>::get(_current_data));
    }

    template <typename T>
    class MO_EXPORTS TInputParamPtr<std::shared_ptr<T>> : virtual public ITInputParam<T>,
                                                          virtual public ITConstAccessibleParam<T>
    {
      public:
        typedef typename ParamTraits<T>::Storage_t Storage_t;
        typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
        typedef typename ParamTraits<T>::InputStorage_t InputStorage_t;
        // typedef typename ParamTraits<T>::Input_t Input_t;
        typedef std::shared_ptr<const T> Input_t;

        typedef void(TUpdateSig_t)(ConstStorageRef_t,
                                   IParam*,
                                   Context*,
                                   OptionalTime_t,
                                   size_t,
                                   const std::shared_ptr<ICoordinateSystem>&,
                                   UpdateFlags);
        typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
        typedef TSlot<TUpdateSig_t> TUpdateSlot_t;

        TInputParamPtr(const std::string& name = "", std::shared_ptr<T>* user_var_ = nullptr, Context* ctx = nullptr)
            : _user_var(user_var_), ITInputParam<T>(name, ctx), IParam(name, mo::ParamFlags::Input_e)
        {
            // static_assert(std::is_same<InputStorage_t, std::shared_ptr<Input_t>>::value,
            //              "std::is_same<InputStorage_t, std::shared_ptr<Input_t>>::value");
        }

        bool setInput(std::shared_ptr<IParam> input)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (ITInputParam<T>::setInput(input))
            {
                if (_user_var)
                {
                    InputStorage_t data;
                    if (this->_input && this->_input->getData(data))
                    {
                        _current_data = data;
                        *_user_var = _current_data;
                        return true;
                    }
                }
                return true;
            }
            return false;
        }

        bool setInput(IParam* input)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (ITInputParam<T>::setInput(input))
            {
                if (_user_var)
                {
                    InputStorage_t data;
                    if (ITInputParam<T>::_input && ITInputParam<T>::_input->getData(data))
                    {
                        _current_data = data;
                        *_user_var = _current_data;
                    }
                }
                return true;
            }
            return false;
        }

        void setUserDataPtr(std::shared_ptr<T>* user_var_)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            _user_var = user_var_;
        }

        bool getInput(const OptionalTime_t& ts_, size_t* fn_)
        {
            OptionalTime_t ts = ts_;
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (_user_var && ITInputParam<T>::_input)
            {
                size_t fn;
                std::shared_ptr<ICoordinateSystem> cs;
                if (ITInputParam<T>::_input)
                {
                    mo::Mutex_t::scoped_lock input_param_loakc(ITInputParam<T>::_input->mtx());
                    if (!ITInputParam<T>::_input->getData(_current_data, ts, this->_ctx, &fn))
                    {
                        return false;
                    }else
                    {
                        if(!ts)
                        {
                            this->_ts = ITInputParam<T>::_input->getTimestamp();
                        }
                    }
                    cs = ITInputParam<T>::_input->getCoordinateSystem();
                }
                *_user_var = _current_data;
                if (fn_)
                {
                    *fn_ = fn;
                }
                this->_ts = ts;
                this->_fn = fn;
                this->setCoordinateSystem(cs);
                return true;
            }
            return false;
        }

        bool getInput(size_t fn, OptionalTime_t* ts_)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (_user_var && (ITInputParam<T>::_shared_input || ITInputParam<T>::_input))
            {
                OptionalTime_t ts;
                if (ITInputParam<T>::_input)
                {
                    if (!this->_input->getData(_current_data, fn, this->_ctx, &ts))
                    {
                        return false;
                    }
                }
                auto cs = ITInputParam<T>::_input->getCoordinateSystem();
                *_user_var = _current_data;
                if (ts_)
                {
                    *ts_ = ts;
                }
                this->_ts = ts;
                this->_fn = fn;
                this->setCoordinateSystem(cs);
                return true;
            }
            return false;
        }

        ConstAccessToken<T> read() const override
        {
            return ConstAccessToken<T>(*this, ParamTraits<T>::get(_current_data));
        }

        bool canAccess() const override { return _current_data != nullptr; }

      protected:
        bool updateDataImpl(const Storage_t&,
                            const OptionalTime_t&,
                            Context*,
                            size_t,
                            const std::shared_ptr<ICoordinateSystem>&) override
        {
            return true;
        }

        bool updateDataImpl(
            Storage_t&&, const OptionalTime_t&, Context*, size_t, const std::shared_ptr<ICoordinateSystem>&) override
        {
            return true;
        }

        virtual void onInputUpdate(ConstStorageRef_t data,
                                   IParam* param,
                                   Context* ctx,
                                   OptionalTime_t ts,
                                   size_t fn,
                                   const std::shared_ptr<ICoordinateSystem>& cs,
                                   UpdateFlags fg)
        {
            if (fg == mo::BufferUpdated_e && param->checkFlags(mo::ParamFlags::Buffer_e))
            {
                ITParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::InputUpdated_e);
                IParam::emitUpdate(ts, ctx, fn, cs, fg);
                return;
            }
            if (ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id)
            {
                _current_data = data;
                this->_ts = ts;
                this->_fn = fn;
                if (_user_var)
                {
                    *_user_var = _current_data;
                    ITParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::InputUpdated_e);
                    IParam::emitUpdate(ts, ctx, fn, cs, fg);
                }
            }
        }

        std::shared_ptr<T>* _user_var; // Pointer to the user space pointer variable of type T
        InputStorage_t _current_data;
    };
}
#endif
