#pragma once
#ifndef __CUDACC__
#include <boost/thread/recursive_mutex.hpp>
#include "MetaObject/Params/MetaParam.hpp"
namespace mo
{
    template<typename T> class TInputParamPtr;
    class Context;

    template<typename T> TInputParamPtr<T>::TInputParamPtr(const std::string& name, const T** userVar_, Context* ctx) :
            userVar(userVar_),
            ITInputParam<T>(name, ctx),
            IParam(name, mo::Input_e)
    {
    }

    template<typename T>
    bool TInputParamPtr<T>::SetInput(std::shared_ptr<IParam> param)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ITInputParam<T>::SetInput(param))
        {
            if(userVar)
            {
                if(this->input)
                    *userVar = this->input->GetDataPtr();
                if(this->shared_input)
                    *userVar = this->shared_input->GetDataPtr();
            }
            return true;
        }
        return false;
    }

    template<typename T>
    bool TInputParamPtr<T>::SetInput(IParam* param)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ITInputParam<T>::SetInput(param))
        {
            if(userVar)
            {
                if(this->input)
                    *userVar = this->input->GetDataPtr();
                if(this->shared_input)
                    *userVar = this->shared_input->GetDataPtr();
            }
            return true;
        }
        return false;
    }

    template<typename T>
    void TInputParamPtr<T>::SetUserDataPtr(const T** user_var_)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        userVar = user_var_;
    }

    template<typename T>
    void TInputParamPtr<T>::onInputUpdate(Context* ctx, IParam* param)
    {
        if(this->input)
        {
            mo::Mutex_t::scoped_lock lock(this->input->mtx());
            //this->commit(this->input->getTimestamp(), ctx, this->input->getFrameNumber(), this->input->GetCoordinateSystem());
            this->_update_signal(ctx, this);
            if((ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) || (ctx == nullptr &&  this->_ctx == nullptr))
            {
                if(userVar)
                    *userVar = this->input->GetDataPtr();
            }
        }else if(this->shared_input)
        {
            mo::Mutex_t::scoped_lock lock(this->shared_input->mtx());
            //this->commit(this->shared_input->getTimestamp(), ctx, this->shared_input->getFrameNumber(), this->shared_input->GetCoordinateSystem());
            this->_update_signal(ctx, this);
            if((ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) || ((ctx == nullptr) &&  (this->_ctx == nullptr)))
            {
                if(userVar)
                    *userVar = this->shared_input->GetDataPtr();
            }
        }
    }

    template<typename T>
    bool TInputParamPtr<T>::getInput(OptionalTime_t ts, size_t* fn_)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(userVar)
        {
            if(this->shared_input)
            {
                size_t fn;
                *userVar = this->shared_input->GetDataPtr(ts, this->_ctx, &fn);
                if(*userVar != nullptr)
                {
                    this->_ts = ts;
                    if(fn_)
                        *fn_ = fn;
                    this->_fn = fn;
                    return true;
                }
            }
            if(this->input)
            {
                size_t fn;
                *userVar = this->input->GetDataPtr(ts, this->_ctx, &fn);
                if(*userVar != nullptr)
                {
                    this->_ts = ts;
                    if(fn_)
                        *fn_ = fn;
                    this->_fn = fn;
                    return true;
                }
            }
        }
        return false;
    }

    template<typename T>
    bool TInputParamPtr<T>::getInput(size_t fn, OptionalTime_t* ts_)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        OptionalTime_t ts;
        if(userVar)
        {
            if(this->shared_input)
            {
                *userVar = this->shared_input->GetDataPtr(fn, this->_ctx, &ts);
                if(*userVar != nullptr)
                {
                    if(ts_)
                        *ts_ = ts;
                    this->_ts = ts;
                    this->_fn = fn;
                    return true;
                }
            }
            if(this->input)
            {
                *userVar = this->input->GetDataPtr(fn, this->_ctx, &ts);
                if(*userVar != nullptr)
                {
                    if(ts_)
                        *ts_ = ts;
                    this->_ts = ts;
                    this->_fn = fn;
                    return true;
                }
            }
        }
        return false;
    }

    template<typename T>
    void TInputParamPtr<T>::onInputDelete(IParam const* param)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        this->shared_input.reset();
        this->input = nullptr;
    }
}
#endif
