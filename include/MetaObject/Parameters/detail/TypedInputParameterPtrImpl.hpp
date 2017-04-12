#pragma once
#ifndef __CUDACC__
#include <boost/thread/recursive_mutex.hpp>
#include "MetaObject/Parameters/MetaParameter.hpp"
namespace mo
{
    template<typename T> class TypedInputParameterPtr;
    class Context;

    template<typename T> TypedInputParameterPtr<T>::TypedInputParameterPtr(const std::string& name, const T** userVar_, Context* ctx) :
            userVar(userVar_),
            ITypedInputParameter<T>(name, ctx),
            IParameter(name, mo::Input_e)
    {
    }

    template<typename T>
    bool TypedInputParameterPtr<T>::SetInput(std::shared_ptr<IParameter> param)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ITypedInputParameter<T>::SetInput(param))
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
    bool TypedInputParameterPtr<T>::SetInput(IParameter* param)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ITypedInputParameter<T>::SetInput(param))
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
    void TypedInputParameterPtr<T>::SetUserDataPtr(const T** user_var_)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        userVar = user_var_;
    }

    template<typename T>
    void TypedInputParameterPtr<T>::onInputUpdate(Context* ctx, IParameter* param)
    {
        if(this->input)
        {
            boost::recursive_mutex::scoped_lock lock(this->input->mtx());
            //this->Commit(this->input->GetTimestamp(), ctx, this->input->GetFrameNumber(), this->input->GetCoordinateSystem());
            this->_update_signal(ctx, this);
            if((ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) || (ctx == nullptr &&  this->_ctx == nullptr))
            {
                if(userVar)
                    *userVar = this->input->GetDataPtr();
            }
        }else if(this->shared_input)
        {
            boost::recursive_mutex::scoped_lock lock(this->shared_input->mtx());
            //this->Commit(this->shared_input->GetTimestamp(), ctx, this->shared_input->GetFrameNumber(), this->shared_input->GetCoordinateSystem());
            this->_update_signal(ctx, this);
            if((ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) || ((ctx == nullptr) &&  (this->_ctx == nullptr)))
            {
                if(userVar)
                    *userVar = this->shared_input->GetDataPtr();
            }
        }
    }

    template<typename T>
    bool TypedInputParameterPtr<T>::GetInput(boost::optional<mo::time_t> ts, size_t* fn_)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
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
    bool TypedInputParameterPtr<T>::GetInput(size_t fn, boost::optional<mo::time_t>* ts_)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        boost::optional<mo::time_t> ts;
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
    void TypedInputParameterPtr<T>::onInputDelete(IParameter const* param)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        this->shared_input.reset();
        this->input = nullptr;
    }
}
#endif
