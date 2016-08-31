#pragma once
#ifndef __CUDACC__
#include <boost/thread/recursive_mutex.hpp>

namespace mo
{
    template<typename T> class TypedInputParameterPtr;
    template<typename T> TypedInputParameterPtr<T>::TypedInputParameterPtr(const std::string& name, T** userVar_, Context* ctx) :
            userVar(userVar_),
            ITypedInputParameter(name, ctx),
            IParameter(name, Input_e, -1, ctx)
    {
    }
        
    template<typename T>  bool TypedInputParameterPtr<T>::SetInput(std::shared_ptr<IParameter> param)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ITypedInputParameter<T>::SetInput(param))
        {
            if(userVar)
            {
                if(input)
                    *userVar = input->GetDataPtr();
                if(shared_input)
                    *userVar = shared_input->GetDataPtr();
            }
            return true;
        }
        return false;
    }
    template<typename T>  bool TypedInputParameterPtr<T>::SetInput(IParameter* param)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ITypedInputParameter<T>::SetInput(param))
        {
            if(userVar)
            {
                if(input)
                    *userVar = input->GetDataPtr();
                if(shared_input)
                    *userVar = shared_input->GetDataPtr();
            }
            return true;
        }
        return false;
    }
    template<typename T> void TypedInputParameterPtr<T>::SetUserDataPtr(T** user_var_)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        userVar = user_var_;
    }

    template<typename T> void TypedInputParameterPtr<T>::onInputUpdate(Context* ctx, IParameter* param)
    {
        if(input)
        {
            Commit(input->GetTimestamp(), ctx);
            if((ctx && this->_ctx && ctx->thread_id == _ctx->thread_id) || (ctx == nullptr &&  _ctx == nullptr))
            {
                if(userVar)
                    *userVar = input->GetDataPtr();
            }
        }else if(shared_input)
        {
            Commit(shared_input->GetTimestamp(), ctx);
            if((ctx && this->_ctx && ctx->thread_id == _ctx->thread_id) || (ctx == nullptr &&  _ctx == nullptr))
            {
                if(userVar)
                    *userVar = shared_input->GetDataPtr();
            }
        }
    }
    template<typename T>
    bool TypedInputParameterPtr<T>::GetInput(long long ts)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(userVar)
        {
            if(shared_input)
            {
                *userVar = shared_input->GetDataPtr(ts, this->_ctx);
                return *userVar != nullptr;
            }
            if(input)
            {
                *userVar = input->GetDataPtr(ts, this->_ctx);
                return *userVar != nullptr;
            }
        }
        return false;
    }

    template<typename T> void TypedInputParameterPtr<T>::onInputDelete(IParameter const* param)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        shared_input.reset();
        input = nullptr;
    }
}
#endif