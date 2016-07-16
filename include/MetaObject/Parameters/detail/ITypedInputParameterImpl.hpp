#pragma once
#include <functional>
#include "MetaObject/Logging/Log.hpp"
namespace mo
{
    template<class T> class ITypedInputParameter;

    template<class T> ITypedInputParameter<T>::ITypedInputParameter(const std::string& name, Context* ctx):
            ITypedParameter(name, Input_e, -1, ctx)
    {
    }
    template<class T> ITypedInputParameter<T>::~ITypedInputParameter()
    {
        if(input)
            input->Unsubscribe();
        if(shared_input)
            shared_input->Unsubscribe();
    }
    template<class T> bool ITypedInputParameter<T>::SetInput(std::shared_ptr<IParameter> param)
    {
        std::lock_guard<std::recursive_mutex> lock(this->_mtx);
        if(param == nullptr)
        {
            if(shared_input)
            {
                shared_input->Unsubscribe();
            }else if(input)
            {
                input->Unsubscribe();
            }
            input = nullptr;
            shared_input.reset();
            inputConnection.reset();
            deleteConnection.reset();
            this->OnUpdate(nullptr);
            return true;
        }
        auto casted_param = std::dynamic_pointer_cast<ITypedParameter<T>>(param);
        if(casted_param)
        {
            if(input) input->Unsubscribe();
            if(shared_input) shared_input->Unsubscribe();

            shared_input = casted_param;
            inputConnection = casted_param->RegisterUpdateNotifier(
                std::bind(&ITypedInputParameter<T>::onInputUpdate, this, std::placeholders::_1, std::placeholders::_2));
            deleteConnection = casted_param->RegisterDeleteNotifier(
                std::bind(&ITypedInputParameter<T>::onInputDelete, this, std::placeholders::_1));
            this->OnUpdate(casted_param->GetContext());
            return true;
        }
        return false;
    }

    template<class T> bool ITypedInputParameter<T>::SetInput(IParameter* param)
    {
        std::lock_guard<std::recursive_mutex> lock(this->_mtx);
        if(param == nullptr)
        {
            if(shared_input)
            {
                shared_input->Unsubscribe();
            }else if(input)
            {
                input->Unsubscribe();
            }
            input = nullptr;
            shared_input.reset();
            inputConnection.reset();
            deleteConnection.reset();
            this->OnUpdate(nullptr);
            return true;
        }
        auto casted_param = dynamic_cast<ITypedParameter<T>*>(param);
        if(casted_param)
        {
            if(input) input->Unsubscribe();
            if(shared_input) shared_input->Unsubscribe();

            input = casted_param;
            inputConnection = casted_param->RegisterUpdateNotifier(
                std::bind(&ITypedInputParameter<T>::onInputUpdate, this, std::placeholders::_1, std::placeholders::_2));
            deleteConnection = casted_param->RegisterDeleteNotifier(
                std::bind(&ITypedInputParameter<T>::onInputDelete, this, std::placeholders::_1));
            this->OnUpdate(casted_param->GetContext());
            return true;
        }
        return false;
    }

    template<class T> bool ITypedInputParameter<T>::AcceptsInput(std::weak_ptr<IParameter> param) const
    {
        if(auto ptr = param.lock())
            return ptr->GetTypeInfo() == GetTypeInfo();
        return false;
    }

    template<class T> bool ITypedInputParameter<T>::AcceptsInput(IParameter* param) const
    {
        return param->GetTypeInfo() == GetTypeInfo();
    }

    template<class T> bool ITypedInputParameter<T>::AcceptsType(TypeInfo type) const
    {
        return type == GetTypeInfo();
    }

    template<class T> IParameter* ITypedInputParameter<T>::GetInput()
    {
        if(shared_input)
            return shared_input.get();
        return input;
    }
    template<class T> T* ITypedInputParameter<T>::GetDataPtr(long long ts = -1, Context* ctx = nullptr)
    {
        if(input)
            return input->GetDataPtr(ts, ctx);
        if(shared_input)
            return shared_input->GetDataPtr(ts, ctx);
        return nullptr;
    }
    template<class T> bool ITypedInputParameter<T>::GetData(T& value, long long ts = -1, Context* ctx = nullptr)
    {
        if(input)
            return input->GetData(value, ts, ctx);
        if(shared_input)
            return shared_input->GetData(value, ts, ctx);
        return false;
    }
    template<class T> T ITypedInputParameter<T>::GetData(long long ts = -1, Context* ctx = nullptr)
    {
        if(input)
            return input->GetData(ts, ctx);
        if(shared_input)
            return shared_input->GetData(ts, ctx);
        THROW(debug) << "Input not set for " << GetTreeName();
        return T();
    }

    // ---- protected functions
    template<class T> void ITypedInputParameter<T>::onInputDelete(IParameter* param)
    {
        std::lock_guard<std::recursive_mutex> lock(this->_mtx);
        this->shared_input.reset();
        this->input = nullptr;
        this->OnUpdate(GetContext());
    }
    
    template<class T> void ITypedInputParameter<T>::onInputUpdate(Context* ctx, IParameter* param)
    {
        this->OnUpdate(ctx);
    }
}