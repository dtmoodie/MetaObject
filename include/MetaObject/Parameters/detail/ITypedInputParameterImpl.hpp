#pragma once
#ifndef __CUDACC__
#include "MetaObject/Logging/Log.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include <functional>

namespace mo
{
    template<class T> 
    class ITypedInputParameter;

    template<class T> 
    ITypedInputParameter<T>::ITypedInputParameter(const std::string& name, Context* ctx):
            ITypedParameter<T>(name, Input_e, -1* mo::second, ctx),
            input(nullptr),
            IParameter(name, Input_e)
    {
		update_slot = std::bind(&ITypedInputParameter<T>::onInputUpdate, this, std::placeholders::_1, std::placeholders::_2);
		delete_slot = std::bind(&ITypedInputParameter<T>::onInputDelete, this, std::placeholders::_1);
    }
    
    template<class T> 
    ITypedInputParameter<T>::~ITypedInputParameter()
    {
        if(input)
            input->Unsubscribe();
        if(shared_input)
            shared_input->Unsubscribe();
    }
    
    template<class T> 
    bool ITypedInputParameter<T>::SetInput(std::shared_ptr<IParameter> param)
    {
        boost::recursive_mutex::scoped_lock lock(this->mtx());
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
            this->OnUpdate(nullptr);
            return true;
        }
        auto casted_param = std::dynamic_pointer_cast<ITypedParameter<T>>(param);
        if(casted_param)
        {
            if(input) input->Unsubscribe();
            if(shared_input) shared_input->Unsubscribe();
            update_slot.Clear();
            delete_slot.Clear();
            auto ts = casted_param->GetTimestamp();
            //UpdateData(*casted_param->GetDataPtr(), ts, casted_param->GetContext());
            shared_input = casted_param;
            casted_param->RegisterUpdateNotifier(&update_slot);
			casted_param->RegisterDeleteNotifier(&delete_slot);

            this->OnUpdate(casted_param->GetContext());
            return true;
        }
        return false;
    }

    template<class T> 
    bool ITypedInputParameter<T>::SetInput(IParameter* param)
    {
        boost::recursive_mutex::scoped_lock lock(this->mtx());
        if(param == nullptr)
        {
            if(shared_input)
            {
                shared_input->Unsubscribe();
            }else if(input)
            {
                input->Unsubscribe();
            }
            update_slot.Clear();
            delete_slot.Clear();
            input = nullptr;
            shared_input.reset();
            this->OnUpdate(nullptr);
            return true;
        }
        auto casted_param = dynamic_cast<ITypedParameter<T>*>(param);
        if(casted_param)
        {
            if(input) input->Unsubscribe();
            if(shared_input) shared_input->Unsubscribe();
            //ITypedParameter<T>::UpdateData(*casted_param->GetDataPtr(), tag::_timestamp = casted_param->GetTimestamp(), tag::_context = casted_param->GetContext());

            input = casted_param;
            input->Subscribe();
			casted_param->RegisterUpdateNotifier(&update_slot);
			casted_param->RegisterDeleteNotifier(&delete_slot);
            this->OnUpdate(casted_param->GetContext());
            return true;
        }
        return false;
    }

    template<class T> 
    bool ITypedInputParameter<T>::AcceptsInput(std::weak_ptr<IParameter> param) const
    {
        if(auto ptr = param.lock())
            return ptr->GetTypeInfo() == GetTypeInfo();
        return false;
    }

    template<class T> 
    bool ITypedInputParameter<T>::AcceptsInput(IParameter* param) const
    {
        return param->GetTypeInfo() == GetTypeInfo();
    }

    template<class T> 
    bool ITypedInputParameter<T>::AcceptsType(TypeInfo type) const
    {
        return type == GetTypeInfo();
    }

    template<class T> 
    IParameter* ITypedInputParameter<T>::GetInputParam()
    {
        if(shared_input)
            return shared_input.get();
        return input;
    }
    
    template<class T> 
    T* ITypedInputParameter<T>::GetDataPtr(boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
    {
        if(input)
            return input->GetDataPtr(ts, ctx, fn);
        if(shared_input)
            return shared_input->GetDataPtr(ts, ctx, fn);
        return nullptr;
    }

    template<class T>
    T* ITypedInputParameter<T>::GetDataPtr(size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
    {
        if(input)
            return input->GetDataPtr(fn, ctx, ts);
        if(shared_input)
            return shared_input->GetDataPtr(fn, ctx, ts);
        return nullptr;
    }

    template<class T> 
    bool ITypedInputParameter<T>::GetData(T& value, boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
    {
        if(input)
            return input->GetData(value, ts, ctx, fn);
        if(shared_input)
            return shared_input->GetData(value, ts, ctx, fn);
        return false;
    }

    template<class T>
    bool ITypedInputParameter<T>::GetData(T& value, size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
    {
        if(input)
            return input->GetData(value, fn, ctx, ts);
        if(shared_input)
            return shared_input->GetData(value, fn, ctx, ts);
        return false;
    }
    
    template<class T> 
    T ITypedInputParameter<T>::GetData(boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
    {
        if(input)
            return input->GetData(ts, ctx, fn);
        if(shared_input)
            return shared_input->GetData(ts, ctx, fn);
        THROW(debug) << "Input not set for " << GetTreeName();
        return T();
    }

    template<class T>
    T ITypedInputParameter<T>::GetData(size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
    {
        if(input)
            return input->GetData(fn, ctx, ts);
        if(shared_input)
            return shared_input->GetData(fn, ctx, ts);
        THROW(debug) << "Input not set for " << GetTreeName();
        return T();
    }

    template<class T>
    bool ITypedInputParameter<T>::GetInput(mo::time_t ts)
    {
        return true;
    }

    template<class T>
    bool ITypedInputParameter<T>::GetInput(size_t ts)
    {
        return true;
    }
    
    // ---- protected functions
    template<class T> 
    void ITypedInputParameter<T>::onInputDelete(IParameter const* param)
    {
        boost::recursive_mutex::scoped_lock lock(this->mtx());
        this->shared_input.reset();
        this->input = nullptr;
        this->OnUpdate(GetContext());
    }
    
    
    template<class T> 
    void ITypedInputParameter<T>::onInputUpdate(Context* ctx, IParameter* param)
    {
        this->OnUpdate(ctx);
    }
}
#endif
