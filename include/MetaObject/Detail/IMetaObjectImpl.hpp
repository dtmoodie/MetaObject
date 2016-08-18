#pragma once
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Parameters/TypedParameter.hpp"
namespace mo
{
    class IMetaObject;

    template<class T> 
    ITypedParameter<T>* IMetaObject::GetParameter(const std::string& name) const
    {
        auto param = GetParameter(name);
        auto typed = static_cast<ITypedParameter<T>*>(param);
        if(typed)
        {
            return typed;
        }
        THROW(debug) << "Parameter \"" << name << "\" not convertable to type " << TypeInfo(typeid(T)).name();
    }

    template<class T> 
    T IMetaObject::GetParameterValue(const std::string& name, long long ts, Context* ctx) const
    {
        return GetParameter<T>(name)->_Get_data(ts, ctx);
    }
    template<class T> 
    ITypedParameter<T>* IMetaObject::GetParameterOptional(const std::string& name) const
    {
        auto param = GetParameterOptional(name);
        auto typed = static_cast<ITypedParameter<T>*>(param);
        return typed;
    }

    template<class T> 
    ITypedParameter<T>* IMetaObject::UpdateParameter(const std::string& name, T&& value, long long ts, Context* ctx)
    {
        if(ctx == nullptr)
            ctx = _ctx;
        auto param = GetParameterOptional<T>(name);
        if(param)
        {
            param->UpdateData(std::forward<T>(value), ts, ctx);
            return param;
        }else
        {
            std::shared_ptr<ITypedParameter<T>> new_param(new TypedParameter<T>(name, value));
            AddParameter(new_param);
            return new_param.get();
        }
    }
    template<class T> 
    ITypedParameter<T>* IMetaObject::UpdateParameterPtr(const std::string& name, T& ptr)
    {
        return nullptr;
    }

    template<class Sig> 
    bool IMetaObject::ConnectCallback(const std::string& callback_name, const std::string& slot_name, IMetaObject* slot_owner, bool force_queue)
    {
        ConnectCallback(TypedInfo(typeid(Sig)), callback_name, slot_name, slot_owner, force_queue);
    }

}