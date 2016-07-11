/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/parameters
*/
#pragma once
#include "ITypedParameter.hpp"
#include "MetaParameter.hpp"
#include "ParameterFactory.hpp"
#include "parameters/Persistence/CerealPolicy.hpp"
namespace Parameters
{
    template<typename T> class PARAMETER_EXPORTS TypedParameter : 
        public ITypedParameter<T>
#ifdef AUTO_REGISTER_META_PARAMETER
        , public MetaTypedParameter < T >
#endif
        , public Persistence::Cereal::policy<TypedParameter<T>>
    {
    public:
        TypedParameter(const std::string& name = "", const T& init = T(), ParameterType type = kControl);
        
        virtual T*   GetData(long long time_index = -1, Signals::context* ctx = nullptr);
        virtual bool GetData(T& value, long long time_index = -1, Signals::context* ctx = nullptr);

        virtual ITypedParameter<T>* UpdateData(T& data_,       long long time_index = -1, Signals::context* ctx = nullptr);
        virtual ITypedParameter<T>* UpdateData(const T& data_, long long time_index = -1, Signals::context* ctx = nullptr);
        virtual ITypedParameter<T>* UpdateData(T* data_,       long long time_index = -1, Signals::context* ctx = nullptr);

        virtual Parameter* DeepCopy() const;
        virtual bool Update(Parameter* other, Signals::context* ctx = nullptr);
        template<class Archive> void serialize(Archive& ar);

    protected:
        T data;
    private:
        static FactoryRegisterer<TypedParameter<T>, T, TypedParameter_c> _typed_parameter_constructor;
    };
    

    template<typename T> class TypedParameterPtr : 
        public ITypedParameter < T >
#ifdef AUTO_REGISTER_META_PARAMETER
        , public MetaTypedParameter < T >
#endif
    {
    public:
        TypedParameterPtr(const std::string& name = "", T* ptr_ = nullptr, ParameterType type = Control, bool ownsData_ = false);
        virtual ~TypedParameterPtr();

        virtual T*   GetData(long long time_index = -1, Signals::context* ctx = nullptr);
        virtual bool GetData(T& value, long long time_index = -1, Signals::context* ctx = nullptr);

        virtual ITypedParameter<T>* UpdateData(T& data_,       long long time_index = -1, Signals::context* ctx = nullptr);
        virtual ITypedParameter<T>* UpdateData(const T& data_, long long time_index = -1, Signals::context* ctx = nullptr);
        virtual ITypedParameter<T>* UpdateData(T* data_,       long long time_index = -1, Signals::context* ctx = nullptr);
        virtual bool Update(Parameter* other, Signals::context* other_ctx = nullptr);
        virtual Parameter* DeepCopy() const;
    protected:
        T* ptr;
        bool ownsData;
    };


    // Definitions

    template<typename T> TypedParameter<T>::TypedParameter(const std::string& name, const T& init, ParameterType type) :
        ITypedParameter<T>(name, type), data(init) 
    {
        (void)&_typed_parameter_constructor;
    }
    template<typename T> T* TypedParameter<T>::GetData(long long time_index, Signals::context* ctx)
    {
        if(time_index != -1)
            LOGIF_NEQ(time_index, Parameter::_current_time_index, trace);
        return &data;
    }
    template<typename T> bool TypedParameter<T>::GetData(T& value, long long time_index, Signals::context* ctx)
    {
        std::lock_guard<std::recursive_mutex> lock(_mtx);
        T* ptr = GetData(time_index, ctx);
        if (ptr)
        {
            value = *ptr;
            return true;
        }
        return false;
    }
    template<typename T> ITypedParameter<T>* TypedParameter<T>::UpdateData(T& data_, long long time_index = -1, Signals::context* ctx  = nullptr)
    {
        data = data_;
        Commit(time_index, ctx);
        return this;
    }
    template<typename T> ITypedParameter<T>* TypedParameter<T>::UpdateData(const T& data_, long long time_index = -1, Signals::context* ctx = nullptr)
    {
        data = data_;
        Commit(time_index, ctx);
        return this;
    }
    template<typename T> ITypedParameter<T>* TypedParameter<T>::UpdateData(T* data_, long long time_index = -1, Signals::context* ctx  = nullptr)
    {
        data = *data_;
        Commit(time_index, ctx);
        return this;
    }
    template<typename T> Parameter* TypedParameter<T>::DeepCopy() const
    {
        return new TypedParameter<T>(Parameter::GetName(), data);
    }

    template<typename T>  bool TypedParameter<T>::Update(Parameter* other, Signals::context* ctx)
    {
        auto typed = dynamic_cast<ITypedParameter<T>*>(other);
        if (typed)
        {
            if(typed->GetData(data, -1, ctx))
            {
                Commit(other->GetTimeIndex(), ctx);
                return true;
            }
        }
        return false;
    }

    template<typename T> template<class Archive>
    void  TypedParameter<T>::serialize(Archive& ar)
    {
        Parameter::serialize(ar);
        ar(data);
    }

    template<typename T> FactoryRegisterer<TypedParameter<T>, T, TypedParameter_c> TypedParameter<T>::_typed_parameter_constructor;

    template<typename T> TypedParameterPtr<T>::TypedParameterPtr(const std::string& name, T* ptr_, ParameterType type, bool ownsData_) :
            ptr(ptr_), ownsData(ownsData_)
    {
    }
    template<typename T> TypedParameterPtr<T>::~TypedParameterPtr()
    {
        if(ownsData && ptr)
            delete ptr;
    }

    template<typename T> T* TypedParameterPtr<T>::GetData(long long time_index = -1, Signals::context* ctx = nullptr)
    {
        LOGIF_NEQ(time_index, Parameter::_current_time_index, trace);
        return ptr;
    }
    template<typename T> bool TypedParameterPtr<T>::GetData(T& value, long long time_index = -1, Signals::context* ctx = nullptr)
    {
        std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
        LOGIF_NEQ(time_index, Parameter::_current_time_index, trace);
        if (ptr)
        {
            value = *ptr;
            return true;
        }
        return false;
    }
    template<typename T> ITypedParameter<T>* TypedParameterPtr<T>::UpdateData(T& data_, long long time_index, Signals::context* ctx)
    {
        ptr = &data;
        Parameter::_current_time_index = time_index;
        Parameter::changed = true;
        Parameter::OnUpdate(stream);
    }
    template<typename T> ITypedParameter<T>* TypedParameterPtr<T>::UpdateData(const T& data_, long long time_index, Signals::context* ctx)
    {
        if (ptr)
        {
            *ptr = data;
            Parameter::_current_time_index = time_index;
            Parameter::changed = true;
            Parameter::OnUpdate(stream);
        }                
    }
    template<typename T> ITypedParameter<T>* TypedParameterPtr<T>::UpdateData(T* data_, long long time_index, Signals::context* ctx)
    {
        ptr = data_;
        Parameter::_current_time_index = time_index;
        Parameter::changed = true;
        Parameter::OnUpdate(stream);
    }
    template<typename T> bool TypedParameterPtr<T>::Update(Parameter* other, Signals::context* other_ctx)
    {
        auto typed = dynamic_cast<ITypedParameter<T>*>(other);
        if(typed)
        {
            *ptr = *(typed->Data());
            Parameter::_current_time_index = other->GetTimeIndex();
            Parameter::changed = true;
            Parameter::OnUpdate(nullptr);
            return true;
        }
        return false;
    }
    template<typename T> Parameter* TypedParameterPtr<T>::DeepCopy() const
    {
        return new TypedParameter<T>(Parameter::GetName(), *ptr);
    }

}
