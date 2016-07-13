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
#include "InputParameter.hpp"

namespace mo
{
    template<typename T> class TypedInputParameter : public ITypedParameter<T>, public InputParameter
    {
    public:
        TypedInputParameter(const std::string& name, std::function<bool(Parameter*)> qualifier_ = std::function<bool(Parameter*)>());
        ~TypedInputParameter();

        virtual bool SetInput(weak_ptr<IParameter> param);
        std::weak_ptr<IParameter> GetInput();

        virtual bool AcceptsInput(weak_ptr<IParameter> param) const;
        virtual bool AcceptsType(TypeInfo type) const;

        virtual T*   GetDataPtr(long long time_step = -1, Context* ctx = nullptr);
        virtual T    GetData(long long ts = -1, Context* ctx = nullptr);
        virtual bool GetData(T& value, long long ts = -1);

        virtual void UpdateData(T& data_,       long long ts = -1, Context* ctx = nullptr);
        virtual void UpdateData(const T& data_, long long ts = -1, Context* ctx = nullptr);
        virtual void UpdateData(T* data_,       long long ts = -1, Context* ctx = nullptr);

        virtual Loki::TypeInfo GetTypeInfo();
        virtual IParameter::Ptr DeepCopy() const;
    private:
        std::shared_ptr<ITypedParameter<T>> input;
        std::shared_ptr<Signals::connection> inputConnection;
        std::shared_ptr<Signals::connection> deleteConnection;
        virtual void onInputUpdate();
        virtual void onInputDelete();
    };

    template<typename T> class TypedInputParameterCopy : public MetaTypedParameter<T>, public InputParameter
    {
        T* userVar; // Pointer to the user space variable of type T
        ITypedParameter<T>* input;
        std::shared_ptr<Signals::connection> inputConnection;
        std::shared_ptr<Signals::connection> deleteConnection;

        virtual void onInputUpdate()
        {
            *userVar = *input->Data();
            Parameter::OnUpdate(nullptr);
        }
        virtual void onInputDelete()
        {
            input = nullptr;
            Parameter::OnUpdate(nullptr);
        }
    public:
        typedef std::shared_ptr<TypedInputParameterCopy<T>> Ptr;
        static Ptr create(T* userVar_)
        {
            return Ptr(new TypedInputParameterCopy(userVar_));
        }
        TypedInputParameterCopy(const std::string& name, T* userVar_,
            ParameterType type = kControl) :
            MetaTypedParameter<T>(name, type, tooltip), userVar(userVar_) 
        {
            input = nullptr;
        }
        ~TypedInputParameterCopy()
        {
            //inputConnection.disconnect();
        }
        virtual bool SetInput(const std::string& name_)
        {
            return false;
        }

        virtual bool SetInput(Parameter* param)
        {
            if(input)
                input->subscribers--;
            ITypedParameter<T>* castedParam = dynamic_cast<ITypedParameter<T>*>(param);
            if (castedParam)
            {
                input = castedParam;
                param->subscribers++;
                inputConnection = castedParam->RegisterNotifier(std::bind(&TypedInputParameterCopy<T>::onInputUpdate, this));
                deleteConnection = castedParam->RegisterDeleteNotifier(std::bind(&TypedInputParameterCopy<T>::onInputDelete, this));
                *userVar = *input->Data();
                return true;
            }
            return false;
        }
        Parameter* GetInput()
        {
            return input;
        }

        virtual bool AcceptsInput(Parameter* param)
        {
            if (qualifier)
                return qualifier(param);
            return Loki::TypeInfo(typeid(T)) == param->GetTypeInfo();
        }

        virtual bool AcceptsType(const Loki::TypeInfo& type)
        {
            return Loki::TypeInfo(typeid(T)) == type;
        } 
        virtual T* Data(long long time_step = -1)
        {
            return userVar;
        }
        virtual bool GetData(T& value, long long time_step = -1)
        {
            if (userVar)
            {
                value = *userVar;
                return true;
            }
            return false;                
        }
        virtual void UpdateData(T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {
            *userVar = data_;
        }
        virtual void UpdateData(const T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {
            *userVar = data_;
        }
        virtual void UpdateData(T* data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {
            *userVar = *data_;
        }
        virtual Loki::TypeInfo GetTypeInfo()
        {
            return Loki::TypeInfo(typeid(T));
        }
        virtual Parameter::Ptr DeepCopy() const
        {
            return Parameter::Ptr();
        }
    };

    // Meant to reference a pointer variable in user space, and to update that variable whenever 
    // IE int* myVar; 
    // auto typedParam = TypedInputParameterPtr(&myVar); // TypedInputParameter now updates myvar to point to whatever the
    // input variable is for typedParam.
    template<typename T> class TypedInputParameterPtr : public InputParameter, public ITypedParameter<T>
    {
    public:
        typedef std::shared_ptr<TypedInputParameterPtr<T>> Ptr;
        TypedInputParameterPtr(const std::string& name, T** userVar_, ParameterType type = Control_e);
        virtual bool SetInput(std::shared_ptr<IParameter> param);
        std::shared_ptr<IParameter> GetInput();
        virtual bool AcceptsInput(std::weak_ptr<IParameter> param) const;
        virtual bool AcceptsType(TypeInfo type) const;
        virtual T* GetDataPtr(long long ts, Context* ctx = nullptr);
        T GetData(long long ts = -1, Context* ctx = nullptr);
        bool GetData(T& value, long long ts = -1, Context* ctx = nullptr);
        virtual void UpdateData(T& data_, long long time_index = -1, Context* ctx = nullptr);
        virtual void UpdateData(const T& data_, long long time_index = -1, Context* ctx = nullptr);
        virtual void UpdateData(T* data_, long long time_index = -1, Context* ctx = nullptr);
        virtual TypeInfo GetTypeInfo() const;
        virtual IParameter::Ptr DeepCopy() const;
    private:
        T** userVar; // Pointer to the user space pointer variable of type T
        std::shared_ptr<ITypedParameter<T>> input;
        std::shared_ptr<Signals::connection> inputConnection;
        std::shared_ptr<Signals::connection> deleteConnection;
        virtual void onInputUpdate();
        virtual void onInputDelete();
    };
}
#include "MetaObject/Parameters/detail/TypedInputParameterImpl.hpp"
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
