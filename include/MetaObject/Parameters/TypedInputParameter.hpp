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

namespace Parameters
{
    template<typename T> class TypedInputParameter : public ITypedParameter<T>, public InputParameter
    {
        ITypedParameter<T>* input;
        std::shared_ptr<Signals::connection> inputConnection;
        std::shared_ptr<Signals::connection> deleteConnection;
        virtual void onInputUpdate()
        {
            Parameter::OnUpdate(nullptr);
        }
        virtual void onInputDelete()
        {
            std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
            input = nullptr;
            Parameter::OnUpdate(nullptr);
        }
    public:
        typedef std::shared_ptr<TypedInputParameter<T>> Ptr;

        TypedInputParameter(const std::string& name, 
            const std::string& tooltip = "", 
            const std::function<bool(Parameter*)>& qualifier_ = std::function<bool(Parameter*)>()) :
            ITypedParameter<T>(name, Parameter::ParameterType::Input, tooltip)
        { 
            qualifier = qualifier_;
            input = nullptr;
        }
        ~TypedInputParameter()
        {
            if (input)
                input->subscribers--;
        }

        virtual bool SetInput(const std::string& name_)
        {
            return false;
        }

        virtual bool SetInput(Parameter* param)
        {
            std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
            if (param == nullptr)
            {
                if (input)
                {
                    input->subscribers--;
                }
                input = nullptr;
                inputConnection.reset();
                deleteConnection.reset();
                Parameter::OnUpdate(nullptr);
                return true;
            }                
            ITypedParameter<T>* castedParam = dynamic_cast<ITypedParameter<T>*>(param);
            if(input)
            {
                input->subscribers--;
            }
            if (castedParam)
            {
                input = castedParam;
                input->subscribers++;
                inputConnection = castedParam->RegisterNotifier(std::bind(&TypedInputParameter<T>::onInputUpdate, this));
                deleteConnection = castedParam->RegisterDeleteNotifier(std::bind(&TypedInputParameter<T>::onInputDelete, this));
                Parameter::OnUpdate(nullptr);
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
            std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
            if (input)
                return input->Data(time_step);
            return nullptr;
        }
        virtual bool GetData(T& value, long long time_step = -1)
        {
            std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
            if (input)
            {
                return input->GetData(value, time_step);
            }
            return nullptr;
        }
        virtual void UpdateData(T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {

        }
        virtual void UpdateData(const T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {

        }
        virtual void UpdateData(T* data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {

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
    template<typename T> class TypedInputParameterPtr : public MetaTypedParameter<T>, public InputParameter
    {
        T** userVar; // Pointer to the user space pointer variable of type T
        ITypedParameter<T>* input;
        std::shared_ptr<Signals::connection> inputConnection;
        std::shared_ptr<Signals::connection> deleteConnection;
        virtual void onInputUpdate()
        {
            // The input variable has been updated, update user var
            *userVar = input->Data();
            Parameter::OnUpdate(nullptr);
        }
        virtual void onInputDelete()
        {
            // The input variable has been updated, update user var
            input = nullptr;
            inputConnection.reset();
            deleteConnection.reset();
            Parameter::OnUpdate(nullptr);
        }
    public:
        typedef std::shared_ptr<TypedInputParameterPtr<T>> Ptr;
        static Ptr create(T** userVar_)
        {
            return Ptr(new TypedInputParameterPtr(userVar_));
        }
        TypedInputParameterPtr(const std::string& name, T** userVar_, ParameterType type = kControl) :
            userVar(userVar_)
        {
            input = nullptr;
        }
        
        virtual bool SetInput(const std::string& name_)
        {
            return false;
        }

        virtual bool SetInput(Parameter* param)
        {
            if (param == nullptr)
            {
                if (input)
                    --input->subscribers;
                input = nullptr;
                inputConnection.reset();
                deleteConnection.reset();
                Parameter::OnUpdate(nullptr);
                return true;
            }
            typename ITypedParameter<T>::Ptr castedParam = std::dynamic_pointer_cast<ITypedParameter<T>>(param);
            if (castedParam)
            {
                input = castedParam;
                inputConnection.reset();
                inputConnection = castedParam->RegisterNotifier(std::bind(&TypedInputParameterPtr<T>::onInputUpdate, this));
                deleteConnection = castedParam->RegisterDeleteNotifier(std::bind(&TypedInputParameterPtr<T>::onInputDelete, this));
                *userVar = input->Data();
                return true;
            }
            return false;
        }
        std::shared_ptr<Parameter> GetInput()
        {
            return input;
        }

        virtual bool AcceptsInput(const Parameter::Ptr param)
        {
            if (qualifier)
                return qualifier(param.get());
            return Loki::TypeInfo(typeid(T)) == param->GetTypeInfo();
        }

        virtual bool AcceptsType(const Loki::TypeInfo& type)
        {
            return Loki::TypeInfo(typeid(T)) == type;
        }
        virtual T* Data()
        {
            if (input)
                return input->Data();
            return nullptr;
        }
        virtual void UpdateData(T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {

        }
        virtual void UpdateData(const T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {

        }
        virtual void UpdateData(T* data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {

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
}
