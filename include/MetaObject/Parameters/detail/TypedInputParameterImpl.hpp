#pragma once
#include <functional>
namespace mo
{
    template<typename T> class TypedInputParameter;

    template<typename T> TypedInputParameter<T>::TypedInputParameter(const std::string& name, std::function<bool(Parameter*)> qualifier_):
        ITypedParameter<T>(name, Input_e)
    { 
        qualifier = qualifier_;
        input = nullptr;
    }
    template<typename T> TypedInputParameter<T>::~TypedInputParameter()
    {
        if (input)
            input->subscribers--;
    }

    template<typename T>  bool TypedInputParameter<T>::SetInput(weak_ptr<IParameter> param_)
    {
        std::shared_ptr<IParameter> param(param_);
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
        auto castedParam = std::dynamic_pointer_cast<ITypedParameter<T>>(param);
        if (castedParam)
        {
            if(input)
            {
                input->subscribers--;
            }
            input = castedParam;
            input->subscribers++;
            inputConnection = castedParam->RegisterNotifier(std::bind(&TypedInputParameter<T>::onInputUpdate, this));
            deleteConnection = castedParam->RegisterDeleteNotifier(std::bind(&TypedInputParameter<T>::onInputDelete, this));
            Parameter::OnUpdate(nullptr);
            return true;
        }
        return false;
    }

    template<typename T> weak_ptr<IParameter> TypedInputParameter<T>::GetInput()
    {
        return input;
    }

     template<typename T> bool TypedInputParameter<T>::AcceptsInput(weak_ptr<IParameter> param)
    {
        if (qualifier)
            return qualifier(param);
        return Loki::TypeInfo(typeid(T)) == param->GetTypeInfo();
    }

     template<typename T> bool TypedInputParameter<T>::AcceptsType(TypeInfo type)
    {
        return TypeInfo(typeid(T)) == type;
    }
     template<typename T>  T* TypedInputParameter<T>::GetDataPtr(long long ts)
    {
        std::lock_guard<std::recursive_mutex> lock(mtx());
        if (input)
            return input->Data(time_step);
        return nullptr;
    }
     template<typename T>  bool TypedInputParameter<T>::GetData(T& value, long long time_step)
    {
        std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
        if (input)
        {
            return input->GetData(value, time_step);
        }
        return nullptr;
    }
    template<typename T>  void TypedInputParameter<T>::UpdateData(T& data_, long long time_index, Context* ctx){ }
    template<typename T>  void TypedInputParameter<T>::UpdateData(const T& data_, long long time_index, Context* ctx){   }
    template<typename T>  void TypedInputParameter<T>::UpdateData(T* data_, long long time_index, Context* ctx){ }

    template<typename T>  TypeInfo TypedInputParameter<T>::GetTypeInfo()
    {
        return TypeInfo(typeid(T));
    }
    template<typename T>  IParameter::Ptr TypedInputParameter<T>::DeepCopy() const
    {
        return IParameter::Ptr();
    }
    template<typename T>  void TypedInputParameter<T>::onInputUpdate()
    {
        Parameter::OnUpdate(nullptr);
    }
    template<typename T>  void TypedInputParameter<T>::onInputDelete()
    {
        std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
        input = nullptr;
        Parameter::OnUpdate(nullptr);
    }
}