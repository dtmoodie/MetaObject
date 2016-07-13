namespace mo
{
    template<typename T> class TypedInputParameterPtr;
    template<typename T> TypedInputParameterPtr<T>::TypedInputParameterPtr(const std::string& name, T** userVar_, ParameterType type) :
            userVar(userVar_)
    {
        input = nullptr;
    }
        
    template<typename T>  bool TypedInputParameterPtr<T>::SetInput(std::shared_ptr<IParameter> param)
    {
        if (param == nullptr)
        {
            if(input)
                input->Unsubscribe();
                
            input = nullptr;
            inputConnection.reset();
            deleteConnection.reset();
            IParameter::OnUpdate(nullptr);
            return true;
        }
        auto castedParam = std::dynamic_pointer_cast<ITypedParameter<T>>(param);
        if (castedParam)
        {
            if(input)
                input->Unsubscribe();
                
            input = castedParam;
            input->Subscribe();
            inputConnection.reset();
            inputConnection = castedParam->RegisterNotifier(std::bind(&TypedInputParameterPtr<T>::onInputUpdate, this));
            deleteConnection = castedParam->RegisterDeleteNotifier(std::bind(&TypedInputParameterPtr<T>::onInputDelete, this));
            *userVar = input->GetDataPtr();
            return true;
        }
        return false;
    }
    template<typename T> std::shared_ptr<IParameter> TypedInputParameterPtr<T>::GetInput()
    {
        return input;
    }

    template<typename T> bool TypedInputParameterPtr<T>::AcceptsInput(std::weak_ptr<IParameter> param) const
    {
        if (qualifier)
            return qualifier(param);
        return TypeInfo(typeid(T)) == param->GetTypeInfo();
    }

    template<typename T> bool TypedInputParameterPtr<T>::AcceptsType(TypeInfo type) const
    {
        return TypeInfo(typeid(T)) == type;
    }
    template<typename T> T* TypedInputParameterPtr<T>::GetDataPtr(long long ts, Context* ctx = nullptr)
    {
        if (input)
            return input->Data();
        return nullptr;
    }
    template<typename T> T TypedInputParameterPtr<T>::GetData(long long ts = -1, Context* ctx = nullptr)
    {
        if(input)
            return input->GetData();

    }
    template<typename T> bool TypedInputParameterPtr<T>::GetData(T& value, long long ts = -1, Context* ctx = nullptr)
    {
        if(input)
            return input->GetData(value, ts, ctx);
        return false;
    }
    template<typename T> void TypedInputParameterPtr<T>::UpdateData(T& data_, long long time_index = -1, Context* ctx = nullptr){}
    template<typename T> void TypedInputParameterPtr<T>::UpdateData(const T& data_, long long time_index = -1, Context* ctx = nullptr){}
    template<typename T> void TypedInputParameterPtr<T>::UpdateData(T* data_, long long time_index = -1, Context* ctx = nullptr){}
    template<typename T> TypeInfo TypedInputParameterPtr<T>::GetTypeInfo() const
    {
        return TypeInfo(typeid(T));
    }
    template<typename T> IParameter::Ptr DeepCopy() const
    {
        return IParameter::Ptr();
    }  
}