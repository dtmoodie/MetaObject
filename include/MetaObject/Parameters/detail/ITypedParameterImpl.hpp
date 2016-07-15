#pragma once

namespace mo
{
	template<typename T> class ITypedParameter;

	template<typename T> ITypedParameter<T>::ITypedParameter(const std::string& name, ParameterType flags, long long ts, Context* ctx) :
		IParameter(name, flags, ts, ctx)
	{
	}

	template<typename T> TypeInfo ITypedParameter<T>::GetTypeInfo() const
	{
		return TypeInfo(typeid(T));
	}

	template<typename T> bool ITypedParameter<T>::Update(IParameter* other)
	{
		auto typedParameter = dynamic_cast<ITypedParameter<T>*>(other);
		if (typedParameter)
		{
			std::lock_guard<std::recursive_mutex> lock(typedParameter->mtx());
			UpdateData(typedParameter->GetData(), other->GetTimestamp(), other->GetContext());
			OnUpdate(other->GetContext());
			return true;
		}
		return false;
	}
}