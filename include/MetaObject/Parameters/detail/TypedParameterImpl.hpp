#pragma once

namespace mo
{
	template<typename T> TypedParameter<T>::TypedParameter(const std::string& name, const T& init, ParameterType type, long lnog ts, Context* ctx) :
		ITypedParameter<T>(name, type, ts, ctx), data(init)
	{
		(void)&_typed_parameter_constructor;
	}
	template<typename T> T* TypedParameter<T>::GetData(long long ts, Context* ctx)
	{
		if (time_index != -1)
			LOGIF_NEQ(time_index, Parameter::_current_time_index, trace);
		return &data;
	}
	template<typename T> bool TypedParameter<T>::GetData(T& value, long long ts, Context* ctx)
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
	template<typename T> ITypedParameter<T>* TypedParameter<T>::UpdateData(T& data_, long long ts = -1, Context* ctx = nullptr)
	{
		data = data_;
		Commit(time_index, ctx);
		return this;
	}
	template<typename T> ITypedParameter<T>* TypedParameter<T>::UpdateData(const T& data_, long long ts, Context* ctx)
	{
		data = data_;
		Commit(time_index, ctx);
		return this;
	}
	template<typename T> ITypedParameter<T>* TypedParameter<T>::UpdateData(T* data_, long long ts, Context* ctx)
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
			if (typed->GetData(data, -1, ctx))
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
}