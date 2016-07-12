#pragma once
namespace mo
{
	template<typename T> class TypedParameterPtr;

	template<typename T> TypedParameterPtr<T>::TypedParameterPtr(const std::string& name, T* ptr_, ParameterType type, bool ownsData_) :
		ptr(ptr_), ownsData(ownsData_)
	{
	}
	template<typename T> TypedParameterPtr<T>::~TypedParameterPtr()
	{
		if (ownsData && ptr)
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
		if (typed)
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