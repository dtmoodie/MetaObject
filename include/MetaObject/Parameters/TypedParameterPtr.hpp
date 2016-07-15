#pragma once
#include "ITypedParameter.hpp"
namespace mo
{
	template<typename T> class TypedParameterPtr :public ITypedParameter < T >
	{
	public:
		TypedParameterPtr(const std::string& name = "", T* ptr_ = nullptr, ParameterType type = Control_e, bool ownsData_ = false);
		virtual ~TypedParameterPtr();

		virtual T*   GetDataPtr(long long ts = -1, Context* ctx = nullptr);
        virtual T    GetData(long long ts = -1, Context* ctx = nullptr);
		virtual bool GetData(T& value, long long ts = -1, Context* ctx = nullptr);

		virtual ITypedParameter<T>* UpdateData(T& data_,       long long ts = -1, Context* ctx = nullptr);
		virtual ITypedParameter<T>* UpdateData(const T& data_, long long ts = -1, Context* ctx = nullptr);
		virtual ITypedParameter<T>* UpdateData(T* data_,       long long ts = -1, Context* ctx = nullptr);
		virtual bool Update(IParameter* other);
		virtual std::shared_ptr<IParameter> DeepCopy() const;
	protected:
		T* ptr;
		bool ownsData;
	};
}
#include "detail/TypedParameterPtrImpl.hpp"