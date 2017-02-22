#pragma once
#include "ITypedParameter.hpp"
#include "MetaParameter.hpp"
namespace mo
{
	template<typename T> class TypedParameterPtr :public ITypedParameter< T >
	{
	public:
		TypedParameterPtr(const std::string& name = "", T* ptr_ = nullptr, ParameterType type = Control_e, bool ownsData_ = false);
		virtual ~TypedParameterPtr();

		virtual T*   GetDataPtr(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr);
        virtual T    GetData(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr);
		virtual bool GetData(T& value, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr);

        virtual ITypedParameter<T>* UpdateData(T& data_,       mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t fn = std::numeric_limits<size_t>::max());
        virtual ITypedParameter<T>* UpdateData(const T& data_, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t fn = std::numeric_limits<size_t>::max());
        virtual ITypedParameter<T>* UpdateData(T* data_,       mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t fn = std::numeric_limits<size_t>::max());

		virtual bool Update(IParameter* other);
		virtual std::shared_ptr<IParameter> DeepCopy() const;
        ITypedParameter<T>* UpdatePtr(T* ptr);
	protected:
		T* ptr;
		bool ownsData;
        static MetaParameter<T, 100> _meta_parameter;
	};
}
#include "detail/TypedParameterPtrImpl.hpp"
