#pragma once
#ifndef __CUDACC__
#include <boost/thread/recursive_mutex.hpp>
namespace mo
{
	template<typename T> class ITypedParameter;
    template<typename T> class TUpdateToken;
    template<typename T>
    ITypedParameter<T>::ITypedParameter(const std::string& name,
                                        ParameterType flags,
                                        boost::optional<mo::time_t> ts,
                                        Context* ctx,
                                        size_t fn):
        IParameter(name, flags, ts, ctx, fn)
	{
	}

    template<typename T>
    const TypeInfo& ITypedParameter<T>::GetTypeInfo() const
	{
		return _type_info;
	}

    template<typename T>
    bool ITypedParameter<T>::Update(IParameter* other)
	{
		auto typedParameter = dynamic_cast<ITypedParameter<T>*>(other);
		if (typedParameter)
		{
			boost::recursive_mutex::scoped_lock lock(typedParameter->mtx());
            //UpdateData(typedParameter->GetData(), other->GetTimestamp(), other->GetContext());
			OnUpdate(other->GetContext());
			return true;
		}
		return false;
	}
    /*template<typename T>
    TUpdateToken<T> ITypedParameter<T>::Update()
    {
        return TUpdateToken<T>(*this);
    }*/

    template<typename T>
    const TypeInfo ITypedParameter<T>::_type_info(typeid(T));
}
#endif
