#pragma once
#include "MetaObject/Logging/Log.hpp"
namespace mo
{
	template<typename T> 
    TypedParameter<T>::TypedParameter(const std::string& name, const T& init, ParameterType type, mo::time_t ts, Context* ctx) :
		ITypedParameter<T>(name, type, ts, ctx), data(init),
        IParameter(name, mo::Control_e, ts, ctx)
	{
		(void)&_typed_parameter_constructor;
        (void)&_meta_parameter;
	}

	template<typename T> 
    T* TypedParameter<T>::GetDataPtr(mo::time_t ts, Context* ctx, boost::optional<size_t>* fn)
	{
        if (ts > 0 * mo::second)
        {
            if(ts != IParameter::_ts)
            {
                LOG(debug) << "Requested timestamp " << ts << " != " << this->_ts;
                return nullptr;
            }
        }
        if(fn)
            *fn = this->_fn;
		return &data;
	}

    template<typename T>
    T* TypedParameter<T>::GetDataPtr(size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
    {
        if (fn != std::numeric_limits<size_t>::max())
        {
            if(fn != this->_fn && this->_fn != std::numeric_limits<size_t>::max())
            {
                LOG(debug) << "Requested frame " << fn << " != " << this->_fn;
                return nullptr;
            }
        }
        if(ts)
            *ts = this->_ts;
        return &data;
    }

	template<typename T> 
    bool TypedParameter<T>::GetData(T& value, mo::time_t ts, Context* ctx, boost::optional<size_t>* fn)
	{
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if (ts < 0 * mo::second)
        {
            value = data;
            return true;
        }else
        {
            if(ts == IParameter::_ts)
            {
                value = data;
                if(fn)
                    *fn = this->_fn;
                return true;
            }
        }
        return false;
	}

    template<typename T>
    bool TypedParameter<T>::GetData(T& value, size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if (this->_fn == fn)
        {
            if(ts)
                *ts = this->_ts;
            value = data;
            return true;
        }else
        {
            if(this->_fn == std::numeric_limits<size_t>::max())
            {
                value = data;
                if(ts)
                    *ts = this->_ts;
                return true;
            }
        }
        return false;
    }

    template<typename T>
    T TypedParameter<T>::GetData(mo::time_t ts, Context* ctx, boost::optional<size_t>* fn)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ts < 0 * mo::second)
        {
            if(fn)
                *fn = this->_fn;
            return data;
        }else
        {
            ASSERT_EQ(ts, this->_ts) << " Timestamps do not match";
            if(fn)
                *fn = this->_fn;
            return data;
        }
    }
	
    template<typename T>
    T TypedParameter<T>::GetData(size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(this->_fn == std::numeric_limits<size_t>::max())
        {
            if(ts)
                *ts = this->_ts;
            return data;
        }else
        {
            ASSERT_EQ(fn, this->_fn) << " Frame numbers do not match";
            if(ts)
                *ts = this->_ts;
            return data;
        }
    }

    template<typename T>
    ITypedParameter<T>* TypedParameter<T>::UpdateData(const T& data_,
                                                      mo::time_t ts,
                                                      Context* ctx,
                                                      size_t fn,
                                                      ICoordinateSystem* cs)
	{
        data = data_;
        this->Commit(ts, ctx, fn, cs);
		return this;
	}
	
    template<typename T> 
    std::shared_ptr<IParameter> TypedParameter<T>::DeepCopy() const
	{
        return std::shared_ptr<IParameter>(new TypedParameter<T>(IParameter::GetName(), data));
	}

	template<typename T>  
    bool TypedParameter<T>::Update(IParameter* other, Context* ctx)
	{
		auto typed = dynamic_cast<ITypedParameter<T>*>(other);
		if (typed)
		{
			if (typed->GetData(data, -1, ctx))
			{
                IParameter::Commit(other->GetTimestamp(), ctx);
				return true;
			}
		}
		return false;
	}

	template<typename T> ParameterConstructor<TypedParameter<T>> TypedParameter<T>::_typed_parameter_constructor;
    template<typename T> MetaParameter<T, 100>  TypedParameter<T>::_meta_parameter;
}
