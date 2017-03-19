#pragma once
#ifndef __CUDACC__
#include "MetaObject/Logging/Log.hpp"
#include <boost/thread/recursive_mutex.hpp>
namespace mo
{
	template<typename T> class TypedParameterPtr;
    template<typename T, int N, typename Enable> struct MetaParameter;
    

    template<typename T>
    TypedParameterPtr<T>::TypedParameterPtr(const std::string& name,
                                            T* ptr_,
                                            ParameterType type,
                                            bool ownsData_) :
		ptr(ptr_), ownsData(ownsData_), ITypedParameter<T>(name, type)
	{
        (void)&_meta_parameter;
	}
	
    template<typename T> 
    TypedParameterPtr<T>::~TypedParameterPtr()
	{
		if (ownsData && ptr)
			delete ptr;
	}

    template<typename T> 
    T* TypedParameterPtr<T>::GetDataPtr(boost::optional<mo::time_t> ts, Context* ctx, size_t* fn_)
	{
        if(ts)
        {
			if (ts == this->_ts)
			{
				return ptr;
			}
			else
			{
				if (this->_ts)
				{
					LOG(trace) << "Requested timestamp != current [" << ts
							   << " != " << this->_fn
							   << "] for parameter " << this->GetTreeName();
					return nullptr;
				}
				else
				{
					if (this->CheckFlags(mo::Unstamped_e))
					{
						return ptr;
					}
					else
					{
						return nullptr;
					}
				}
			}
        }
        if(fn_)
        {
            *fn_ = IParameter::_fn;
        }
		return ptr;
	}

    template<typename T>
    T* TypedParameterPtr<T>::GetDataPtr(size_t fn, Context* ctx, boost::optional<mo::time_t>* ts_)
    {
        if(fn != std::numeric_limits<size_t>::max())
        {
            if(fn != this->_fn &&
               this->_fn != std::numeric_limits<size_t>::max())
            {
                LOG(trace) << "Requested frame != current [" << fn
                           << " != " << this->_fn
                           << "] for parameter " << this->GetTreeName();
                return nullptr;
            }else if(this->_fn == std::numeric_limits<size_t>::max())
            {
                return ptr;
            }
        }
        if(ts_)
        {
            *ts_ = IParameter::_ts;
        }
        return ptr;
    }
	
    template<typename T> 
    T TypedParameterPtr<T>::GetData(boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
	{
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ts)
        {
            if(ts != IParameter::_ts && this->_ts)
            {
                THROW(debug) << "Requested timestamp != current ["
                             << ts << " != " << IParameter::_ts
                             << "] for parameter " << this->GetTreeName();
            }
        }
        if(ptr == nullptr)
            THROW(debug) << "Data pointer not set";
        if(fn)
            *fn = this->_fn;
		return *ptr;
	}
	
    template<typename T>
    T TypedParameterPtr<T>::GetData(size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(fn != std::numeric_limits<size_t>::max())
        {
            if(fn != IParameter::_fn && IParameter::_fn != std::numeric_limits<size_t>::max())
            {
                THROW(debug) << "Requested frame != current ["
                             << fn << " != " << IParameter::_fn
                             << "] for parameter " << this->GetTreeName();
            }
        }
        if(ptr == nullptr)
            THROW(debug) << "Data pointer not set";
        if(ts)
            *ts = IParameter::_ts;
        return *ptr;
    }

    template<typename T>
    bool TypedParameterPtr<T>::GetData(T& value, boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
	{
		boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ts)
        {
            if(ts != this->_ts)
            {
                LOG(trace) << "Requested timestamp != current ["
                           << ts << " != " << IParameter::_ts
                           << "] for parameter " << this->GetTreeName();
                return false;
            }
        }
		
		if (ptr)
		{
			value = *ptr;
            if(fn)
                *fn = this->_fn;
			return true;
		}
		return false;
	}
	
    template<typename T>
    bool TypedParameterPtr<T>::GetData(T& value, size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(fn != std::numeric_limits<size_t>::max())
        {
            if(fn != this->_fn)
            {
                LOG(trace) << "Requested frame != current ["
                           << fn << " != " << this->_fn
                           << "] for parameter " << this->GetTreeName();
                return false;
            }
        }

        if (ptr)
        {
            value = *ptr;
            if(ts)
                *ts = this->_ts;
            return true;
        }
        return false;
    }

    /*template<typename T>
    ITypedParameter<T>* TypedParameterPtr<T>::UpdateData(const T& data_, boost::optional<mo::time_t> ts,
                                                         Context* ctx, size_t fn, ICoordinateSystem* cs)
	{
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ptr)
        {
            *ptr = data_;
            this->Commit(ts, ctx, fn, cs);
        }
        return this;
    }*/
    template<typename T>
    bool TypedParameterPtr<T>::UpdateDataImpl(const T& data, boost::optional<mo::time_t> ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ptr)
        {
            *ptr = data;
            this->Commit(ts, ctx, fn, cs);
        }
        return this;
    }

    template<typename T> 
    bool TypedParameterPtr<T>::Update(IParameter* other)
	{
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
		auto typed = dynamic_cast<ITypedParameter<T>*>(other);
		if (typed)
		{
			*ptr = typed->GetData();
            this->Commit(other->GetTimestamp(),
                         other->GetContext(),
                         other->GetFrameNumber(),
                         other->GetCoordinateSystem());
			return true;
		}
		return false;
	}
	
    template<typename T> 
    std::shared_ptr<IParameter> TypedParameterPtr<T>::DeepCopy() const
	{
		return std::shared_ptr<IParameter>(new TypedParameterPtr<T>(IParameter::GetName(), ptr));
	}

    template<typename T>
    ITypedParameter<T>* TypedParameterPtr<T>::UpdatePtr(T* ptr, bool ownsData_)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        this->ptr = ptr;
        this->ownsData = ownsData_;
        return this;
    }
    
    template<typename T>
    MetaParameter<T, 100, void> TypedParameterPtr<T>::_meta_parameter;
}
#endif
