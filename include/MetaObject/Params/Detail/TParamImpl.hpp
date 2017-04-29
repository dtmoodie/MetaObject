#pragma once
#include "MetaObject/Logging/Log.hpp"
namespace mo
{
	template<typename T> 
    TParam<T>::TParam() :
		ITParam<T>(), _data(),
        IParam()
	{
		(void)&_typed_param_constructor;
        (void)&_meta_param;
	}

	template<typename T> 
    bool TParam<T>::getData(Storage_t& value, OptionalTime_t ts, Context* ctx, size_t* fn)
	{
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if (!ts)
        {
            value = data;
            return true;
        }else
        {
            if(ts == IParam::_ts)
            {
                value = _data;
                if(fn)
                    *fn = this->_fn;
                return true;
            }
        }
        return false;
	}

    template<typename T>
    bool TParam<T>::getData(Storage_t& value, size_t fn, Context* ctx, OptionalTime_t* ts)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if (this->_fn == fn)
        {
            if(ts)
                *ts = this->_ts;
            value = _data;
            return true;
        }
        return false;
    }

    template<typename T>
    bool TParam<T>::updateDataImpl(ConstStorageRef_t data, OptionalTime_t ts, Context* ctx, size_t fn, ICoordinateSystem* cs){
        _data = data;
        this->_fn = fn;
        this->_ts = ts;
        _typed_update_signal(data, this, ctx, ts, this->_fn, cs, ValueUpdated_e);
        return true;
	}
	
	template<typename T> ParamConstructor<TParam<T>> TParam<T>::_typed_param_constructor;
    template<typename T> MetaParam<T, 100>  TParam<T>::_meta_param;
}
