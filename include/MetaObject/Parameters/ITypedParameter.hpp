#pragma once
/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/parameters
*/
#include "IParameter.hpp"

namespace mo
{
    class TUpdateToken;
    template<typename T> class MO_EXPORTS ITypedParameter : virtual public IParameter
    {
    public:
        typedef std::shared_ptr<ITypedParameter<T>> Ptr;
        typedef T ValueType;
        
        ITypedParameter(const std::string& name, ParameterType flags = Control_e, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr);

        // The call is thread safe but the returned pointer may be modified by a different thread
        // ts is the timestamp for which you are requesting data, -1 indicates newest
        // ctx is the context of the data request, such as the thread of the object requesting the data
        virtual T*   GetDataPtr(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr) = 0;
		
        // Copies data into value
        // Time index is the index for which you are requesting data
        // ctx is the context of the data request, such as the thread of the object requesting the data
		virtual T    GetData(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr) = 0;
        virtual bool GetData(T& value, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr) = 0;
        
        // Update data, will call update_signal and set changed to true
        virtual ITypedParameter<T>* UpdateData(T& data_,       mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t fn = std::numeric_limits<size_t>::max()) = 0;
        virtual ITypedParameter<T>* UpdateData(const T& data_, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t fn = std::numeric_limits<size_t>::max()) = 0;
        virtual ITypedParameter<T>* UpdateData(T* data_,       mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t fn = std::numeric_limits<size_t>::max()) = 0;

        virtual TUpdateToken Update();

        virtual const TypeInfo& GetTypeInfo() const;

        virtual bool Update(IParameter* other);
    private:
        static const TypeInfo _type_info;
    };

    template<class T>
    class TUpdateToken
    {
    public:
        TUpdateToken(ITypedParameter<T>& param):
            _param(param),
            _ts(mo::time_t(-1 * mo::second)),
            _fn(std::numeric_limits<size_t>::max()),
            _cs(nullptr),
            _ctx(nullptr)
        {
        }

        ~TUpdateToken()
        {
            if(_cs)
                _param.SetCoordinateSystem(_cs);
            _param.UpdateData(_data, _ts, _ctx, _fn);
        }

        TUpdateToken& operator()(T&& data)
        {
            _data = std::forward<T>(data);
        }

        TUpdateToken& operator()(time_t&& ts)
        {
            _ts = ts;
            return *this;
        }

        TUpdateToken& operator()(size_t fn)
        {
            _fn = fn;
            return *this;
        }
        TUpdateToken& operator()(Context* ctx)
        {
            _ctx = ctx;
            return *this;
        }

        TUpdateToken& operator()(ICoordinateSystem* cs)
        {
            _cs = cs;
            return *this;
        }

    private:
        T& _data;
        ITypedParameter<T>& _param;
        size_t _fn;
        mo::time_t _ts;
        ICoordinateSystem* _cs;
        Context* _ctx;
        IParameter& _param;
    };
}
#include "detail/ITypedParameterImpl.hpp"
