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
#pragma once
#include "ITypedParameter.hpp"
#include "ParameterConstructor.hpp"
#include "MetaObject/Parameters/MetaParameter.hpp"
namespace mo
{
    template<typename T> 
    class MO_EXPORTS TypedParameter : public ITypedParameter<T>
    {
    public:
        typedef T ValueType;
        static const ParameterTypeFlags Type = TypedParameter_e;

        TypedParameter(const std::string& name = "",
                       const T& init = T(),
                       ParameterType type = Control_e,
                       mo::time_t ts = -1 * mo::second,
                       Context* ctx = nullptr);
        
        T*   GetDataPtr(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t* fn_ = nullptr);
        T*   GetDataPtr(size_t fn, Context* ctx = nullptr, mo::time_t* ts_ = nullptr);

        T    GetData(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t* fn = nullptr);
        T    GetData(size_t fn, Context* ctx = nullptr, mo::time_t* ts = nullptr);

        bool GetData(T& value, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t* fn = nullptr);
        bool GetData(T& value, size_t fn, Context* ctx = nullptr, mo::time_t* ts = nullptr);

        virtual ITypedParameter<T>* UpdateData(T&& data,
                                               mo::time_t ts = -1 * mo::second,
                                               Context* ctx = nullptr,
                                               size_t fn = std::numeric_limits<size_t>::max(),
                                               ICoordinateSystem* cs = nullptr);

        virtual std::shared_ptr<IParameter> DeepCopy() const;
        bool Update(IParameter* other, Context* ctx);
        
    protected:
        T data;
    private:
        static ParameterConstructor<TypedParameter<T>> _typed_parameter_constructor;
        static MetaParameter<T, 100> _meta_parameter;
    };
}
#include "detail/TypedParameterImpl.hpp"
