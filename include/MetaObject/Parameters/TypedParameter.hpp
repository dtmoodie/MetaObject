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
#include "ParameterFactory.hpp"

namespace mo
{
    template<typename T> class MO_EXPORTS TypedParameter : 
        public ITypedParameter<T>
    {
    public:
        TypedParameter(const std::string& name = "", const T& init = T(), ParameterType type = kControl, long lnog ts = -1, Context* ctx = nullptr);
        
		virtual T    GetData(long long ts= -1, Context* ctx = nullptr);
        virtual T*   GetDataPtr(long long ts= -1, Context* ctx = nullptr);
        virtual bool GetData(T& value, long long ts= -1, Context* ctx = nullptr);

        virtual ITypedParameter<T>* UpdateData(T& data_,       long long ts = -1, Context* ctx = nullptr);
        virtual ITypedParameter<T>* UpdateData(const T& data_, long long ts = -1, Context* ctx = nullptr);
        virtual ITypedParameter<T>* UpdateData(T* data_,       long long ts = -1, Context* ctx = nullptr);

        virtual IParameter* DeepCopy() const;
        virtual bool Update(IParameter* other);
        template<class Archive> void serialize(Archive& ar);
    protected:
        T data;
    private:
        static FactoryRegisterer<TypedParameter<T>, T, TypedParameter_c> _typed_parameter_constructor;
    };
}
#include "detail/TypedParameterImpl.hpp"