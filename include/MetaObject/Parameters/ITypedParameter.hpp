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

#include "Parameter.hpp"
namespace cv
{
    namespace cuda
    {
        class Stream;
    }
}
namespace Parameters
{
    template<typename T> class PARAMETER_EXPORTS ITypedParameter : public Parameter
    {
    public:
        typedef std::shared_ptr<ITypedParameter<T>> Ptr;
        
        ITypedParameter(const std::string& name, ParameterType flags = kControl);

        // The call is thread safe but the returned pointer may be modified by a different thread
        // Time index is the index for which you are requesting data
        // ctx is the context of the data request, such as the thread of the object requesting the data
        virtual T*   GetData(long long time_index = -1, Signals::context* ctx = nullptr) = 0;
        // Copies data into value
        // Time index is the index for which you are requesting data
        // ctx is the context of the data request, such as the thread of the object requesting the data
        virtual bool GetData(T& value, long long time_index = -1, Signals::context* ctx = nullptr) = 0;
        
        // Update data, will call update_signal and set changed to true
        virtual ITypedParameter<T>* UpdateData(T& data_,       long long time_index = -1, Signals::context* ctx = nullptr) = 0;
        virtual ITypedParameter<T>* UpdateData(const T& data_, long long time_index = -1, Signals::context* ctx = nullptr) = 0;
        virtual ITypedParameter<T>* UpdateData(T* data_,       long long time_index = -1, Signals::context* ctx = nullptr) = 0;

        virtual Loki::TypeInfo GetTypeInfo() const;

        virtual bool Update(Parameter* other, Signals::context* other_ctx = nullptr);

    };

    template<typename T> ITypedParameter<T>::ITypedParameter(const std::string& name, ParameterType flags) :
            Parameter(name, flags) 
    {
    }

    template<typename T> Loki::TypeInfo ITypedParameter<T>::GetTypeInfo() const
    {
        return Loki::TypeInfo(typeid(T));
    }

    template<typename T> bool ITypedParameter<T>::Update(Parameter* other, Signals::context* other_ctx)
    {
        auto typedParameter = dynamic_cast<ITypedParameter<T>*>(other);
        if (typedParameter)
        {
            std::lock_guard<std::recursive_mutex> lock(typedParameter->mtx());
            UpdateData(typedParameter->GetData(), other->GetTimeIndex(), other_ctx);
            OnUpdate(other_ctx);
            return true;
        }
        return false;
    }
}
