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

#include "MetaObject/Parameters/ITypedParameter.hpp"
#include "MetaObject/Parameters/ParameterConstructor.hpp"
#include "MetaObject/Parameters/ITypedInputParameter.hpp"
#include "IBuffer.hpp"
#include "BufferConstructor.hpp"
#include <boost/circular_buffer.hpp>

namespace mo
{
    namespace Buffer
    {
        template<class T>
        class CircularBuffer: public IBuffer, public ITypedInputParameter<T>
        {
            static ParameterConstructor<CircularBuffer<T>> _circular_buffer_parameter_constructor;
            static BufferConstructor<CircularBuffer<T>> _circular_buffer_constructor;
            boost::circular_buffer<State<T>> _data_buffer;
        public:
            typedef T ValueType;
            static const ParameterTypeFlags Type = cbuffer_e;

            CircularBuffer(T&& init, const std::string& name = "",
                mo::time_t ts = -1 * mo::second,
                ParameterType type = Buffer_e);

            CircularBuffer(const std::string& name = "",
                mo::time_t ts = -1 * mo::second,
                ParameterType type = Buffer_e);

            T*   GetDataPtr(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t* fn_ = nullptr);
            T*   GetDataPtr(size_t fn, Context* ctx = nullptr, mo::time_t* ts_ = nullptr);

            T    GetData(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t* fn = nullptr);
            T    GetData(size_t fn, Context* ctx = nullptr, mo::time_t* ts = nullptr);

            bool GetData(T& value, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t* fn = nullptr);
            bool GetData(T& value, size_t fn, Context* ctx = nullptr, mo::time_t* ts = nullptr);

            ITypedParameter<T>* UpdateData(T&& data,
                                           mo::time_t ts = -1 * mo::second,
                                           Context* ctx = nullptr,
                                           size_t fn = std::numeric_limits<size_t>::max(),
                                           ICoordinateSystem* cs = nullptr);
    
            bool Update(IParameter* other, Context* ctx = nullptr);
            std::shared_ptr<IParameter> DeepCopy() const;

            virtual void SetSize(long long size);
            virtual long long GetSize();
            virtual void GetTimestampRange(mo::time_t& start, mo::time_t& end);
            
            void onInputUpdate(Context* ctx, IParameter* param);
            virtual ParameterTypeFlags GetBufferType() const{ return cbuffer_e;}
        };
    }
    
    #define MO_METAPARAMETER_INSTANCE_CBUFFER_(N) \
    template<class T> struct MetaParameter<T, N>: public MetaParameter<T, N-1, void> \
    { \
        static ParameterConstructor<Buffer::CircularBuffer<T>> _circular_buffer_parameter_constructor; \
        static BufferConstructor<Buffer::CircularBuffer<T>> _circular_buffer_constructor;  \
        MetaParameter<T, N>(const char* name): \
            MetaParameter<T, N-1>(name) \
        { \
            (void)&_circular_buffer_constructor; \
            (void)&_circular_buffer_parameter_constructor; \
        } \
    }; \
    template<class T> ParameterConstructor<Buffer::CircularBuffer<T>> MetaParameter<T, N>::_circular_buffer_parameter_constructor; \
    template<class T> BufferConstructor<Buffer::CircularBuffer<T>> MetaParameter<T, N>::_circular_buffer_constructor;
    
    MO_METAPARAMETER_INSTANCE_CBUFFER_(__COUNTER__)
}
#include "detail/CircularBufferImpl.hpp"
