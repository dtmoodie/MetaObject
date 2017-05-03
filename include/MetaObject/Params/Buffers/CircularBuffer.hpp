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

https://github.com/dtmoodie/Params
*/
#pragma once

#include "MetaObject/Params/ITParam.hpp"
#include "MetaObject/Params/ParamConstructor.hpp"
#include "MetaObject/Params/ITInputParam.hpp"
#include "IBuffer.hpp"
#include "BufferConstructor.hpp"
#include <boost/circular_buffer.hpp>

namespace mo
{


    namespace Buffer
    {
        template<class T>
        class CircularBuffer: public IBuffer, public ITInputParam<T>{
            static ParamConstructor<CircularBuffer<T>> _circular_buffer_param_constructor;
            static BufferConstructor<CircularBuffer<T>> _circular_buffer_constructor;
            boost::circular_buffer<State<Storage_t>> _data_buffer;
        public:
            typedef T ValueType;
            static const ParamType Type = CircularBuffer_e;

            CircularBuffer(T&& init, const std::string& name = "",
                OptionalTime_t ts = {},
                ParamFlags type = Buffer_e);

            CircularBuffer(const std::string& name = "",
                OptionalTime_t ts = {},
                ParamFlags type = Buffer_e);

            virtual bool getData(Storage_t& data, const OptionalTime_t& ts = OptionalTime_t(),
                Context* ctx = nullptr, size_t* fn_ = nullptr);

            virtual bool getData(Storage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

            virtual void setFrameBufferCapacity(size_t size);
            virtual void setTimePaddingCapacity(mo::Time_t time);
            virtual boost::optional<size_t> getFrameBufferCapacity();
            virtual OptionalTime_t getTimePaddingCapacity();

            virtual size_t getSize();
            bool getTimestampRange(mo::Time_t& start, mo::Time_t& end);
            bool getFrameNumberRange(size_t& start,size_t& end);
            
            void onInputUpdate(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags);
            virtual ParamType getBufferType() const{ return CircularBuffer_e;}
        protected:
            bool updateDataImpl(ConstStorageRef_t data, OptionalTime_t ts, Context* ctx, size_t fn, ICoordinateSystem* cs);
        };
    }
    
    #define MO_METAParam_INSTANCE_CBUFFER_(N) \
    template<class T> struct MetaParam<T, N>: public MetaParam<T, N-1, void> \
    { \
        static ParamConstructor<Buffer::CircularBuffer<T>> _circular_buffer_param_constructor; \
        static BufferConstructor<Buffer::CircularBuffer<T>> _circular_buffer_constructor;  \
        MetaParam<T, N>(const char* name): \
            MetaParam<T, N-1>(name) \
        { \
            (void)&_circular_buffer_constructor; \
            (void)&_circular_buffer_param_constructor; \
        } \
    }; \
    template<class T> ParamConstructor<Buffer::CircularBuffer<T>> MetaParam<T, N>::_circular_buffer_param_constructor; \
    template<class T> BufferConstructor<Buffer::CircularBuffer<T>> MetaParam<T, N>::_circular_buffer_constructor;
    
    MO_METAParam_INSTANCE_CBUFFER_(__COUNTER__)
}
#include "detail/CircularBufferImpl.hpp"
