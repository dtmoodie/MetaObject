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

#include "MetaObject/Parameters/ITypedInputParameter.hpp"
#include "MetaObject/Parameters/ParameterConstructor.hpp"
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "BufferConstructor.hpp"
#include "IBuffer.hpp"
#include <map>

namespace mo
{
    class Context;
    namespace Buffer
    {
        struct SequenceKey
        {
            SequenceKey(mo::time_t ts, size_t fn):
                ts(ts), fn(fn){}
            SequenceKey(mo::time_t ts):
                ts(ts), fn(std::numeric_limits<size_t>::max()){}
            SequenceKey(size_t fn):
                fn(fn){}
            boost::optional<mo::time_t> ts;
            size_t fn;
        };
        inline std::ostream& operator<<(std::ostream& os, const SequenceKey& key)
        {
          if(key.ts)
              os << *key.ts << " ";
          if(key.fn != std::numeric_limits<size_t>::max())
              os << key.fn;
          return os;
        }

        inline bool operator<(const SequenceKey& lhs, const SequenceKey& rhs)
        {

            if(lhs.ts && rhs.ts)
            {
                return lhs.ts < rhs.ts;
            }else
            {
                return lhs.fn < rhs.fn;
            }
        }

        template<typename T>
        class Map: public ITypedInputParameter<T>, public IBuffer
        {
        public:
            typedef T ValueType;
            static const ParameterTypeFlags Type = map_e;

            Map(const std::string& name = "");

            T*   GetDataPtr(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                                    Context* ctx = nullptr, size_t* fn_ = nullptr);
            T*   GetDataPtr(size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts_ = nullptr);

            T    GetData(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                                 Context* ctx = nullptr, size_t* fn = nullptr);
            T    GetData(size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts = nullptr);

            bool GetData(T& value, boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                                 Context* ctx = nullptr, size_t* fn = nullptr);
            bool GetData(T& value, size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts = nullptr);

            /*ITypedParameter<T>* UpdateData(const T& data,
                                           mo::time_t ts = -1 * mo::second,
                                           Context* ctx = nullptr,
                                           size_t fn = std::numeric_limits<size_t>::max(),
                                           ICoordinateSystem* cs = nullptr);*/

            bool Update(IParameter* other, Context* ctx = nullptr);
            std::shared_ptr<IParameter> DeepCopy() const;

            void SetSize(size_t size);
            void SetSize(mo::time_t size);
            size_t GetSize();
            bool GetTimestampRange(mo::time_t& start, mo::time_t& end);
            bool GetFrameNumberRange(size_t& start, size_t& end);
            virtual ParameterTypeFlags GetBufferType() const{ return map_e;}
        protected:
            bool UpdateDataImpl(const T& data, boost::optional<mo::time_t> ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs);
            std::map<SequenceKey, T> _data_buffer;
            virtual void onInputUpdate(Context* ctx, IParameter* param);
        };
    }

#define MO_METAPARAMETER_INSTANCE_MAP_(N) \
    template<class T> struct MetaParameter<T, N, void>: public MetaParameter<T, N-1, void> \
    { \
        static ParameterConstructor<Buffer::Map<T>> _map_parameter_constructor; \
        static BufferConstructor<Buffer::Map<T>> _map_constructor;  \
        MetaParameter<T, N>(const char* name): \
            MetaParameter<T, N-1>(name) \
        { \
            (void)&_map_parameter_constructor; \
            (void)&_map_constructor; \
        } \
    }; \
    template<class T> ParameterConstructor<Buffer::Map<T>> MetaParameter<T, N, void>::_map_parameter_constructor; \
    template<class T> BufferConstructor<Buffer::Map<T>> MetaParameter<T, N, void>::_map_constructor;

    MO_METAPARAMETER_INSTANCE_MAP_(__COUNTER__)
}
#include "detail/MapImpl.hpp"
