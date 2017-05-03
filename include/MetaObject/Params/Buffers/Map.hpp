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

#include "MetaObject/Params/ITInputParam.hpp"
#include "MetaObject/Params/ParamConstructor.hpp"
#include "MetaObject/Params/MetaParam.hpp"
#include "BufferConstructor.hpp"
#include "IBuffer.hpp"
#include <map>

namespace mo{
    class Context;
    namespace Buffer{
        struct SequenceKey{
            SequenceKey(OptionalTime_t ts, size_t fn):ts(ts), fn(fn){}
            SequenceKey(mo::Time_t ts):ts(ts), fn(0){}
            SequenceKey(size_t fn):fn(fn){}
            OptionalTime_t ts;
            size_t fn;
        };
        inline std::ostream& operator<<(std::ostream& os, const SequenceKey& key){
          if(key.ts)os << *key.ts << " ";
          if(key.fn != std::numeric_limits<size_t>::max()) os << key.fn;
          return os;
        }

        inline bool operator<(const SequenceKey& lhs, const SequenceKey& rhs){
            if(lhs.ts && rhs.ts) return *lhs.ts < *rhs.ts;
            else return lhs.fn < rhs.fn;
            
        }

        template<typename T>
        class Map: public ITInputParam<T>, public IBuffer{
        public:
            static const ParamType Type = Map_e;
            typedef T ValueType;

            Map(const std::string& name = "");

            virtual bool getData(Storage_t& data, const OptionalTime_t& ts = OptionalTime_t(),
                Context* ctx = nullptr, size_t* fn_ = nullptr);
            virtual bool getData(Storage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

            virtual void setFrameBufferCapacity(size_t size);
            virtual void setTimePaddingCapacity(mo::Time_t time);
            virtual boost::optional<size_t> getFrameBufferCapacity();
            virtual OptionalTime_t getTimePaddingCapacity();

            virtual size_t getSize();
            virtual bool getTimestampRange(mo::Time_t& start, mo::Time_t& end);
            virtual bool getFrameNumberRange(size_t& start, size_t& end);
            virtual ParamType getBufferType() const{ return Map_e;}
        protected:
            virtual bool updateDataImpl(ConstStorageRef_t data, OptionalTime_t ts, Context* ctx, size_t fn, ICoordinateSystem* cs);
            virtual void onInputUpdate(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags);
			typename std::map<SequenceKey, Storage_t>::iterator search(OptionalTime_t ts);
			typename std::map<SequenceKey, Storage_t>::iterator search(size_t fn);

            std::map<SequenceKey, Storage_t> _data_buffer;
        };
    }

#define MO_METAParam_INSTANCE_MAP_(N) \
    template<class T> struct MetaParam<T, N, void>: public MetaParam<T, N-1, void>{ \
        static ParamConstructor<Buffer::Map<T>> _map_param_constructor; \
        static BufferConstructor<Buffer::Map<T>> _map_constructor;  \
        MetaParam<T, N>(const char* name): \
            MetaParam<T, N-1>(name){ \
            (void)&_map_param_constructor; \
            (void)&_map_constructor; \
        } \
    }; \
    template<class T> ParamConstructor<Buffer::Map<T>> MetaParam<T, N, void>::_map_param_constructor; \
    template<class T> BufferConstructor<Buffer::Map<T>> MetaParam<T, N, void>::_map_constructor;

    MO_METAParam_INSTANCE_MAP_(__COUNTER__)
}
#include "detail/MapImpl.hpp"
