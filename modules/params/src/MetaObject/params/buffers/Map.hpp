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

https://github.com/dtmoodie/MetaObject
*/
#pragma once

#include "BufferConstructor.hpp"
#include "IBuffer.hpp"
#include "MetaObject/params/ITInputParam.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/ParamConstructor.hpp"
#include <map>

namespace mo
{
    class Context;
    namespace Buffer
    {
        struct SequenceKey
        {
            SequenceKey(OptionalTime_t ts,
                        size_t fn = 0,
                        const std::shared_ptr<ICoordinateSystem>& cs_ = nullptr,
                        Context* ctx_ = nullptr)
                : ts(ts), fn(fn), cs(cs_), ctx(ctx_)
            {
            }
            SequenceKey(mo::Time_t ts_) : ts(ts_), fn(0), cs(nullptr), ctx(nullptr) {}
            SequenceKey(size_t fn) : fn(fn) {}
            OptionalTime_t ts;
            size_t fn;
            std::shared_ptr<ICoordinateSystem> cs;
            Context* ctx;
        };

        inline std::ostream& operator<<(std::ostream& os, const SequenceKey& key)
        {
            if (key.ts)
            {
                os << *key.ts << " ";
            }
            else
            {
            }
            if (key.fn != std::numeric_limits<size_t>::max())
            {
                os << key.fn;
            }
            else
            {
            }

            return os;
        }

        inline bool operator<(const SequenceKey& lhs, const SequenceKey& rhs)
        {
            if (lhs.ts && rhs.ts)
            {
                return *lhs.ts < *rhs.ts;
            }
            else
            {
                return lhs.fn < rhs.fn;
            }
        }

        template <typename T>
        class Map : public ITInputParam<T>, public IBuffer
        {
          public:
            static const ParamType Type = Map_e;
            typedef T ValueType;
            typedef typename ParamTraits<T>::Storage_t Storage_t;
            typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
            typedef typename ParamTraits<T>::InputStorage_t InputStorage_t;
            typedef typename ParamTraits<T>::Input_t Input_t;
            typedef void(TUpdateSig_t)(ConstStorageRef_t,
                                       IParam*,
                                       Context*,
                                       OptionalTime_t,
                                       size_t,
                                       const std::shared_ptr<ICoordinateSystem>&,
                                       UpdateFlags);
            typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
            typedef TSlot<TUpdateSig_t> TUpdateSlot_t;

            Map(const std::string& name = "");

            virtual bool getData(InputStorage_t& data,
                                 const OptionalTime_t& ts = OptionalTime_t(),
                                 Context* ctx = nullptr,
                                 size_t* fn_ = nullptr);

            virtual bool
            getData(InputStorage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

            virtual void setFrameBufferCapacity(size_t size);
            virtual void setTimePaddingCapacity(mo::Time_t time);
            virtual boost::optional<size_t> getFrameBufferCapacity();
            virtual OptionalTime_t getTimePaddingCapacity();

            virtual size_t getSize();
            virtual bool getTimestampRange(mo::Time_t& start, mo::Time_t& end);
            virtual bool getFrameNumberRange(size_t& start, size_t& end);
            virtual ParamType getBufferType() const { return Map_e; }

          protected:
            bool updateDataImpl(const Storage_t& data,
                                const OptionalTime_t& ts,
                                Context* ctx,
                                size_t fn,
                                const std::shared_ptr<ICoordinateSystem>& cs) override;

            bool updateDataImpl(Storage_t&& data,
                                const OptionalTime_t& ts,
                                Context* ctx,
                                size_t fn,
                                const std::shared_ptr<ICoordinateSystem>& cs) override;

            void onInputUpdate(ConstStorageRef_t,
                               IParam*,
                               Context*,
                               OptionalTime_t,
                               size_t,
                               const std::shared_ptr<ICoordinateSystem>&,
                               UpdateFlags) override;
            typename std::map<SequenceKey, InputStorage_t>::iterator search(const OptionalTime_t& ts);
            typename std::map<SequenceKey, InputStorage_t>::iterator search(size_t fn);

            std::map<SequenceKey, InputStorage_t> _data_buffer;
        };
    }

#define MO_METAPARAM_INSTANCE_MAP_(N)                                                                                  \
    template <class T>                                                                                                 \
    struct MetaParam<T, N, void> : public MetaParam<T, N - 1, void>                                                    \
    {                                                                                                                  \
        static ParamConstructor<Buffer::Map<T>> _map_param_constructor;                                                \
        static BufferConstructor<Buffer::Map<T>> _map_constructor;                                                     \
        MetaParam<T, N>(SystemTable * table, const char* name) : MetaParam<T, N - 1>(table, name)                      \
        {                                                                                                              \
            (void)&_map_param_constructor;                                                                             \
            (void)&_map_constructor;                                                                                   \
        }                                                                                                              \
    };                                                                                                                 \
    template <class T>                                                                                                 \
    ParamConstructor<Buffer::Map<T>> MetaParam<T, N, void>::_map_param_constructor;                                    \
    template <class T>                                                                                                 \
    BufferConstructor<Buffer::Map<T>> MetaParam<T, N, void>::_map_constructor;

    MO_METAPARAM_INSTANCE_MAP_(__COUNTER__)
}
#include "detail/MapImpl.hpp"
