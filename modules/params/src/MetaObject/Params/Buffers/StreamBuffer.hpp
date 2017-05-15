#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/ITInputParam.hpp"
#include "Map.hpp"
#include "IBuffer.hpp"
#include <boost/thread/condition_variable.hpp>
namespace mo
{
    namespace Buffer
    {
        template<class T> class MO_EXPORTS StreamBuffer: public Map<T>{
        public:
            typedef T ValueType;
            typedef typename ParamTraits<T>::Storage_t Storage_t;
            typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
            typedef typename ParamTraits<T>::InputStorage_t InputStorage_t;
            typedef typename ParamTraits<T>::Input_t Input_t;
            typedef void(TUpdateSig_t)(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags);
            typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
            typedef TSlot<TUpdateSig_t> TUpdateSlot_t;
            static const ParamType Type = StreamBuffer_e;

            StreamBuffer(const std::string& name = "");

            virtual bool getData(Storage_t& data, const OptionalTime_t& ts = OptionalTime_t(),
                Context* ctx = nullptr, size_t* fn_ = nullptr);

            virtual bool getData(Storage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

            virtual void setFrameBufferCapacity(size_t size);
            virtual void setTimePaddingCapacity(mo::Time_t time);
            virtual boost::optional<size_t> getFrameBufferCapacity();
            virtual OptionalTime_t getTimePaddingCapacity();

            virtual ParamType GetBufferType() const{ return StreamBuffer_e;}
        protected:
            virtual void prune();
            OptionalTime_t          _current_timestamp;
            size_t                  _current_frame_number;
            OptionalTime_t          _time_padding;
            boost::optional<size_t> _frame_padding;
        };

        template<class T> class MO_EXPORTS BlockingStreamBuffer : public StreamBuffer<T>{
        public:
            typedef T ValueType;
            static const ParamType Type = BlockingStreamBuffer_e;

            BlockingStreamBuffer(const std::string& name = "");

            virtual ParamType getBufferType() const{ return BlockingStreamBuffer_e;}
        protected:
            bool updateDataImpl(const T& data, OptionalTime_t ts, Context* ctx, size_t fn, ICoordinateSystem* cs);
            virtual void prune();
            size_t _size;
            boost::condition_variable_any _cv;
        };
    }

#define MO_METAParam_INSTANCE_SBUFFER_(N) \
    template<class T> struct MetaParam<T, N>: public MetaParam<T, N-1, void> \
    { \
        static BufferConstructor<Buffer::StreamBuffer<T>> _stream_buffer_constructor;  \
        static BufferConstructor<Buffer::BlockingStreamBuffer<T>> _blocking_stream_buffer_constructor;  \
        static ParamConstructor<Buffer::StreamBuffer<T>> _stream_buffer_Param_constructor; \
        static ParamConstructor<Buffer::BlockingStreamBuffer<T>> _blocking_stream_buffer_Param_constructor; \
        MetaParam<T, N>(const char* name): \
            MetaParam<T, N-1>(name) \
        { \
            (void)&_stream_buffer_constructor; \
            (void)&_stream_buffer_Param_constructor; \
            (void)&_blocking_stream_buffer_Param_constructor; \
            (void)&_blocking_stream_buffer_constructor; \
        } \
    }; \
    template<class T> BufferConstructor<Buffer::StreamBuffer<T>> MetaParam<T, N>::_stream_buffer_constructor; \
    template<class T> ParamConstructor<Buffer::StreamBuffer<T>> MetaParam<T, N>::_stream_buffer_Param_constructor; \
    template<class T> BufferConstructor<Buffer::BlockingStreamBuffer<T>> MetaParam<T, N>::_blocking_stream_buffer_constructor; \
    template<class T> ParamConstructor<Buffer::BlockingStreamBuffer<T>> MetaParam<T, N>::_blocking_stream_buffer_Param_constructor;
    MO_METAParam_INSTANCE_SBUFFER_(__COUNTER__)
}
#include "detail/StreamBufferImpl.hpp"
