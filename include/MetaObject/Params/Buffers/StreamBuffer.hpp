#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Params/ITInputParam.hpp"
#include "Map.hpp"
#include "IBuffer.hpp"
#include <boost/thread/condition_variable.hpp>
namespace mo
{
    namespace Buffer
    {
        template<class T> class MO_EXPORTS StreamBuffer: public Map<T>
        {
        public:
            typedef T ValueType;
            static const ParamType Type = StreamBuffer_e;

            StreamBuffer(const std::string& name = "");

            T*   GetDataPtr(OptionalTime_t ts = OptionalTime_t(),
                                    Context* ctx = nullptr, size_t* fn_ = nullptr);
            T*   GetDataPtr(size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

            T    GetData(OptionalTime_t ts = OptionalTime_t(),
                                 Context* ctx = nullptr, size_t* fn = nullptr);
            T    GetData(size_t fn, Context* ctx = nullptr, OptionalTime_t* ts = nullptr);

            bool GetData(T& value, OptionalTime_t ts = OptionalTime_t(),
                                 Context* ctx = nullptr, size_t* fn = nullptr);
            bool GetData(T& value, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts = nullptr);


            virtual void SetFrameBufferCapacity(size_t size);
            virtual void SetTimePaddingCapacity(mo::Time_t time);
            virtual boost::optional<size_t> GetFrameBufferCapacity();
            virtual OptionalTime_t GetTimePaddingCapacity();

            std::shared_ptr<IParam> DeepCopy() const;
            virtual ParamType GetBufferType() const{ return StreamBuffer_e;}
        protected:
            //bool UpdateDataImpl(const T& data, OptionalTime_t ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs);
            virtual void prune();
            OptionalTime_t _current_timestamp;
            size_t _current_frame_number;
            OptionalTime_t _time_padding;
            boost::optional<size_t> _frame_padding;
        };

        template<class T> class MO_EXPORTS BlockingStreamBuffer : public StreamBuffer<T>
        {
        public:
            typedef T ValueType;
            static const ParamType Type = BlockingStreamBuffer_e;

            BlockingStreamBuffer(const std::string& name = "");

            virtual ParamType GetBufferType() const{ return BlockingStreamBuffer_e;}
        protected:
            bool UpdateDataImpl(const T& data, OptionalTime_t ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs);
            virtual void prune();
            long long _size;
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
