#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Parameters/ITypedInputParameter.hpp"
#include "map.hpp"
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
            static const ParameterTypeFlags Type = StreamBuffer_e;

            StreamBuffer(const std::string& name = "");

            T*   GetDataPtr(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                                    Context* ctx = nullptr, size_t* fn_ = nullptr);
            T*   GetDataPtr(size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts_ = nullptr);

            T    GetData(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                                 Context* ctx = nullptr, size_t* fn = nullptr);
            T    GetData(size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts = nullptr);

            bool GetData(T& value, boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                                 Context* ctx = nullptr, size_t* fn = nullptr);
            bool GetData(T& value, size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts = nullptr);


            void SetFrameBufferSize(size_t size);
            void SetTimestampSize(mo::time_t size);
            std::shared_ptr<IParameter> DeepCopy() const;
            virtual ParameterTypeFlags GetBufferType() const{ return StreamBuffer_e;}
        protected:
            //bool UpdateDataImpl(const T& data, boost::optional<mo::time_t> ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs);
            virtual void prune();
            boost::optional<mo::time_t> _current_timestamp;
            size_t _current_frame_number;
            boost::optional<mo::time_t> _time_padding;
            boost::optional<size_t> _frame_padding;
        };

        template<class T> class MO_EXPORTS BlockingStreamBuffer : public StreamBuffer<T>
        {
        public:
            typedef T ValueType;
            static const ParameterTypeFlags Type = BlockingStreamBuffer_e;

            BlockingStreamBuffer(const std::string& name = "");

            virtual ParameterTypeFlags GetBufferType() const{ return BlockingStreamBuffer_e;}
        protected:
            bool UpdateDataImpl(const T& data, boost::optional<mo::time_t> ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs);
            virtual void prune();
            long long _size;
            boost::condition_variable_any _cv;
        };
    }

#define MO_METAPARAMETER_INSTANCE_SBUFFER_(N) \
    template<class T> struct MetaParameter<T, N>: public MetaParameter<T, N-1, void> \
    { \
        static BufferConstructor<Buffer::StreamBuffer<T>> _stream_buffer_constructor;  \
        static BufferConstructor<Buffer::BlockingStreamBuffer<T>> _blocking_stream_buffer_constructor;  \
        static ParameterConstructor<Buffer::StreamBuffer<T>> _stream_buffer_parameter_constructor; \
        static ParameterConstructor<Buffer::BlockingStreamBuffer<T>> _blocking_stream_buffer_parameter_constructor; \
        MetaParameter<T, N>(const char* name): \
            MetaParameter<T, N-1>(name) \
        { \
            (void)&_stream_buffer_constructor; \
            (void)&_stream_buffer_parameter_constructor; \
            (void)&_blocking_stream_buffer_parameter_constructor; \
            (void)&_blocking_stream_buffer_constructor; \
        } \
    }; \
    template<class T> BufferConstructor<Buffer::StreamBuffer<T>> MetaParameter<T, N>::_stream_buffer_constructor; \
    template<class T> ParameterConstructor<Buffer::StreamBuffer<T>> MetaParameter<T, N>::_stream_buffer_parameter_constructor; \
    template<class T> BufferConstructor<Buffer::BlockingStreamBuffer<T>> MetaParameter<T, N>::_blocking_stream_buffer_constructor; \
    template<class T> ParameterConstructor<Buffer::BlockingStreamBuffer<T>> MetaParameter<T, N>::_blocking_stream_buffer_parameter_constructor;
    MO_METAPARAMETER_INSTANCE_SBUFFER_(__COUNTER__)
}
#include "detail/StreamBufferImpl.hpp"
