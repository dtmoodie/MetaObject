#pragma once

#include "StreamBuffer.hpp"

namespace mo
{
    namespace Buffer
    {
        template<class T> class MO_EXPORTS NNStreamBuffer: public StreamBuffer<T>
        {
        public:
            typedef T ValueType;
            static const ParamType Type = NNStreamBuffer_e;

            NNStreamBuffer(const std::string& name = "");

            T*   GetDataPtr(OptionalTime_t ts = OptionalTime_t(),
                            Context* ctx = nullptr, 
							size_t* fn_ = nullptr);

            T*   GetDataPtr(size_t fn, 
				            Context* ctx = nullptr, 
				            OptionalTime_t* ts_ = nullptr);

            T    GetData(OptionalTime_t ts = OptionalTime_t(),
                         Context* ctx = nullptr, 
				         size_t* fn = nullptr);

            T    GetData(size_t fn, 
				         Context* ctx = nullptr, 
				         OptionalTime_t* ts = nullptr);

            bool GetData(T& value, 
				         OptionalTime_t ts = OptionalTime_t(),
                         Context* ctx = nullptr, 
				         size_t* fn = nullptr);

            bool GetData(T& value, 
				         size_t fn, 
				         Context* ctx = nullptr, 
				         OptionalTime_t* ts = nullptr);

            virtual ParamType GetBufferType() const{ return NNStreamBuffer_e;}
        protected:
			typename std::map<SequenceKey, T>::iterator Search(OptionalTime_t ts);
			typename std::map<SequenceKey, T>::iterator Search(size_t fn);
        };
    }

#define MO_METAParam_INSTANCE_NNBUFFER_(N) \
    template<class T> struct MetaParam<T, N>: public MetaParam<T, N-1, void> \
    { \
        static BufferConstructor<Buffer::NNStreamBuffer<T>> _nn_stream_buffer_constructor;  \
        static ParamConstructor<Buffer::NNStreamBuffer<T>> _nn_stream_buffer_Param_constructor; \
        MetaParam<T, N>(const char* name): \
            MetaParam<T, N-1>(name) \
        { \
            (void)&_nn_stream_buffer_constructor; \
            (void)&_nn_stream_buffer_Param_constructor; \
        } \
    }; \
    template<class T> BufferConstructor<Buffer::NNStreamBuffer<T>> MetaParam<T, N>::_nn_stream_buffer_constructor; \
    template<class T> ParamConstructor<Buffer::NNStreamBuffer<T>> MetaParam<T, N>::_nn_stream_buffer_Param_constructor;

    MO_METAParam_INSTANCE_NNBUFFER_(__COUNTER__)
}
#include "detail/NNStreamBufferImpl.hpp"
