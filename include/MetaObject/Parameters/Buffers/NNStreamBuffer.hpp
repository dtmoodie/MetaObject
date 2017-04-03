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
            static const ParameterTypeFlags Type = NNStreamBuffer_e;

            NNStreamBuffer(const std::string& name = "");

            T*   GetDataPtr(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                            Context* ctx = nullptr, 
							size_t* fn_ = nullptr);

            T*   GetDataPtr(size_t fn, 
				            Context* ctx = nullptr, 
				            boost::optional<mo::time_t>* ts_ = nullptr);

            T    GetData(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                         Context* ctx = nullptr, 
				         size_t* fn = nullptr);

            T    GetData(size_t fn, 
				         Context* ctx = nullptr, 
				         boost::optional<mo::time_t>* ts = nullptr);

            bool GetData(T& value, 
				         boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                         Context* ctx = nullptr, 
				         size_t* fn = nullptr);

            bool GetData(T& value, 
				         size_t fn, 
				         Context* ctx = nullptr, 
				         boost::optional<mo::time_t>* ts = nullptr);

            virtual ParameterTypeFlags GetBufferType() const{ return NNStreamBuffer_e;}
        protected:
			typename std::map<SequenceKey, T>::iterator Search(boost::optional<mo::time_t> ts);
			typename std::map<SequenceKey, T>::iterator Search(size_t fn);
        };
    }

#define MO_METAPARAMETER_INSTANCE_NNBUFFER_(N) \
    template<class T> struct MetaParameter<T, N>: public MetaParameter<T, N-1, void> \
    { \
        static BufferConstructor<Buffer::NNStreamBuffer<T>> _nn_stream_buffer_constructor;  \
        static ParameterConstructor<Buffer::NNStreamBuffer<T>> _nn_stream_buffer_parameter_constructor; \
        MetaParameter<T, N>(const char* name): \
            MetaParameter<T, N-1>(name) \
        { \
            (void)&_nn_stream_buffer_constructor; \
            (void)&_nn_stream_buffer_parameter_constructor; \
        } \
    }; \
    template<class T> BufferConstructor<Buffer::NNStreamBuffer<T>> MetaParameter<T, N>::_nn_stream_buffer_constructor; \
    template<class T> ParameterConstructor<Buffer::NNStreamBuffer<T>> MetaParameter<T, N>::_nn_stream_buffer_parameter_constructor;

    MO_METAPARAMETER_INSTANCE_NNBUFFER_(__COUNTER__)
}
#include "detail/NNStreamBufferImpl.hpp"
