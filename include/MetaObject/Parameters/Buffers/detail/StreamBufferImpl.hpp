#pragma once

namespace mo
{
    namespace Buffer
    {
        template<class T> StreamBuffer<T>::StreamBuffer(const std::string& name):
            _current_timestamp(-1), _padding(5)
        {
        
        }

        template<class T> T*   StreamBuffer<T>::GetDataPtr(long long ts, Context* ctx)
        {
            T* result = Map<T>::GetDataPtr(ts, ctx);
            if(result && ts != -1)
            {
                _current_timestamp = ts;
                prune();
            }
            return result;
        }
        template<class T> bool StreamBuffer<T>::GetData(T& value, long long ts, Context* ctx)
        {
            if(Map<T>::GetData(value, ts, ctx))
            {
                _current_timestamp = ts;
                prune();
                return true;
            }
            return false;
        }
        template<class T> T StreamBuffer<T>::GetData(long long ts, Context* ctx)
        {
            T result = Map<T>::GetData(ts, ctx);
            _current_timestamp = ts;
            prune();
            return result;
        }
        template<class T> void StreamBuffer<T>::SetSize(long long size)
        {
            _padding = size;
        }
        template<class T> void StreamBuffer<T>::prune()
        {
            if(_current_timestamp != -1)
            {
                auto itr = _data_buffer.begin();
                while(itr != _data_buffer.end())
                {
                    if(itr->first < _current_timestamp - _padding)
                    {
                        itr = _data_buffer.erase(itr);
                    }else
                    {
                        break;
                    }
                }
            }
        }
        template<class T> std::shared_ptr<IParameter> StreamBuffer<T>::DeepCopy() const
        {
            return std::shared_ptr<IParameter>(new StreamBuffer<T>());
        }
    }
#define MO_METAPARAMETER_INSTANCE_STREAM_BUFFER_(N) \
    template<class T> struct MetaParameter<T, N>: public MetaParameter<T, N-1> \
    { \
        static ParameterConstructor<Buffer::StreamBuffer<T>, T, StreamBuffer_e> _stream_buffer_parameter_constructor; \
        static BufferConstructor<Buffer::StreamBuffer<T>, Buffer::BufferFactory::StreamBuffer> _stream_buffer_constructor;  \
        MetaParameter<T, N>(const char* name): \
            MetaParameter<T, N-1>(name) \
        { \
            (void)&_stream_buffer_parameter_constructor; \
            (void)&_stream_buffer_constructor; \
        } \
    }; \
    template<class T> ParameterConstructor<Buffer::StreamBuffer<T>, T, StreamBuffer_e> MetaParameter<T, N>::_stream_buffer_parameter_constructor; \
    template<class T> BufferConstructor<Buffer::StreamBuffer<T>, Buffer::BufferFactory::StreamBuffer> MetaParameter<T, N>::_stream_buffer_constructor;

    MO_METAPARAMETER_INSTANCE_STREAM_BUFFER_(__COUNTER__)
}
