#pragma once

namespace mo
{
    namespace Buffer
    {
        template<class T>
        CircularBuffer<T>::CircularBuffer(T&& init, const std::string& name, mo::time_t ts, ParameterType type ):
            ITypedInputParameter<T>(name),
            ITypedParameter<T>(name, mo::Buffer_e)
        {
            (void)&_circular_buffer_constructor;
            (void)&_circular_buffer_parameter_constructor;
            _data_buffer.set_capacity(10);
            _data_buffer.push_back(State<T>(ts, init));
        }

        template<class T>
        CircularBuffer<T>::CircularBuffer(const std::string& name, mo::time_t ts, ParameterType type ):
            ITypedInputParameter<T>(name),
            ITypedParameter<T>(name, mo::Buffer_e)
        {
            _data_buffer.set_capacity(10);
        }

        template<class T>
        T* CircularBuffer<T>::GetDataPtr(mo::time_t ts, Context* ctx, size_t* fn)
        {
            if (ts < 0 * mo::second && _data_buffer.size())
                return &_data_buffer.back().second;

            for (auto& itr : _data_buffer)
            {
                if (itr.ts == ts)
                {
                    if(fn)
                        *fn = itr.fn;
                    return &itr;
                }
            }
            return nullptr;
        }

        template<class T>
        T* CircularBuffer<T>::GetDataPtr(size_t fn, Context* ctx, mo::time_t* ts)
        {
            for (auto& itr : _data_buffer)
            {
                if (itr.fn == fn)
                {
                    if(ts)
                        *ts = itr.ts;
                    return &itr;
                }
            }
            return nullptr;
        }
         
        template<class T>
        bool CircularBuffer<T>::GetData(T& value, mo::time_t ts, Context* ctx, size_t* fn)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            if (ts < 0 * mo::second  && _data_buffer.size())
            {
                value = _data_buffer.back().second;
                return true;
            }
            for (auto& itr : _data_buffer)
            {
                if (itr.ts == ts)
                {
                    value = itr;
                    if(fn)
                        *fn = itr.fn;
                    return true;
                }
            }
            return false;
        }

        template<class T>
        T CircularBuffer<T>::GetData(mo::time_t ts, Context* ctx, size_t* fn)
        {
            if (ts < 0 * mo::second && _data_buffer.size())
            {
                if(fn)
                    *fn = _data_buffer.back().fn;
                return _data_buffer.back();
            }
            for (auto& itr : _data_buffer)
            {
                if (itr == ts)
                {
                    if(fn)
                        *fn = itr.fn;
                    return itr;
                }
            }
            THROW(debug) << "Could not find timestamp " << ts << " in range (" << _data_buffer.back().first << "," << _data_buffer.front().first <<")";
            return T();
        }
            

        template<class T>
        ITypedParameter<T>* CircularBuffer<T>::UpdateData(T&& data_, mo::time_t ts, Context* ctx, size_t fn, ICoordinateSystem* cs)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            _data_buffer.push_back(State<T>(ts, fn, ctx, cs, data_));
            this->modified = true;
            this->Commit(ts, ctx, fn, cs);
            return this;
        }

        template<class T> bool CircularBuffer<T>::Update(IParameter* other, Context* ctx)
        {
            auto typedParameter = dynamic_cast<ITypedParameter<T>*>(other);
            if (typedParameter)
            {
                T data;
                if (typedParameter->GetData(data))
                {
                    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                    _data_buffer.push_back(std::pair<mo::time_t, T>(typedParameter->GetTimestamp(), data));
                    IParameter::modified = true;
                    IParameter::OnUpdate(ctx);
                }
            }
            return false;
        }

        template<class T> void  CircularBuffer<T>::SetSize(long long size)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            _data_buffer.set_capacity(size);
        }
        template<class T> long long CircularBuffer<T>::GetSize()
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            return _data_buffer.capacity();
        }
        template<class T> void  CircularBuffer<T>::GetTimestampRange(mo::time_t& start, mo::time_t& end) 
        {
            if (_data_buffer.size())
            {
                boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                start = _data_buffer.back().first;
                end = _data_buffer.front().first;
            }
        }
        template<class T> std::shared_ptr<IParameter>  CircularBuffer<T>::DeepCopy() const
        {
            auto buffer = new CircularBuffer<T>(IParameter::_name);
            buffer->SetInput(this->input);
            return std::shared_ptr<IParameter>(buffer);
        }

        template<class T> void  CircularBuffer<T>::onInputUpdate(Context* ctx, IParameter* param)
        {
            if(this->input)
            {
                UpdateData(this->input->GetDataPtr(), this->input->GetTimestamp(), ctx);
            }
        }
        template<typename T> ParameterConstructor<CircularBuffer<T>> CircularBuffer<T>::_circular_buffer_parameter_constructor;
        template<typename T> BufferConstructor<CircularBuffer<T>> CircularBuffer<T>::_circular_buffer_constructor;
    }
}
