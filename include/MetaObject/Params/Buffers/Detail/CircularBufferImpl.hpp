#pragma once

namespace mo
{
    namespace Buffer
    {
        template<class T>
        CircularBuffer<T>::CircularBuffer(T&& init, const std::string& name, OptionalTime_t ts, ParamType type ):
            ITInputParam<T>(name),
            ITParam<T>(name, mo::Buffer_e)
        {
            (void)&_circular_buffer_constructor;
            (void)&_circular_buffer_Param_constructor;
            _data_buffer.set_capacity(10);
            _data_buffer.push_back(State<T>(ts, init));
        }

        template<class T>
        CircularBuffer<T>::CircularBuffer(const std::string& name, OptionalTime_t ts, ParamType type ):
            ITInputParam<T>(name),
            ITParam<T>(name, mo::Buffer_e)
        {
            _data_buffer.set_capacity(10);
        }

        template<class T>
        T* CircularBuffer<T>::GetDataPtr(OptionalTime_t ts, Context* ctx, size_t* fn)
        {
            if(!ts && _data_buffer.size())
                return &_data_buffer.back().data;

            for (auto& itr : _data_buffer)
            {
                if (itr.ts == ts)
                {
                    if(fn && itr.fn)
                        *fn = itr.fn;
                    return &itr.data;
                }
            }
            return nullptr;
        }

        template<class T>
        T* CircularBuffer<T>::GetDataPtr(size_t fn, Context* ctx, OptionalTime_t* ts)
        {
            for (auto& itr : _data_buffer)
            {
                if (itr.fn == fn)
                {
                    if(ts)
                        *ts = itr.ts;
                    return &itr.data;
                }
            }
            return nullptr;
        }
         
        template<class T>
        bool CircularBuffer<T>::GetData(T& value, OptionalTime_t ts, Context* ctx, size_t* fn)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());

            if (!ts && _data_buffer.size())
            {
                value = _data_buffer.back().data;
                return true;
            }
            for (auto& itr : _data_buffer)
            {
                if (itr.ts == ts)
                {
                    value = itr.data;
                    if(fn)
                        *fn = itr.fn;
                    return true;
                }
            }
            return false;
        }

        template<class T>
        bool CircularBuffer<T>::GetData(T& value, size_t fn, Context* ctx, OptionalTime_t* ts)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (fn == std::numeric_limits<size_t>::max() && _data_buffer.size())
            {
                value = _data_buffer.back().data;
                if(ts)
                    *ts = _data_buffer.back().ts;
                return true;
            }
            for (auto& itr : _data_buffer)
            {
                if (itr.fn == fn)
                {
                    value = itr.data;
                    if(ts)
                        *ts = itr.ts;
                    return true;
                }
            }
            return false;
        }

        template<class T>
        T CircularBuffer<T>::GetData(OptionalTime_t ts, Context* ctx, size_t* fn)
        {
            if (!ts && _data_buffer.size())
            {
                if(fn)
                    *fn = _data_buffer.back().fn;
                return _data_buffer.back().data;
            }
            for (auto& itr : _data_buffer)
            {
                if (itr.ts == ts)
                {
                    if(fn)
                        *fn = itr.fn;
                    return itr.data;
                }
            }
            THROW(debug) << "Could not find timestamp " << ts
                         << " in range (" << _data_buffer.back().ts << "," << _data_buffer.front().ts <<")";
            return T();
        }

        template<class T>
        T CircularBuffer<T>::GetData(size_t fn, Context* ctx, OptionalTime_t* ts)
        {
            if (fn == std::numeric_limits<size_t>::max() && _data_buffer.size())
            {
                if(ts)
                    *ts = _data_buffer.back().ts;
                return _data_buffer.back().data;
            }
            for (auto& itr : _data_buffer)
            {
                if (itr.fn == fn)
                {
                    if(ts)
                        *ts = itr.ts;
                    return itr.data;
                }
            }
            THROW(debug) << "Could not find frame number" << ts
                         << " in range (" << _data_buffer.back().fn << ","
                         << _data_buffer.front().fn <<")";
            return T();
        }

        template<class T>
        bool CircularBuffer<T>::UpdateDataImpl(const T& data_,
                                               OptionalTime_t ts,
                                               Context* ctx,
                                               boost::optional<size_t> fn,
                                               ICoordinateSystem* cs)
        {
			{
				mo::Mutex_t::scoped_lock lock(IParam::mtx());
				if (ts)
					_data_buffer.push_back(State<T>(*ts, fn ? *fn : 0, ctx, cs, data_));
				else
					_data_buffer.push_back(State<T>(fn ? *fn : 0, ctx, cs, data_));
				this->_modified = true;
			}
            this->commit(ts, ctx, fn, cs);
            return true;
        }

        template<class T> bool CircularBuffer<T>::Update(IParam* other, Context* ctx)
        {
            auto TParam = dynamic_cast<ITParam<T>*>(other);
            if (TParam)
            {
                T data;
                if (TParam->GetData(data))
                {
                    mo::Mutex_t::scoped_lock lock(IParam::mtx());
                    _data_buffer.push_back(std::pair<mo::Time_t, T>(TParam->getTimestamp(), data));
                    IParam::_modified = true;
                    IParam::OnUpdate(ctx);
                }
            }
            return false;
        }

        template<class T> void CircularBuffer<T>::SetFrameBufferCapacity(size_t size)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            _data_buffer.set_capacity(size);
        }
        
        template<class T> void CircularBuffer<T>::SetTimePaddingCapacity(mo::Time_t time)
        {
            (void)time;
        }
        
        template<class T> boost::optional<size_t> CircularBuffer<T>::GetFrameBufferCapacity()
        {
            return _data_buffer.capacity();
        }
        
        template<class T> OptionalTime_t CircularBuffer<T>::GetTimePaddingCapacity()
        {
            return {};
        }

        template<class T> size_t CircularBuffer<T>::GetSize()
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            return _data_buffer.size();
        }

        template<class T> bool CircularBuffer<T>::getTimestampRange(mo::Time_t& start, mo::Time_t& end)
        {
            if (_data_buffer.size())
            {
                mo::Mutex_t::scoped_lock lock(IParam::mtx());
                if(_data_buffer.back().ts && _data_buffer.front().ts)
                {
                    start = *_data_buffer.back().ts;
                    end = *_data_buffer.front().ts;
                    return true;
                }
            }
            return false;
        }

        template<class T> bool CircularBuffer<T>::getFrameNumberRange(size_t& start,size_t& end)
        {
            if (_data_buffer.size())
            {
                mo::Mutex_t::scoped_lock lock(IParam::mtx());
                start = _data_buffer.back().fn;
                end = _data_buffer.front().fn;
                return true;
            }
            return false;
        }

        template<class T> std::shared_ptr<IParam>  CircularBuffer<T>::DeepCopy() const
        {
            auto buffer = new CircularBuffer<T>(IParam::_name);
            buffer->SetInput(this->input);
            return std::shared_ptr<IParam>(buffer);
        }

        template<class T> void  CircularBuffer<T>::onInputUpdate(Context* ctx, IParam* param)
        {
            if(this->input)
            {
				mo::Mutex_t::scoped_lock lock(this->input->mtx());
				UpdateDataImpl(*this->input->GetDataPtr(), this->input->getTimestamp(), this->input->getContext(), this->input->getFrameNumber(), this->input->GetCoordinateSystem());
            }
        }
        template<typename T> ParamConstructor<CircularBuffer<T>> CircularBuffer<T>::_circular_buffer_Param_constructor;
        template<typename T> BufferConstructor<CircularBuffer<T>> CircularBuffer<T>::_circular_buffer_constructor;
    }
}
