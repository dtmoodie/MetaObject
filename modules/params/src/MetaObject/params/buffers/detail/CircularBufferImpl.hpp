#pragma once

namespace mo
{
    namespace Buffer
    {
        template <class T>
        CircularBuffer<T>::CircularBuffer(T&& init, const std::string& name, OptionalTime_t ts, ParamFlags type)
            : ITInputParam<T>(name), ITParam<T>(name, mo::ParamFlags::Buffer_e), IParam(name, mo::ParamFlags::Buffer_e)
        {
            (void)&_circular_buffer_constructor;
            (void)&_circular_buffer_param_constructor;
            _data_buffer.set_capacity(10);
            _data_buffer.push_back(State<T>(ts, init));
        }

        template <class T>
        CircularBuffer<T>::CircularBuffer(const std::string& name, OptionalTime_t ts, ParamFlags type)
            : ITInputParam<T>(name), ITParam<T>(name, mo::ParamFlags::Buffer_e), IParam(name, mo::ParamFlags::Buffer_e)
        {
            _data_buffer.set_capacity(10);
        }

        template <class T>
        bool CircularBuffer<T>::getData(InputStorage_t& data, const OptionalTime_t& ts, Context* ctx, size_t* fn_)
        {
            if (!ts && _data_buffer.size())
            {
                if (fn_)
                    *fn_ = _data_buffer.back().fn;
                data = _data_buffer.back().data;
                return true;
            }
            for (auto& itr : _data_buffer)
            {
                if (itr.ts == ts)
                {
                    if (fn_)
                        *fn_ = itr.fn;
                    data = itr.data;
                    return true;
                }
            }
            return false;
        }

        template <class T>
        bool CircularBuffer<T>::getData(InputStorage_t& data, size_t fn, Context* ctx, OptionalTime_t* ts_)
        {
            for (auto& itr : _data_buffer)
            {
                if (itr.fn == fn && itr.ts)
                {
                    if (ts_)
                        *ts_ = *itr.ts;
                    data = itr.data;
                    return true;
                }
            }
            return false;
        }

        template <class T>
        bool CircularBuffer<T>::updateDataImpl(const Storage_t& data_,
                                               const OptionalTime_t& ts,
                                               const ContextPtr_t& ctx,
                                               size_t fn,
                                               const std::shared_ptr<ICoordinateSystem>& cs)
        {
            {
                mo::Mutex_t::scoped_lock lock(IParam::mtx());
                if (ts)
                    _data_buffer.push_back(State<T>(*ts, fn, ctx.get(), cs, data_));
                else
                    _data_buffer.push_back(State<T>(fn, ctx.get(), cs, data_));
                this->_modified = true;
            }
            ITParam<T>::_typed_update_signal(data_, this, ctx, ts, fn, cs, mo::InputUpdated_e);
            return true;
        }

        template <class T>
        void CircularBuffer<T>::setFrameBufferCapacity(size_t size)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            _data_buffer.set_capacity(size);
        }

        template <class T>
        void CircularBuffer<T>::setTimePaddingCapacity(mo::Time_t time)
        {
            (void)time;
        }

        template <class T>
        boost::optional<size_t> CircularBuffer<T>::getFrameBufferCapacity()
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            return _data_buffer.capacity();
        }

        template <class T>
        OptionalTime_t CircularBuffer<T>::getTimePaddingCapacity()
        {
            return {};
        }

        template <class T>
        size_t CircularBuffer<T>::getSize()
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            return _data_buffer.size();
        }

        template <class T>
        bool CircularBuffer<T>::getTimestampRange(mo::Time_t& start, mo::Time_t& end)
        {
            if (_data_buffer.size())
            {
                mo::Mutex_t::scoped_lock lock(IParam::mtx());
                if (_data_buffer.back().ts && _data_buffer.front().ts)
                {
                    start = *_data_buffer.back().ts;
                    end = *_data_buffer.front().ts;
                    return true;
                }
            }
            return false;
        }

        template <class T>
        bool CircularBuffer<T>::getFrameNumberRange(size_t& start, size_t& end)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (_data_buffer.size())
            {
                mo::Mutex_t::scoped_lock lock(IParam::mtx());
                start = _data_buffer.back().fn;
                end = _data_buffer.front().fn;
                return true;
            }
            return false;
        }

        template <class T>
        void CircularBuffer<T>::onInputUpdate(ConstStorageRef_t data,
                                              IParam* param,
                                              Context* ctx,
                                              OptionalTime_t ts,
                                              size_t fn,
                                              const std::shared_ptr<ICoordinateSystem>& cs,
                                              UpdateFlags fg)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (ts)
                _data_buffer.push_back(State<T>(*ts, fn, ctx, cs, data));
            else
                _data_buffer.push_back(State<T>(fn, ctx, cs, data));
            this->_modified = true;
            lock.unlock();
            IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            ITParam<T>::_typed_update_signal(data, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
        }
        template <typename T>
        ParamConstructor<CircularBuffer<T>> CircularBuffer<T>::_circular_buffer_param_constructor;
        template <typename T>
        BufferConstructor<CircularBuffer<T>> CircularBuffer<T>::_circular_buffer_constructor;
    }
}
