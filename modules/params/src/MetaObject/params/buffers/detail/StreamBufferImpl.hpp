#pragma once
#include <MetaObject/logging/logging.hpp>
#include <boost/chrono.hpp>
namespace mo
{
    namespace Buffer
    {
        template <class T>
        StreamBuffer<T>::StreamBuffer(const std::string& name)
            : TParam<T>(name, ParamFlags::Buffer_e)
            , _time_padding(mo::ms * 500)
            , _frame_padding(100)
        {
        }

        template <class T>
        bool StreamBuffer<T>::getData(InputStorage_t& data, const OptionalTime& ts, Context* ctx, size_t* fn_)
        {
            Lock_t lock(IParam::mtx());
            if (Map<T>::getData(data, ts, ctx, &_current_frame_number))
            {
                if (!ts)
                {
                    _current_timestamp = this->_data_buffer.rbegin()->first.ts;
                }
                else
                {
                    _current_timestamp = ts;
                }
                prune();
                if (fn_)
                {
                    *fn_ = _current_frame_number;
                }
                return true;
            }
            else
            {
            }
            return false;
        }

        template <class T>
        bool StreamBuffer<T>::getData(InputStorage_t& data, size_t fn, Context* ctx, OptionalTime* ts_)
        {
            Lock_t lock(IParam::mtx());
            if (Map<T>::getData(data, fn, ctx, &_current_timestamp))
            {
                _current_frame_number = fn;
                prune();
                if (ts_)
                {
                    *ts_ = _current_timestamp;
                }
                return true;
            }
            return false;
        }

        template <class T>
        void StreamBuffer<T>::setFrameBufferCapacity(size_t size)
        {
            if (_time_padding)
            {
                _time_padding = boost::none;
            }
            _frame_padding = size;
        }

        template <class T>
        void StreamBuffer<T>::setTimePaddingCapacity(mo::Time time)
        {
            if (_frame_padding)
            {
                _frame_padding = boost::none;
            }
            _time_padding = time;
        }

        template <class T>
        boost::optional<size_t> StreamBuffer<T>::getFrameBufferCapacity()
        {
            return _frame_padding;
        }

        template <class T>
        OptionalTime StreamBuffer<T>::getTimePaddingCapacity()
        {
            return _time_padding;
        }

        template <class T>
        void StreamBuffer<T>::prune()
        {
            Lock_t lock(IParam::mtx());
            if (_current_timestamp && _time_padding)
            {
                auto itr = this->_data_buffer.begin();
                while (itr != this->_data_buffer.end())
                {
                    if (itr->first.ts && *itr->first.ts < mo::Time(*_current_timestamp - *_time_padding))
                    {
                        itr = this->_data_buffer.erase(itr);
                    }
                    else
                    {
                        ++itr;
                    }
                }
            }
            else
            {
            }
            if (_frame_padding && _current_frame_number > *_frame_padding)
            {
                auto itr = this->_data_buffer.begin();
                while (itr != this->_data_buffer.end())
                {
                    if (itr->first < (_current_frame_number - *_frame_padding))
                    {
                        itr = this->_data_buffer.erase(itr);
                    }
                    else
                    {
                        ++itr;
                    }
                }
            }
            else
            {
            }
        }

        // ------------------------------------------------------------
        template <class T>
        BlockingStreamBuffer<T>::BlockingStreamBuffer(const std::string& name)
            : StreamBuffer<T>(name)
            , TParam<T>(name, mo::ParamFlags::Buffer_e)
            , _size(100)
        {
        }

        template <class T>
        bool BlockingStreamBuffer<T>::updateDataImpl(const Storage_t& data_,
                                                     const OptionalTime& ts,
                                                     Context* ctx,
                                                     size_t fn,
                                                     const std::shared_ptr<ICoordinateSystem>& cs)
        {
            Lock_t lock(IParam::mtx());
            while (this->_data_buffer.size() > _size)
            {
                MO_LOG_EVERY_N(debug, 10) << "Pushing to " << this->getTreeName()
                                          << " waiting on read, current buffer size " << this->_data_buffer.size();
                _cv.wait_for(lock, boost::chrono::milliseconds(2));
                // Periodically emit an update signal in case a dirty flag was not set correctly and the read thread is
                // just sleeping
                if (lock)
                {
                    lock.unlock();
                }
                IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
                if (!lock)
                {
                    lock.lock();
                }
            }
            if (!lock)
            {
                lock.lock();
            }
            Map<T>::_data_buffer[{ts, fn, cs, ctx}] = data_;
            this->modified(true);
            lock.unlock();
            IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            this->emitTypedUpdate(data_, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            return true;
        }

        template <class T>
        bool BlockingStreamBuffer<T>::updateDataImpl(Storage_t&& data_,
                                                     const OptionalTime& ts,
                                                     Context* ctx,
                                                     size_t fn,
                                                     const std::shared_ptr<ICoordinateSystem>& cs)
        {
            Lock_t lock(IParam::mtx());
            while (this->_data_buffer.size() > _size)
            {
                MO_LOG_EVERY_N(debug, 10) << "Pushing to " << this->getTreeName()
                                          << " waiting on read, current buffer size " << this->_data_buffer.size();
                _cv.wait_for(lock, boost::chrono::milliseconds(2));
                // Periodically emit an update signal in case a dirty flag was not set correctly and the read thread is
                // just sleeping
                if (lock)
                {
                    lock.unlock();
                }
                IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
                if (!lock)
                {
                    lock.lock();
                }
            }
            if (!lock)
            {
                lock.lock();
            }
            auto itr = Map<T>::_data_buffer.emplace(SequenceKey(ts, fn, cs, ctx), std::move(data_));
            this->modified(true);
            lock.unlock();
            IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            this->emitTypedUpdate(itr.first->second, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            return true;
        }

        template <class T>
        void BlockingStreamBuffer<T>::setFrameBufferCapacity(size_t size)
        {
            _size = size;
            StreamBuffer<T>::setFrameBufferCapacity(size);
        }

        template <class T>
        void BlockingStreamBuffer<T>::prune()
        {
            Lock_t lock(IParam::mtx());
            StreamBuffer<T>::prune();
            auto itr = this->_data_buffer.begin();
            while (this->_data_buffer.size() >= _size)
            {
                if (this->_current_timestamp && itr->first.ts == this->_current_timestamp)
                {
                    break;
                }
                else
                {
                }
                if (this->_current_frame_number && itr->first.fn == this->_current_frame_number)
                {
                    break;
                }
                else
                {
                }
                itr = this->_data_buffer.erase(itr);
            }
            lock.unlock();
            _cv.notify_all();
        }

        template <class T>
        void BlockingStreamBuffer<T>::onInputUpdate(ConstStorageRef_t data,
                                                    IParam* input,
                                                    Context* ctx,
                                                    OptionalTime ts,
                                                    size_t fn,
                                                    const std::shared_ptr<ICoordinateSystem>& cs,
                                                    UpdateFlags)
        {
            Lock_t lock(IParam::mtx());
            while (this->_data_buffer.size() > _size)
            {
                MO_LOG_EVERY_N(debug, 10) << "Pushing to " << this->getTreeName()
                                          << " waiting on read, current buffer size " << this->_data_buffer.size();
                _cv.wait_for(lock, boost::chrono::milliseconds(2));
                // Periodically emit an update signal in case a dirty flag was not set correctly and the read thread is
                // just sleeping
                if (lock)
                {
                    lock.unlock();
                }
                else
                {
                }
                IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
                TParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
                if (!lock)
                {
                    lock.lock();
                }
                else
                {
                }
            }
            if (!lock)
            {
                lock.lock();
            }
            else
            {
            }
            this->_data_buffer[{ts, fn, cs, ctx}] = data;
            this->modified(true);
            lock.unlock();
            IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            TParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
        }

        template <class T>
        DroppingStreamBuffer<T>::DroppingStreamBuffer(const std::string& name)
            : BlockingStreamBuffer<T>(name)
        {
        }

        template <class T>
        bool DroppingStreamBuffer<T>::updateDataImpl(const Storage_t& data_,
                                                     const OptionalTime& ts,
                                                     Context* ctx,
                                                     size_t fn,
                                                     const std::shared_ptr<ICoordinateSystem>& cs)
        {
            Lock_t lock(IParam::mtx());
            if (this->_data_buffer.size() > this->_size)
            {
                return false;
            }
            else
            {
            }
            if (!lock)
            {
                lock.lock();
            }
            Map<T>::_data_buffer[{ts, fn, cs, ctx}] = data_;
            this->modified(true);
            lock.unlock();
            IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            this->emitTypedUpdate(data_, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            return true;
        }

        template <class T>
        bool DroppingStreamBuffer<T>::updateDataImpl(Storage_t&& data_,
                                                     const OptionalTime& ts,
                                                     Context* ctx,
                                                     size_t fn,
                                                     const std::shared_ptr<ICoordinateSystem>& cs)
        {
            Lock_t lock(IParam::mtx());
            if (this->_data_buffer.size() > this->_size)
            {
                return false;
            }
            else
            {
            }
            if (!lock)
            {
                lock.lock();
            }
            auto itr = Map<T>::_data_buffer.emplace(SequenceKey(ts, fn, cs, ctx), std::move(data_));
            this->modified(true);
            lock.unlock();
            IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            this->emitTypedUpdate(itr.first->second, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            return true;
        }

        template <class T>
        void DroppingStreamBuffer<T>::onInputUpdate(ConstStorageRef_t data,
                                                    IParam* input,
                                                    Context* ctx,
                                                    OptionalTime ts,
                                                    size_t fn,
                                                    const std::shared_ptr<ICoordinateSystem>& cs,
                                                    UpdateFlags)
        {

            Lock_t lock(IParam::mtx());
            if (this->_data_buffer.size() > this->_size)
            {
                return;
            }
            if (!lock)
            {
                lock.lock();
            }
            this->_data_buffer[{ts, fn, cs, ctx}] = data;
            this->modified(true);
            lock.unlock();
            IParam::emitUpdate(ts, ctx, fn, cs, mo::BufferUpdated_e);
            TParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
        }
    }
}
