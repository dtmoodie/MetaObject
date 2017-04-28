#pragma once
#include <MetaObject/Logging/Log.hpp>
#include <boost/chrono.hpp>
namespace mo
{
    namespace Buffer
    {
        template<class T> StreamBuffer<T>::StreamBuffer(const std::string& name):
            ITParam<T>(name, Buffer_e),
            _time_padding(500 * mo::milli * mo::second),
            _frame_padding(100)

        {

        }

        template<class T> T*   StreamBuffer<T>::GetDataPtr(OptionalTime_t ts, Context* ctx, size_t* fn)
        {
            T* result = Map<T>::GetDataPtr(ts, ctx, &_current_frame_number);
            if(result)
            {
                if(!ts)
                {
                    boost::unique_lock<boost::recursive_mutex> lock(IParam::mtx());
                    _current_timestamp = this->_data_buffer.rbegin()->first.ts;
                }else
                {
                    _current_timestamp = ts;
                }
                prune();
                if(fn)
                    *fn = _current_frame_number;
            }

            return result;
        }

        template<class T> T* StreamBuffer<T>::GetDataPtr(size_t fn, Context* ctx, OptionalTime_t* ts)
        {
            T* result = Map<T>::GetDataPtr(fn, ctx, ts);
            if(result)
            {
                if(ts)
                    _current_timestamp = *ts;
                prune();
            }
            return result;
        }

        template<class T> bool StreamBuffer<T>::GetData(T& value, OptionalTime_t ts, Context* ctx, size_t* fn)
        {
            if(Map<T>::GetData(value, ts, ctx, &_current_frame_number))
            {
                _current_timestamp = ts;
                if(fn)
                    *fn = _current_frame_number;
                prune();
                return true;
            }
            return false;
        }

        template<class T> bool StreamBuffer<T>::GetData(T& value, size_t fn, Context* ctx, OptionalTime_t* ts)
        {
            if(Map<T>::GetData(value, fn, ctx, &_current_timestamp))
            {
                if(ts)
                    *ts = _current_timestamp;
                prune();
                return true;
            }
            return false;
        }

        template<class T> T StreamBuffer<T>::GetData(OptionalTime_t ts, Context* ctx, size_t* fn)
        {
            T result = Map<T>::GetData(ts, ctx, &_current_frame_number);
            _current_timestamp = ts;
            if(fn)
                *fn = _current_frame_number;
            prune();
            return result;
        }

        template<class T> T StreamBuffer<T>::GetData(size_t fn, Context* ctx, OptionalTime_t* ts)
        {
            T result = Map<T>::GetData(fn, ctx, &_current_timestamp);
            //_current_timestamp = ts;
            _current_frame_number = fn;
            if(ts)
                *ts = _current_timestamp;
            prune();
            return result;
        }
        
        template<class T> void StreamBuffer<T>::SetFrameBufferCapacity(size_t size)
        {
            if (_time_padding)
                _time_padding = boost::none;
            _frame_padding = size;
        }
        
        template<class T> void StreamBuffer<T>::SetTimePaddingCapacity(mo::Time_t time)
        {
            if (_frame_padding)
                _frame_padding = boost::none;
            _time_padding = time;
        }
        
        template<class T> boost::optional<size_t> StreamBuffer<T>::GetFrameBufferCapacity()
        {
            return _frame_padding;
        }

        template<class T> OptionalTime_t StreamBuffer<T>::GetTimePaddingCapacity()
        {
            return _time_padding;
        }

        
        template<class T> void StreamBuffer<T>::prune()
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if(_current_timestamp && _time_padding)
            {
                auto itr = this->_data_buffer.begin();
                while(itr != this->_data_buffer.end())
                {
                    if(itr->first < (*_current_timestamp - *_time_padding))
                    {
                        itr = this->_data_buffer.erase(itr);
                    }else
                    {
                        ++itr;
                    }
                }
                // Temp fix until merged with boost_timestamp, only prune _padding images behind current frame
                /*auto end_itr = this->_data_buffer.find(_current_timestamp);
                for(int i = 0; i < _padding && end_itr != this->_data_buffer.begin(); ++i)
                    --end_itr;
                auto itr = this->_data_buffer.begin();
                while(itr != this->_data_buffer.end())
                {
                    if(itr->first < (*_current_timestamp - *_time_padding))
                    {
                        itr = this->_data_buffer.erase(itr);
                    }else
                    {
                        ++itr;
                    }
                }*/
            }
            if(_frame_padding && _current_frame_number > *_frame_padding)
            {
                auto itr = this->_data_buffer.begin();
                while(itr != this->_data_buffer.end())
                {
                    if(itr->first < (_current_frame_number - *_frame_padding))
                    {
                        itr = this->_data_buffer.erase(itr);
                    }else
                    {
                        ++itr;
                    }
                }
            }
        }
        template<class T> std::shared_ptr<IParam> StreamBuffer<T>::DeepCopy() const
        {
            return std::shared_ptr<IParam>(new StreamBuffer<T>());
        }

        // ------------------------------------------------------------
        template<class T> BlockingStreamBuffer<T>::BlockingStreamBuffer(const std::string& name) :
            StreamBuffer<T>(name),
            ITParam<T>(name, mo::Buffer_e),
            _size(100)
        {

        }

        template<class T>
        bool BlockingStreamBuffer<T>::UpdateDataImpl(const T& data_, OptionalTime_t ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            while (this->_data_buffer.size() > _size)
            {
                LOG_EVERY_N(debug, 100) << "Pushing to " << this->getTreeName() << " waiting on read, current buffer size " << this->_data_buffer.size();
                _cv.wait_for(lock, boost::chrono::milliseconds(2));
                // Periodically emit an update signal in case a dirty flag was not set correctly and the read thread is just sleeping
                IParam::_update_signal(ctx, this);
            }
            if(fn)
                IParam::_fn = *fn;
            else
                ++IParam::_fn;
            Map<T>::_data_buffer[{ts,IParam::_fn}] = data_;
            IParam::_modified = true;
            IParam::_ts = ts;
            lock.unlock();
            IParam::OnUpdate(ctx);
            return true;
        }

        template<class T>
        void BlockingStreamBuffer<T>::prune()
        {
            boost::unique_lock<boost::recursive_mutex> lock(IParam::mtx());
            StreamBuffer<T>::prune();
            auto itr = this->_data_buffer.begin();
            while(this->_data_buffer.size() >= _size)
            {
                if(this->_current_timestamp)
                    if(itr->first.ts == this->_current_timestamp)
                        break;
                if(this->_current_frame_number)
                    if(itr->first.fn == this->_current_frame_number)
                        break;
#ifdef _DEBUG
                LOG(trace) << "Removing item at (fn/ts) " << itr->first.fn << "/" << itr->first.ts << " from " << this->getTreeName();
#endif
                itr = this->_data_buffer.erase(itr);
            }

            lock.unlock();
            _cv.notify_all();
        }
    }
}
