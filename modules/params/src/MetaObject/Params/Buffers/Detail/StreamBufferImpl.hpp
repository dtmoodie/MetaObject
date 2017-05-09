#pragma once
#include <MetaObject/Logging/Log.hpp>
#include <boost/chrono.hpp>
namespace mo{
    namespace Buffer{
        template<class T> StreamBuffer<T>::StreamBuffer(const std::string& name):
            ITParam<T>(name, Buffer_e),
            _time_padding(500 * mo::milli * mo::second),
            _frame_padding(100){
        }

        template<class T>
        bool StreamBuffer<T>::getData(Storage_t& data, const OptionalTime_t& ts, Context* ctx, size_t* fn_){
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if(Map<T>::getData(data, ts, ctx, &_current_frame_number)){
                if(!ts) _current_timestamp = this->_data_buffer.rbegin()->first.ts;
                else    _current_timestamp = ts;
                prune();
                if(fn_) *fn_ = _current_frame_number;
                return true;
            }
            return false;
        }

        template<class T>
        bool StreamBuffer<T>::getData(Storage_t& data, size_t fn, Context* ctx, OptionalTime_t* ts_){
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (Map<T>::getData(data, fn, ctx, &_current_timestamp)) {
                _current_frame_number = fn;
                prune();
                if (ts_) *ts_ = _current_timestamp;
                return true;
            }
            return false;
        }
        
        template<class T> void StreamBuffer<T>::setFrameBufferCapacity(size_t size){
            if (_time_padding) _time_padding = boost::none;
            _frame_padding = size;
        }
        
        template<class T> void StreamBuffer<T>::setTimePaddingCapacity(mo::Time_t time){
            if (_frame_padding) _frame_padding = boost::none;
            _time_padding = time;
        }
        
        template<class T> boost::optional<size_t> StreamBuffer<T>::getFrameBufferCapacity(){return _frame_padding;}

        template<class T> OptionalTime_t StreamBuffer<T>::getTimePaddingCapacity(){ return _time_padding; }

        template<class T> void StreamBuffer<T>::prune(){
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if(_current_timestamp && _time_padding){
                auto itr = this->_data_buffer.begin();
                while(itr != this->_data_buffer.end()){
                    if(itr->first < (*_current_timestamp - *_time_padding)){
                        itr = this->_data_buffer.erase(itr);
                    }else{
                        ++itr;
                    }
                }
            }
            if(_frame_padding && _current_frame_number > *_frame_padding){
                auto itr = this->_data_buffer.begin();
                while(itr != this->_data_buffer.end()){
                    if(itr->first < (_current_frame_number - *_frame_padding)){
                        itr = this->_data_buffer.erase(itr);
                    }else{
                        ++itr;
                    }
                }
            }
        }

        // ------------------------------------------------------------
        template<class T> BlockingStreamBuffer<T>::BlockingStreamBuffer(const std::string& name) :
            StreamBuffer<T>(name),
            ITParam<T>(name, mo::Buffer_e),
            _size(100){
        }

        template<class T>
        bool BlockingStreamBuffer<T>::updateDataImpl(const T& data_, OptionalTime_t ts, Context* ctx, size_t fn, ICoordinateSystem* cs){
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            while (this->_data_buffer.size() > _size){
                LOG_EVERY_N(debug, 10) << "Pushing to " << this->getTreeName() << " waiting on read, current buffer size " << this->_data_buffer.size();
                _cv.wait_for(lock, boost::chrono::milliseconds(2));
                // Periodically emit an update signal in case a dirty flag was not set correctly and the read thread is just sleeping
                IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            }
            Map<T>::_data_buffer[{ts,IParam::_fn}] = data_;
            IParam::_modified = true;
            lock.unlock();
            IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            ITParam<T>::_typed_update_signal(data_, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            return true;
        }

        template<class T>
        void BlockingStreamBuffer<T>::prune(){
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            StreamBuffer<T>::prune();
            auto itr = this->_data_buffer.begin();
            while(this->_data_buffer.size() >= _size){
                if(this->_current_timestamp && itr->first.ts == this->_current_timestamp)
                        break;
                if(this->_current_frame_number && itr->first.fn == this->_current_frame_number)
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
