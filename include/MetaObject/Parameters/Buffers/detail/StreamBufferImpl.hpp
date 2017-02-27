#pragma once
#include <MetaObject/Logging/Log.hpp>
#include <boost/chrono.hpp>
namespace mo
{
    namespace Buffer
    {
        template<class T> StreamBuffer<T>::StreamBuffer(const std::string& name):
            ITypedParameter<T>(name, Buffer_e),
            _time_padding(500 * mo::milli * mo::second)
        {
        
        }

        template<class T> T*   StreamBuffer<T>::GetDataPtr(boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
        {
            T* result = Map<T>::GetDataPtr(ts, ctx, &_current_frame_number);
            if(result)
            {
                _current_timestamp = ts;
                prune();
            }
            if(fn)
                *fn = _current_frame_number;
            return result;
        }

        template<class T> T* StreamBuffer<T>::GetDataPtr(size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
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

        template<class T> bool StreamBuffer<T>::GetData(T& value, boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
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
        template<class T> T StreamBuffer<T>::GetData(boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
        {
            T result = Map<T>::GetData(ts, ctx, &_current_frame_number);
            _current_timestamp = ts;
            if(fn)
                *fn = _current_frame_number;
            prune();
            return result;
        }
        template<class T> void StreamBuffer<T>::SetSize(size_t size)
        {
            if(_time_padding)
                _time_padding = boost::none;
            _frame_padding = size;
        }
        template<class T> void StreamBuffer<T>::SetSize(mo::time_t time)
        {
            if(_frame_padding)
                _frame_padding = boost::none;
            _time_padding = time;
        }
        template<class T> void StreamBuffer<T>::prune()
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
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
                        break;
                    }
                }
            }if(_frame_padding)
            {

            }
        }
        template<class T> std::shared_ptr<IParameter> StreamBuffer<T>::DeepCopy() const
        {
            return std::shared_ptr<IParameter>(new StreamBuffer<T>());
        }

        // ------------------------------------------------------------
        template<class T> BlockingStreamBuffer<T>::BlockingStreamBuffer(const std::string& name) :
            StreamBuffer<T>(name),
            ITypedParameter<T>(name, mo::Buffer_e),
            _size(100)
        {

        }
        template<class T> void BlockingStreamBuffer<T>::SetSize(long long size)
        {
            _size = size;
        }
        
        /*template<class T>
        ITypedParameter<T>* BlockingStreamBuffer<T>::UpdateData(T& data_, mo::time_t ts, Context* ctx)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            while (this->_data_buffer.size() >= _size)
            {
                LOG(trace) << "Pushing to " << this->GetTreeName() << " waiting on read";
                _cv.wait(lock);
            }
            this->_data_buffer[ts] = data_;
            IParameter::modified = true;
            this->_timestamp = ts;
            IParameter::OnUpdate(ctx);
            return this;
        }
        template<class T>
        ITypedParameter<T>* BlockingStreamBuffer<T>::UpdateData(const T& data_, mo::time_t ts, Context* ctx)
        {
            boost::unique_lock<boost::recursive_mutex> lock(IParameter::mtx());
            while (this->_data_buffer.size() >= _size)
            {
                LOG(trace) << "Pushing to " << this->GetTreeName() << " waiting on read";
                _cv.wait(lock);
            }
            this->_data_buffer[ts] = data_;
            IParameter::modified = true;
            this->_timestamp = ts;
            IParameter::OnUpdate(ctx);
            return this;
        }
        template<class T>
        ITypedParameter<T>* BlockingStreamBuffer<T>::UpdateData(T* data_, mo::time_t ts, Context* ctx)
        {
            {
                boost::unique_lock<boost::recursive_mutex> lock(IParameter::mtx());
                while(this->_data_buffer.size() >= _size)
                {
                    LOG(trace) << "Pushing to " << this->GetTreeName() << " waiting on read";
                    _cv.wait_for(lock, boost::chrono::microseconds(100));
                    lock.unlock();
                    IParameter::OnUpdate(ctx);
                    lock.lock();
                }
                this->_data_buffer[ts] = *data_;
                IParameter::modified = true;
                this->_timestamp = ts;
            }
            IParameter::OnUpdate(ctx);
            return this;
        }*/
        template<class T>
        void BlockingStreamBuffer<T>::prune()
        {
            boost::unique_lock<boost::recursive_mutex> lock(IParameter::mtx());
            if (StreamBuffer<T>::_current_timestamp && StreamBuffer<T>::_time_padding)
            {
                auto itr = this->_data_buffer.begin();
                while (itr != this->_data_buffer.end())
                {
                    if (itr->first < *StreamBuffer<T>::_current_timestamp - *StreamBuffer<T>::_time_padding)
                    {
                        itr = this->_data_buffer.erase(itr);
                    }
                    else
                    {
                        break;
                    }
                }
            }
            lock.unlock();
            _cv.notify_all();
        }
    }
}
