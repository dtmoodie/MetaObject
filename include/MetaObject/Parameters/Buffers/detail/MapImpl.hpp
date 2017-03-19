#pragma once

namespace mo
{
    namespace Buffer
    {
        template<class T> Map<T>::Map(const std::string& name) :
            ITypedInputParameter<T>(name),
            ITypedParameter<T>(name, mo::Buffer_e)
        {
            this->SetFlags(Buffer_e);
        }

        template<class T>
        T* Map<T>::GetDataPtr(boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
			auto itr = Search(ts);
			if (itr != _data_buffer.end())
			{
				IParameter::_ts = itr->first.ts;
				IParameter::_fn = itr->first.fn;
				if (fn)
					*fn = this->_fn;
				return &(itr->second);
			}
            return nullptr;
        }

        template<class T>
        T* Map<T>::GetDataPtr(size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
        {
			boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
			auto itr = Search(fn);
			if (itr != _data_buffer.end())
			{
				IParameter::_ts = itr->first.ts;
				IParameter::_fn = itr->first.fn;
				if (ts)
					*ts = this->_ts;
				return &(itr->second);
			}
			return nullptr;
        }

        template<class T>
        bool Map<T>::GetData(T& value, boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
        {
			boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
			auto itr = Search(ts);
			if (itr != _data_buffer.end())
			{
				IParameter::_ts = itr->first.ts;
				IParameter::_fn = itr->first.fn;
				if (fn)
					*fn = this->_fn;
				value = (itr->second);
				return true;
			}
			return false;
        }

        template<class T>
        bool Map<T>::GetData(T& value, size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
        {
			boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
			auto itr = Search(fn);
			if (itr != _data_buffer.end())
			{
				IParameter::_ts = itr->first.ts;
				IParameter::_fn = itr->first.fn;
				if (ts)
					*ts = this->_ts;
				value = (itr->second);
				return true;
			}
			return false;
        }

        template<class T>
        T Map<T>::GetData(boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
        {
			boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
			auto itr = Search(ts);
			if (itr != _data_buffer.end())
			{
				IParameter::_ts = itr->first.ts;
				IParameter::_fn = itr->first.fn;
				if (fn)
					*fn = this->_fn;
				return itr->second;
				
			}
            THROW(debug) << "Desired time (" << ts << ") not found "
                         << _data_buffer.begin()->first << ", "
                         << _data_buffer.rbegin()->first;
            return T();
        }

        template<class T>
        T Map<T>::GetData(size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
        {
			boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
			auto itr = Search(fn);
			if (itr != _data_buffer.end())
			{
				IParameter::_ts = itr->first.ts;
				IParameter::_fn = itr->first.fn;
				if (ts)
					*ts = this->_ts;
				return itr->second;

			}
			THROW(debug) << "Desired time (" << ts << ") not found "
				<< _data_buffer.begin()->first << ", "
				<< _data_buffer.rbegin()->first;
			return T();
        }

        template<class T>
        bool Map<T>::UpdateDataImpl(const T& data_, boost::optional<mo::time_t> ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs)
        {
            {
                boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                if(fn)
                    IParameter::_fn = *fn;
                else
                    ++IParameter::_fn;
                _data_buffer[{ts,IParameter::_fn}] = data_;
                IParameter::_modified = true;
            }
            IParameter::OnUpdate(ctx);
            return true;
        }



        template<class T> bool Map<T>::Update(IParameter* other, Context* ctx)
        {
            auto typedParameter = dynamic_cast<ITypedParameter<T>*>(other);
            if (typedParameter)
            {
                auto ptr = typedParameter->GetDataPtr();
                if (ptr)
                {
                    {
                        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
						_data_buffer[{typedParameter->GetTimestamp(), typedParameter->GetFrameNumber()}] = *ptr;
                        IParameter::_modified = true;
                    }
                    IParameter::OnUpdate(ctx);
                }
            }
            return false;
        }
        template<class T> void Map<T>::SetFrameBufferSize(size_t size)
        {

        }
        template<class T> void Map<T>::SetTimestampSize(mo::time_t size)
        {

        }
        template<class T> size_t Map<T>::GetSize()
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            return _data_buffer.size();
        }
        template<class T>
        bool Map<T>::GetTimestampRange(mo::time_t& start, mo::time_t& end)
        {
            if (_data_buffer.size())
            {
                boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                if(_data_buffer.begin()->first.ts && _data_buffer.rbegin()->first.ts)
                {
                    start = *_data_buffer.begin()->first.ts;
                    end = *_data_buffer.rbegin()->first.ts;
                    return true;
                }
            }
            return false;
        }

        template<class T> bool Map<T>::GetFrameNumberRange(size_t& start, size_t& end)
        {
            if (_data_buffer.size())
            {
                boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                start = _data_buffer.begin()->first.fn;
                end = _data_buffer.rbegin()->first.fn;
                return true;
            }
            return false;
        }
		template<class T>
		typename std::map<SequenceKey, T>::iterator  Map<T>::Search(boost::optional<mo::time_t> ts)
		{
			if (_data_buffer.size() == 0)
				return _data_buffer.end();
			if (!ts)
			{
				if(_data_buffer.size())
					return _data_buffer.rbegin().base();
                return _data_buffer.end();
			}
			return _data_buffer.find(*ts);
		}
		template<class T>
		typename std::map<SequenceKey, T>::iterator Map<T>::Search(size_t fn)
		{
			if (_data_buffer.size() == 0)
				return _data_buffer.end();
			return _data_buffer.find(fn);
		}
        template<class T> std::shared_ptr<IParameter> Map<T>::DeepCopy() const
        {
            auto ptr = new Map<T>(IParameter::_name);
            ptr->_data_buffer = this->_data_buffer;
            return std::shared_ptr<IParameter>(ptr);
        }
        template<class T> void Map<T>::onInputUpdate(Context* ctx, IParameter* param)
        {
            T* data = this->input->GetDataPtr();
            if(data)
                //ITypedParameter<T>::UpdateData(_data = *data, _timestamp = this->input->GetTimestamp(), _context = ctx);
                UpdateDataImpl(*data, this->input->GetTimestamp(), ctx, this->input->GetFrameNumber(), this->input->GetCoordinateSystem());
        }
    }
}
