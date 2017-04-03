#pragma once

namespace mo
{
namespace Buffer
{

template<class T>
NNStreamBuffer<T>::NNStreamBuffer(const std::string& name):
    ITypedParameter<T>(name, Buffer_e),
    StreamBuffer<T>(name)
{

}

template<class T>
typename std::map<SequenceKey, T>::iterator NNStreamBuffer<T>::Search(boost::optional<mo::time_t> ts)
{
	if (!ts) // default timestamp passed in, get newest value
	{
		if (this->_data_buffer.size())
			return this->_data_buffer.rbegin().base();
		return this->_data_buffer.end();
	}
	else
	{
		auto upper = this->_data_buffer.upper_bound(*ts);
		auto lower = this->_data_buffer.lower_bound(*ts);

		if (upper != this->_data_buffer.end() && lower != this->_data_buffer.end())
		{
			if (*upper->first.ts - *ts < *lower->first.ts - *ts)
			{
				return upper;
			}
			else
			{
				return lower;
			}
		}
		else if (lower != this->_data_buffer.end())
		{
			return lower;
		}
		else if (upper != this->_data_buffer.end())
		{
			return upper;
		}
	}
	return this->_data_buffer.end();
}
template<class T>
typename std::map<SequenceKey, T>::iterator NNStreamBuffer<T>::Search(size_t fn)
{
	auto upper = this->_data_buffer.upper_bound(fn);
	auto lower = this->_data_buffer.lower_bound(fn);

	if (upper != this->_data_buffer.end() && lower != this->_data_buffer.end())
	{
		if (upper->first.fn - fn < lower->first.fn - fn)
		{
			return upper;
		}
		else
		{
			return lower;
		}
	}
	else if (lower != this->_data_buffer.end())
	{
		return lower;
	}
	else if (upper != this->_data_buffer.end())
	{
		return upper;
	}
	return this->_data_buffer.end();
}

template<class T>
T* NNStreamBuffer<T>::GetDataPtr(boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
	auto itr = Search(ts);
	if (itr != this->_data_buffer.end())
	{
		this->_current_timestamp = itr->first.ts;
		this->_current_frame_number = itr->first.fn;
		if (fn)
			*fn = this->_current_frame_number;
		this->prune();
		return &itr->second;
	}
	return nullptr;
}

template<class T>
T* NNStreamBuffer<T>::GetDataPtr(size_t fn_, Context* ctx_, boost::optional<mo::time_t>* ts_)
{
	boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
	auto itr = Search(fn_);
	if (itr != this->_data_buffer.end())
	{
		this->_current_timestamp = itr->first.ts;
		this->_current_frame_number = itr->first.fn;
		if (ts_)
			*ts_ = this->_current_timestamp;
		this->prune();
		return &itr->second;
	}
	return nullptr;
}

template<class T>
T NNStreamBuffer<T>::GetData(boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
	auto itr = Search(ts);
	if (itr != this->_data_buffer.end())
	{
		this->_current_timestamp = itr->first.ts;
		this->_current_frame_number = itr->first.fn;
		if (fn)
			*fn = this->_current_frame_number;
		this->prune();
		return itr->second;
	}
	THROW(warning) << "Unable to find data near timestamp " << *ts;
    return T();
}

template<class T>
T NNStreamBuffer<T>::GetData(size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
{
	boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
	auto itr = Search(fn);
	if (itr != this->_data_buffer.end())
	{
		this->_current_timestamp = itr->first.ts;
		this->_current_frame_number = itr->first.fn;
		if (ts)
			*ts = this->_current_timestamp;
		this->prune();
		return itr->second;
	}
	THROW(warning) << "Unable to find data near timestamp " << *ts;
	return T();
}

template<class T>
bool NNStreamBuffer<T>::GetData(T& value, boost::optional<mo::time_t> ts, Context* ctx, size_t* fn)
{
	boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
	auto itr = Search(ts);
	if (itr != this->_data_buffer.end())
	{
		this->_current_timestamp = itr->first.ts;
		this->_current_frame_number = itr->first.fn;
		if (fn)
			*fn = this->_current_frame_number;
		this->prune();
		value = itr->second;
		return true;
	}
	return false;
}
template<class T>
bool NNStreamBuffer<T>::GetData(T& value, size_t fn, Context* ctx, boost::optional<mo::time_t>* ts)
{
	boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
	auto itr = Search(fn);
	if (itr != this->_data_buffer.end())
	{
		this->_current_timestamp = itr->first.ts;
		this->_current_frame_number = itr->first.fn;
		if (ts)
			*ts = this->_current_timestamp;
		this->prune();
		value = itr->second;
		return true;
	}
	return false;
}

}
}
