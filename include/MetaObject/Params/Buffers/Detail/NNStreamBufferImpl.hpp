#pragma once

namespace mo
{
namespace Buffer
{

template<class T>
NNStreamBuffer<T>::NNStreamBuffer(const std::string& name):
    ITParam<T>(name, Buffer_e),
    StreamBuffer<T>(name)
{

}

template<class T>
typename std::map<SequenceKey, typename ITParam<T>::Storage_t>::iterator NNStreamBuffer<T>::search(OptionalTime_t ts){
	if (!ts){ // default timestamp passed in, get newest value
		if (this->_data_buffer.size())
			return this->_data_buffer.rbegin().base();
		return this->_data_buffer.end();
	}else{
		auto upper = this->_data_buffer.upper_bound(*ts);
		auto lower = this->_data_buffer.lower_bound(*ts);
		if (upper != this->_data_buffer.end() && lower != this->_data_buffer.end()){
			if (*upper->first.ts - *ts < *lower->first.ts - *ts){
				return upper;
			}else{
				return lower;
			}
		}
		else if (lower != this->_data_buffer.end()){
			return lower;
		}
		else if (upper != this->_data_buffer.end()){
			return upper;
		}
	}
	return this->_data_buffer.end();
}
template<class T>
typename std::map<SequenceKey, typename ITParam<T>::Storage_t>::iterator NNStreamBuffer<T>::search(size_t fn){
	auto upper = this->_data_buffer.upper_bound(fn);
	auto lower = this->_data_buffer.lower_bound(fn);
	if (upper != this->_data_buffer.end() && lower != this->_data_buffer.end()){
		if (upper->first.fn - fn < lower->first.fn - fn){
			return upper;
		}else{
			return lower;
		}
	}else if (lower != this->_data_buffer.end()){
		return lower;
	}else if (upper != this->_data_buffer.end()){
		return upper;
	}
	return this->_data_buffer.end();
}

template<class T>
bool NNStreamBuffer<T>::getData(Storage_t& data, const OptionalTime_t& ts, Context* ctx, size_t* fn_){
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    auto itr = search(ts);
    if(itr != this->_data_buffer.end()){
        this->_current_timestamp = itr->first.ts;
        this->_current_frame_number = itr->first.fn;
        if(fn_) *fn_ = this->_current_frame_number;
        this->prune();
        data = itr->second;
        return true;
    }
    return false;
}

template<class T>
bool NNStreamBuffer<T>::getData(Storage_t& data, size_t fn, Context* ctx, OptionalTime_t* ts_) {
    mo::Mutex_t::scoped_lock lock(IParam::mtx());
    auto itr = search(fn);
    if (itr != this->_data_buffer.end()) {
        this->_current_timestamp = itr->first.ts;
        this->_current_frame_number = itr->first.fn;
        if (ts_) *ts_ = this->_current_timestamp;
        this->prune();
        data = itr->second;
        return true;
    }
    return false;
}


}
}
