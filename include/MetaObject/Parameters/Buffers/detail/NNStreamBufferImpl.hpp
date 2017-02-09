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
T* NNStreamBuffer<T>::GetDataPtr(long long ts, Context* ctx)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if (ts == -1 && this->_data_buffer.size())
    {
        return &(this->_data_buffer.rbegin()->second);
    }else
    {
        auto upper = this->_data_buffer.upper_bound(ts);
        auto lower = this->_data_buffer.lower_bound(ts);
        if(upper != this->_data_buffer.end() && lower != this->_data_buffer.end())
        {
            if(upper->first - ts < lower->first - ts)
            {
                return &upper->second;
            }else
            {
                return &lower->second;
            }
        }else if(lower != this->_data_buffer.end())
        {
            return &lower->second;
        }else if(upper != this->_data_buffer.end())
        {
            return &upper->second;
        }else return nullptr;
    }
    return nullptr;
}

template<class T>
bool NNStreamBuffer<T>::GetData(T& value, long long ts, Context* ctx)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if (ts == -1 && this->_data_buffer.size())
    {
        value = this->_data_buffer.rbegin()->second;
        return true;
    }else
    {
        auto upper = this->_data_buffer.upper_bound(ts);
        auto lower = this->_data_buffer.lower_bound(ts);
        if(upper != this->_data_buffer.end() && lower != this->_data_buffer.end())
        {
            if(upper->first - ts < lower->first - ts)
            {
                value = upper->second;
                return true;
            }else
            {
                value = lower->second;
                return true;
            }
        }else if(lower != this->_data_buffer.end())
        {
            value = lower->second;
            return true;
        }else if(upper != this->_data_buffer.end())
        {
            value = upper->second;
        }else return false;
    }
    return false;
}

template<class T>
T NNStreamBuffer<T>::GetData(long long ts, Context* ctx)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if (ts == -1 && this->_data_buffer.size())
    {
        return this->_data_buffer.rbegin()->second;
    }else
    {
        auto upper = this->_data_buffer.upper_bound(ts);
        auto lower = this->_data_buffer.lower_bound(ts);
        if(upper != this->_data_buffer.end() && lower != this->_data_buffer.end())
        {
            if(upper->first - ts < lower->first - ts)
            {
                return upper->second;
            }else
            {
                return  lower->second;
            }
        }else if(lower != this->_data_buffer.end())
        {
            return lower->second;
        }else if(upper != this->_data_buffer.end())
        {
            return upper->second;
        }else THROW(warning) << "Unable to find data near timestamp "  << ts;
    }
    return T();
}

}
}
